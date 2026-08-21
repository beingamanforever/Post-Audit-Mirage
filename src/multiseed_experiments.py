from __future__ import annotations

import html
import json
import math
import os
import random
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import fmean, pvariance

from .decision_methods import ALPHA, METHOD_NAMES, Decision, build_methods
from .environment_lifecycles import (
    FAMILIES,
    LIFECYCLE_LENGTH,
    SCENARIOS,
    ProtectedRoundTruth,
    RealizedLifecycle,
    realize_lifecycle,
)
from .experiments import _clopper_pearson
from .paired_inference import METHODS as INFERENCE_METHODS
from .paired_inference import LifecycleOutcome, run_paired_inference

ENVIRONMENTS_PER_CELL = 100
VARIATION_ENVIRONMENTS_PER_CELL = 10
VARIATION_LIFECYCLE_SEEDS = 3
VARIATION_METHOD_SEEDS = 4
MAX_WORKERS = 32
SHARP_VERSION = "0.35.3"
_SEED_LIMIT = 2**31
_METRICS = (
    "harmful_lifecycle",
    "genuine_acceptance",
    "final_utility",
    "first_harmful_acceptance",
    "abstention",
    "worst_group_harm",
    "deployed_count",
)


@dataclass(frozen=True)
class _BenchmarkUnit:
    family: str
    scenario: str
    environment_index: int
    environment_seed: int
    lifecycle_seed: int
    method_seed: int
    variation_seeds: tuple[tuple[int, tuple[int, ...]], ...]


def run_multiseed_experiments(
    output_dir: Path,
    artifacts_dir: Path,
    data_dir: Path,
    *,
    seed: int = 20260821,
    workers: int = 1,
    node_modules: Path | None = None,
) -> dict[str, object]:
    """Run and atomically publish the fixed three-family multi-seed benchmark."""
    output_dir, artifacts_dir, planning_templates, node_modules = _validate_settings(
        output_dir, artifacts_dir, data_dir, seed, workers, node_modules
    )
    _preflight_renderer(node_modules)
    units = _seed_plan(seed)
    realized = _realize_units(units, planning_templates, workers)

    primary_rows: list[dict[str, object]] = []
    variation_rows: list[dict[str, object]] = []
    inference_outcomes: list[LifecycleOutcome] = []
    for unit, lifecycle in zip(units, realized, strict=True):
        rows = _score_methods(
            lifecycle,
            unit,
            unit.lifecycle_seed,
            unit.method_seed,
            panel="primary",
        )
        primary_rows.extend(rows)
        inference_outcomes.extend(_inference_outcomes(rows))
        for lifecycle_repeat, (lifecycle_seed, method_seeds) in enumerate(
            unit.variation_seeds
        ):
            for method_repeat, method_seed in enumerate(method_seeds):
                nested = _score_methods(
                    lifecycle,
                    unit,
                    lifecycle_seed,
                    method_seed,
                    panel="variation",
                )
                for row in nested:
                    row["lifecycle_repeat"] = lifecycle_repeat
                    row["method_repeat"] = method_repeat
                variation_rows.extend(nested)

    inference = run_paired_inference(inference_outcomes)
    cell_summaries = _cell_summaries(primary_rows)
    seed_variation = _seed_variation(variation_rows)
    summary = _summary(seed, units, cell_summaries, seed_variation, inference)
    svg = _comparison_svg(cell_summaries, inference)
    payloads: dict[Path, str | bytes] = {
        output_dir / "multiseed_outcomes.jsonl": _jsonl(primary_rows),
        output_dir / "multiseed_inference.json": _json(inference),
        output_dir / "multiseed_summary.json": _json(summary),
        output_dir / "multiseed_variation.jsonl": _jsonl(variation_rows),
        artifacts_dir / "multiseed_method_comparison.svg": svg,
        artifacts_dir / "multiseed_method_comparison.png": _render_png(
            svg, node_modules
        ),
    }
    _publish(payloads)
    return summary


def _validate_settings(
    output_dir: Path,
    artifacts_dir: Path,
    data_dir: Path,
    seed: int,
    workers: int,
    node_modules: Path | None,
) -> tuple[Path, Path, Path, Path | None]:
    paths = {
        "output_dir": output_dir,
        "artifacts_dir": artifacts_dir,
        "data_dir": data_dir,
    }
    for name, path in paths.items():
        if not isinstance(path, Path):
            raise TypeError(f"{name} must be a pathlib.Path")
        if path.exists() and not path.is_dir():
            raise ValueError(f"{name} must be a directory")
    if output_dir.resolve() == artifacts_dir.resolve():
        raise ValueError("output_dir and artifacts_dir must be different directories")
    if not data_dir.is_dir():
        raise ValueError(f"data_dir does not exist: {data_dir}")
    planning_templates = data_dir / "planning_templates.json"
    if not planning_templates.is_file():
        raise ValueError(f"missing planning templates: {planning_templates}")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer from 0 through 9223372036854775807")
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= MAX_WORKERS
    ):
        raise ValueError(f"workers must be an integer from 1 through {MAX_WORKERS}")
    if node_modules is not None:
        if not isinstance(node_modules, Path):
            raise TypeError("node_modules must be a pathlib.Path")
        if not (node_modules / "sharp").is_dir():
            raise ValueError("node_modules must contain sharp")
    targets = (
        output_dir / "multiseed_outcomes.jsonl",
        output_dir / "multiseed_inference.json",
        output_dir / "multiseed_summary.json",
        output_dir / "multiseed_variation.jsonl",
        artifacts_dir / "multiseed_method_comparison.svg",
        artifacts_dir / "multiseed_method_comparison.png",
    )
    if any(target.exists() and not target.is_file() for target in targets):
        raise ValueError("every benchmark output target must be a regular file")
    return output_dir, artifacts_dir, planning_templates, node_modules


def _seed_plan(seed: int) -> tuple[_BenchmarkUnit, ...]:
    total_environments = len(FAMILIES) * len(SCENARIOS) * ENVIRONMENTS_PER_CELL
    total_variation_lifecycles = (
        len(FAMILIES)
        * len(SCENARIOS)
        * VARIATION_ENVIRONMENTS_PER_CELL
        * VARIATION_LIFECYCLE_SEEDS
    )
    total_seeds = 3 * total_environments + total_variation_lifecycles * (
        1 + VARIATION_METHOD_SEEDS
    )
    source = iter(
        random.Random(f"multiseed:{seed}").sample(range(_SEED_LIMIT), total_seeds)
    )
    units = []
    for family in FAMILIES:
        for scenario in SCENARIOS:
            for environment_index in range(ENVIRONMENTS_PER_CELL):
                environment_seed = next(source)
                lifecycle_seed = next(source)
                method_seed = next(source)
                variation_seeds = ()
                if environment_index < VARIATION_ENVIRONMENTS_PER_CELL:
                    variation_seeds = tuple(
                        (
                            next(source),
                            tuple(next(source) for _ in range(VARIATION_METHOD_SEEDS)),
                        )
                        for _ in range(VARIATION_LIFECYCLE_SEEDS)
                    )
                units.append(
                    _BenchmarkUnit(
                        family,
                        scenario,
                        environment_index,
                        environment_seed,
                        lifecycle_seed,
                        method_seed,
                        variation_seeds,
                    )
                )
    try:
        next(source)
    except StopIteration:
        pass
    else:
        raise AssertionError("seed plan did not consume every allocated seed")
    _validate_seed_plan(units, total_seeds)
    return tuple(units)


def _validate_seed_plan(units: list[_BenchmarkUnit], expected: int) -> None:
    environment = [unit.environment_seed for unit in units]
    lifecycle = [unit.lifecycle_seed for unit in units]
    method = [unit.method_seed for unit in units]
    for unit in units:
        for lifecycle_seed, method_seeds in unit.variation_seeds:
            lifecycle.append(lifecycle_seed)
            method.extend(method_seeds)
    seeds = environment + lifecycle + method
    if len(seeds) != expected or len(set(seeds)) != expected:
        raise AssertionError("seed roles must be globally unique")


def _realize_units(
    units: tuple[_BenchmarkUnit, ...], planning_templates: Path, workers: int
) -> Iterable[RealizedLifecycle]:
    tasks = tuple(
        (unit.family, unit.scenario, unit.environment_seed, planning_templates)
        for unit in units
    )
    if workers == 1:
        yield from (_realize_task(task) for task in tasks)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(_realize_task, tasks, chunksize=1)


def _realize_task(task: tuple[str, str, int, Path]) -> RealizedLifecycle:
    family, scenario, environment_seed, planning_templates = task
    return realize_lifecycle(
        family,
        scenario,
        environment_seed,
        planning_templates=planning_templates,
    )


def _score_methods(
    lifecycle: RealizedLifecycle,
    unit: _BenchmarkUnit,
    lifecycle_seed: int,
    method_seed: int,
    *,
    panel: str,
) -> list[dict[str, object]]:
    rounds = list(zip(lifecycle.public_rounds, lifecycle.protected_truth, strict=True))
    random.Random(lifecycle_seed).shuffle(rounds)
    answers = {
        (public.update.family, public.update.update_id): truth.safe_to_deploy
        for public, truth in rounds
    }
    methods = build_methods(answers, alpha=ALPHA, seed=method_seed)
    rows = []
    for method in methods:
        decisions = []
        for public, truth in rounds:
            monitor = truth.monitor if method.name == "monitor" else None
            decisions.append(method.decide(public.update, monitor))
        rows.append(
            _outcome_row(
                unit,
                lifecycle_seed,
                method_seed,
                method.name,
                decisions,
                [truth for _, truth in rounds],
                panel,
            )
        )
    return rows


def _outcome_row(
    unit: _BenchmarkUnit,
    lifecycle_seed: int,
    method_seed: int,
    method: str,
    decisions: list[Decision],
    truths: list[ProtectedRoundTruth],
    panel: str,
) -> dict[str, object]:
    deployed_count = 0
    safe_deployed = 0
    safe_opportunities = 0
    harmful_lifecycle = False
    first_harmful_acceptance = None
    utility = 0
    monitor_unavailable_rounds = 0
    group_totals: dict[str, list[Fraction]] = {}
    for round_index, (decision, truth) in enumerate(
        zip(decisions, truths, strict=True), start=1
    ):
        for group in truth.group_harm:
            totals = group_totals.setdefault(
                group.group, [Fraction(), Fraction(), Fraction()]
            )
            totals[2] += group.exposure_mass
        deploy = decision.deploy
        deployed_count += deploy
        monitor_unavailable_rounds += decision.reason == "live-style stream unavailable"
        if truth.safe_to_deploy:
            safe_opportunities += 1
            safe_deployed += deploy
            utility += deploy
        elif deploy:
            harmful_lifecycle = True
            if first_harmful_acceptance is None:
                first_harmful_acceptance = round_index
            utility -= 1
        if deploy:
            for group in truth.group_harm:
                totals = group_totals[group.group]
                totals[0] += group.harm_mass
                totals[1] += group.exposure_mass
    group_harm = {
        group: {
            "eligible_exposure_mass": _fraction_text(eligible_exposure),
            "exposure_mass": _fraction_text(exposure),
            "harm_mass": _fraction_text(harm),
            "rate": _fraction_text(harm / exposure) if exposure else None,
        }
        for group, (harm, exposure, eligible_exposure) in sorted(group_totals.items())
    }
    group_rates = [
        float(harm / exposure)
        for harm, exposure, _ in group_totals.values()
        if exposure
    ]
    return {
        "abstention": deployed_count == 0,
        "deployed_count": deployed_count,
        "environment_index": unit.environment_index,
        "environment_seed": unit.environment_seed,
        "family": unit.family,
        "final_utility": utility / LIFECYCLE_LENGTH,
        "first_harmful_acceptance": first_harmful_acceptance,
        "genuine_acceptance": (
            safe_deployed / safe_opportunities if safe_opportunities else None
        ),
        "group_harm": group_harm,
        "harmful_lifecycle": harmful_lifecycle,
        "lifecycle_seed": lifecycle_seed,
        "method": method,
        "method_seed": method_seed,
        "monitor_unavailable_rounds": monitor_unavailable_rounds,
        "panel": panel,
        "safe_accepted": safe_deployed,
        "safe_opportunities": safe_opportunities,
        "scenario": unit.scenario,
        "worst_group_harm": max(group_rates) if group_rates else None,
    }


def _inference_outcomes(rows: list[dict[str, object]]) -> list[LifecycleOutcome]:
    return [
        LifecycleOutcome(
            family=str(row["family"]),
            scenario=str(row["scenario"]),
            environment_seed=int(row["environment_seed"]),
            lifecycle_seed=int(row["lifecycle_seed"]),
            method_seed=int(row["method_seed"]),
            method=str(row["method"]),
            harmful_lifecycle=bool(row["harmful_lifecycle"]),
            genuine_acceptance=(
                float(row["genuine_acceptance"])
                if row["genuine_acceptance"] is not None
                else 0.0
            ),
            final_utility=float(row["final_utility"]),
        )
        for row in rows
        if row["method"] in INFERENCE_METHODS
    ]


def _cell_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for family in FAMILIES:
        for scenario in SCENARIOS:
            for method in METHOD_NAMES:
                cohort = [
                    row
                    for row in rows
                    if row["family"] == family
                    and row["scenario"] == scenario
                    and row["method"] == method
                ]
                if len(cohort) != ENVIRONMENTS_PER_CELL:
                    raise AssertionError("primary cell has the wrong lifecycle count")
                summary: dict[str, object] = {
                    "environments": len(cohort),
                    "family": family,
                    "method": method,
                    "scenario": scenario,
                }
                for metric in _METRICS:
                    values = _observed_values(row[metric] for row in cohort)
                    summary[metric] = fmean(values) if values else None
                    summary[f"{metric}_observed"] = len(values)
                harmful_count = sum(bool(row["harmful_lifecycle"]) for row in cohort)
                harm_lower, harm_upper = _clopper_pearson(harmful_count, len(cohort))
                summary.update(
                    {
                        "harmful_lifecycle_ci_lower": harm_lower,
                        "harmful_lifecycle_ci_upper": harm_upper,
                        "harmful_lifecycle_successes": harmful_count,
                        "harmful_lifecycle_trials": len(cohort),
                        "harmful_lifecycle_uncertainty": (
                            "exact_one_sided_95_clopper_pearson"
                        ),
                    }
                )
                summaries.append(summary)
    return summaries


def _seed_variation(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for family in FAMILIES:
        for scenario in SCENARIOS:
            for method in METHOD_NAMES:
                cohort = [
                    row
                    for row in rows
                    if row["family"] == family
                    and row["scenario"] == scenario
                    and row["method"] == method
                ]
                expected = (
                    VARIATION_ENVIRONMENTS_PER_CELL
                    * VARIATION_LIFECYCLE_SEEDS
                    * VARIATION_METHOD_SEEDS
                )
                if len(cohort) != expected:
                    raise AssertionError("variation cell has the wrong lifecycle count")
                for metric in _METRICS:
                    environment_means: list[float] = []
                    within_environment: list[float] = []
                    lifecycle_variance: list[float] = []
                    method_variance: list[float] = []
                    for environment_index in range(VARIATION_ENVIRONMENTS_PER_CELL):
                        environment_rows = [
                            row
                            for row in cohort
                            if row["environment_index"] == environment_index
                        ]
                        values = _observed_values(
                            row[metric] for row in environment_rows
                        )
                        if not values:
                            continue
                        environment_means.append(fmean(values))
                        within_environment.append(pvariance(values))
                        lifecycle_means: list[float] = []
                        for lifecycle_repeat in range(VARIATION_LIFECYCLE_SEEDS):
                            lifecycle_values = _observed_values(
                                row[metric]
                                for row in environment_rows
                                if row["lifecycle_repeat"] == lifecycle_repeat
                            )
                            if not lifecycle_values:
                                continue
                            lifecycle_means.append(fmean(lifecycle_values))
                            method_variance.append(pvariance(lifecycle_values))
                        if lifecycle_means:
                            lifecycle_variance.append(pvariance(lifecycle_means))
                    summaries.append(
                        {
                            "across_environment_variance": (
                                pvariance(environment_means)
                                if environment_means
                                else None
                            ),
                            "environments": VARIATION_ENVIRONMENTS_PER_CELL,
                            "family": family,
                            "lifecycle_seed_variance": (
                                fmean(lifecycle_variance)
                                if lifecycle_variance
                                else None
                            ),
                            "method": method,
                            "method_seed_variance": (
                                fmean(method_variance) if method_variance else None
                            ),
                            "metric": metric,
                            "nested_repetitions_per_environment": (
                                VARIATION_LIFECYCLE_SEEDS * VARIATION_METHOD_SEEDS
                            ),
                            "observed_environments": len(environment_means),
                            "observed_outcomes": sum(
                                row[metric] is not None for row in cohort
                            ),
                            "scenario": scenario,
                            "within_environment_variance": (
                                fmean(within_environment)
                                if within_environment
                                else None
                            ),
                        }
                    )
    return summaries


def _summary(
    seed: int,
    units: tuple[_BenchmarkUnit, ...],
    cells: list[dict[str, object]],
    variation: list[dict[str, object]],
    inference: dict[str, object],
) -> dict[str, object]:
    environment_seeds = {unit.environment_seed for unit in units}
    lifecycle_seeds = {unit.lifecycle_seed for unit in units}
    method_seeds = {unit.method_seed for unit in units}
    for unit in units:
        for lifecycle_seed, nested_method_seeds in unit.variation_seeds:
            lifecycle_seeds.add(lifecycle_seed)
            method_seeds.update(nested_method_seeds)
    return {
        "cell_summaries": cells,
        "config": {
            "environment_seeds": len(environment_seeds),
            "environments_per_cell": ENVIRONMENTS_PER_CELL,
            "families": list(FAMILIES),
            "globally_unique_seed_roles": True,
            "lifecycle_length": LIFECYCLE_LENGTH,
            "lifecycle_seeds": len(lifecycle_seeds),
            "method_seeds": len(method_seeds),
            "methods": list(METHOD_NAMES),
            "primary_lifecycles": len(units),
            "primary_outcome_rows": len(units) * len(METHOD_NAMES),
            "primary_update_rounds": len(units) * LIFECYCLE_LENGTH,
            "png_renderer": f"sharp_{SHARP_VERSION}",
            "root_seed": seed,
            "scenarios": list(SCENARIOS),
            "seed_roles": {
                "environment": "exact generated environment semantics",
                "lifecycle": "permutation of one fixed realized lifecycle",
                "method": "fresh method state for one paired method run",
            },
            "variation_environments_per_cell": VARIATION_ENVIRONMENTS_PER_CELL,
            "variation_lifecycle_seeds": VARIATION_LIFECYCLE_SEEDS,
            "variation_method_seeds": VARIATION_METHOD_SEEDS,
            "variation_lifecycles": (
                len(FAMILIES)
                * len(SCENARIOS)
                * VARIATION_ENVIRONMENTS_PER_CELL
                * VARIATION_LIFECYCLE_SEEDS
                * VARIATION_METHOD_SEEDS
            ),
            "variation_outcome_rows": (
                len(FAMILIES)
                * len(SCENARIOS)
                * VARIATION_ENVIRONMENTS_PER_CELL
                * VARIATION_LIFECYCLE_SEEDS
                * VARIATION_METHOD_SEEDS
                * len(METHOD_NAMES)
            ),
        },
        "inference_overall_label": inference["overall_label"],
        "seed_variation": variation,
    }


def _comparison_svg(
    cells: list[dict[str, object]], inference: dict[str, object]
) -> str:
    width, height = 1500, 920
    left, top, panel_width = 230, 210, 350
    panels = (
        ("harmful_lifecycle", "Lifecycle harm", "lower is better", 0.0, 1.0),
        ("genuine_acceptance", "Safe acceptance", "higher is better", 0.0, 1.0),
        ("final_utility", "Normalized utility", "higher is better", -1.0, 1.0),
    )
    labels = {
        "always_hold": "Always hold",
        "greedy": "Greedy",
        "fixed_threshold": "Fixed threshold",
        "shrinking_budget": "Shrinking budget",
        "addis_spending": "ADDIS spending",
        "online_closed_e": "Online closed e-test",
        "pace_reset": "PACE reset",
        "reused_holdout": "Reusable holdout",
        "sgm_transferred": "Transferred SGM",
        "monitor": "Live monitor",
        "oracle": "True answer",
    }
    scenario_sets = {
        "harmful_lifecycle": {"null_only", "mixed"},
        "genuine_acceptance": {"all_good", "mixed"},
        "final_utility": set(SCENARIOS),
    }
    colors = {
        "online_closed_e": "#2F6FED",
        "addis_spending": "#7C5CE7",
        "shrinking_budget": "#0F9D8A",
        "oracle": "#D9485F",
    }
    claim_labels = {
        "best": "PASSES FROZEN TESTS VS BOTH",
        "superior_to_addis_spending": "PASSES FROZEN TEST VS ADDIS ONLY",
        "superior_to_shrinking_budget": "PASSES FROZEN TEST VS SHRINKING ONLY",
        "no_superiority_established": "NO SUPERIORITY ESTABLISHED",
    }
    overall_label = str(inference["overall_label"])
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        '<title id="title">Multi-seed lifecycle method comparison</title>',
        '<desc id="description">Primary outcomes from 900 independent exact environment lifecycles across AuthGate, ConstraintPlan, and BatchTriage.</desc>',
        '<rect width="1500" height="920" fill="#F7F9FC"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}</style>',
        _svg_text(55, 62, "Multi-seed lifecycle benchmark", 31, "#183153", 600),
        _svg_text(
            55,
            96,
            "900 independent 50-update lifecycles  |  three exact environment families  |  paired methods",
            15,
            "#68768A",
        ),
        _svg_rect(1110, 42, 335, 42, "#E8EEF9", 21),
        _svg_text(
            1277,
            69,
            claim_labels.get(overall_label, overall_label.replace("_", " ").upper()),
            12,
            "#183153",
            600,
            "middle",
        ),
        '<line x1="55" y1="127" x2="1445" y2="127" stroke="#DDE3EC"/>',
    ]
    for method_index, method in enumerate(METHOD_NAMES):
        y = top + method_index * 55
        if method in INFERENCE_METHODS:
            svg.append(_svg_rect(38, y - 23, 1407, 44, "#EDF3FE", 8))
        svg.append(_svg_text(55, y + 5, labels[method], 13, "#243247", 500))
    for panel_index, (metric, title, note, low, high) in enumerate(panels):
        x = left + panel_index * 410
        svg.extend(
            (
                _svg_text(x, 158, title, 19, "#183153", 600),
                _svg_text(x, 181, note, 12, "#68768A"),
                f'<line x1="{x}" y1="195" x2="{x + panel_width}" y2="195" stroke="#AAB6C7"/>',
            )
        )
        for tick_index in range(5):
            value = low + (high - low) * tick_index / 4
            tick_x = x + panel_width * tick_index / 4
            svg.extend(
                (
                    f'<line x1="{tick_x:.1f}" y1="191" x2="{tick_x:.1f}" y2="810" stroke="#DDE3EC"/>',
                    _svg_text(
                        tick_x,
                        832,
                        f"{value:.2f}".rstrip("0").rstrip("."),
                        11,
                        "#68768A",
                        anchor="middle",
                    ),
                )
            )
        for method_index, method in enumerate(METHOD_NAMES):
            cohort = [
                float(row[metric])
                for row in cells
                if row["method"] == method and row["scenario"] in scenario_sets[metric]
            ]
            value = fmean(cohort)
            point_x = x + (value - low) / (high - low) * panel_width
            y = top + method_index * 55
            color = colors.get(method, "#68768A")
            svg.extend(
                (
                    f'<circle cx="{point_x:.2f}" cy="{y}" r="6" fill="{color}" stroke="#FFFFFF" stroke-width="2"/>',
                    _svg_text(
                        point_x + (10 if point_x < x + panel_width - 48 else -10),
                        y - 10,
                        f"{value:.2f}",
                        10,
                        color,
                        600,
                        "start" if point_x < x + panel_width - 48 else "end",
                    ),
                )
            )
    svg.extend(
        (
            '<line x1="55" y1="860" x2="1445" y2="860" stroke="#DDE3EC"/>',
            _svg_text(
                55,
                892,
                "Dots are descriptive averages across prespecified cells. The frozen paired claim passes only versus shrinking budget; family-specific effects vary.",
                13,
                "#68768A",
            ),
            "</svg>\n",
        )
    )
    result = "".join(svg)
    if "\N{EM DASH}" in result:
        raise AssertionError("SVG must not contain em dash characters")
    return result


def _svg_text(
    x: float,
    y: float,
    value: str,
    size: int,
    color: str,
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{html.escape(value)}</text>'
    )


def _svg_rect(
    x: float, y: float, width: float, height: float, fill: str, radius: float
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{fill}"/>'
    )


def _run_sharp(
    script: str,
    node_modules: Path | None,
    *,
    input_data: bytes = b"",
    failure_label: str,
) -> bytes:
    node = shutil.which("node")
    if node is None:
        raise ValueError("PNG publication requires Node.js")
    environment = os.environ.copy()
    if node_modules is not None:
        environment["NODE_PATH"] = str(node_modules)
    result = subprocess.run(
        [node, "-e", script],
        input=input_data,
        env=environment,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        error = result.stderr.decode(errors="replace").strip()
        raise ValueError(f"{failure_label}: {error}")
    return result.stdout


def _preflight_renderer(node_modules: Path | None) -> None:
    script = (
        'const sharp=require("sharp");'
        f'if(sharp.versions.sharp!=="{SHARP_VERSION}")'
        '{throw new Error("expected sharp '
        f"{SHARP_VERSION}"
        '");}'
        "process.stdout.write(sharp.versions.sharp);"
    )
    version = _run_sharp(
        script,
        node_modules,
        failure_label="PNG renderer preflight failed",
    ).decode(errors="replace")
    if version != SHARP_VERSION:
        raise ValueError("PNG renderer preflight returned an unexpected version")


def _render_png(svg: str, node_modules: Path | None) -> bytes:
    script = (
        'const fs=require("fs"),sharp=require("sharp");'
        f'if(sharp.versions.sharp!=="{SHARP_VERSION}")'
        '{throw new Error("expected sharp '
        f"{SHARP_VERSION}"
        '");}'
        "sharp(fs.readFileSync(0),{density:144})"
        ".png({compressionLevel:9,adaptiveFiltering:false}).toBuffer()"
        ".then(data=>process.stdout.write(data))"
        ".catch(error=>{console.error(error);process.exit(1)});"
    )
    png = _run_sharp(
        script,
        node_modules,
        input_data=svg.encode(),
        failure_label="PNG renderer failed",
    )
    if not png.startswith(b"\x89PNG\r\n\x1a\n") or len(png) < 24:
        raise ValueError("PNG renderer returned invalid output")
    if struct.unpack(">II", png[16:24]) != (3000, 1840):
        raise ValueError("PNG renderer returned unexpected dimensions")
    return png


def _numeric(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    raise ValueError("outcome metric must be finite and numeric")


def _observed_values(values: Iterable[object]) -> list[float]:
    return [_numeric(value) for value in values if value is not None]


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


def _jsonl(rows: list[dict[str, object]]) -> str:
    return "".join(_json(row) for row in rows)


def _publish(payloads: dict[Path, str | bytes]) -> None:
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    published: set[Path] = set()
    try:
        for target, payload in payloads.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", dir=target.parent
            )
            temporary = Path(temporary_name)
            if isinstance(payload, bytes):
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            else:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            os.chmod(temporary, 0o644)
            staged[target] = temporary
        for target in payloads:
            if target.exists():
                descriptor, backup_name = tempfile.mkstemp(
                    prefix=f".{target.name}.backup.", dir=target.parent
                )
                os.close(descriptor)
                backup = Path(backup_name)
                try:
                    os.replace(target, backup)
                except Exception:
                    backup.unlink(missing_ok=True)
                    raise
                backups[target] = backup
        for target, temporary in staged.items():
            os.replace(temporary, target)
            published.add(target)
    except Exception:
        for target in published:
            if target not in backups:
                target.unlink(missing_ok=True)
        for target, backup in backups.items():
            os.replace(backup, target)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
