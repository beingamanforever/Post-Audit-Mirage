from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

from .decision_methods import ALPHA, build_public_methods
from .environment_lifecycles import (
    FAMILIES,
    LIFECYCLE_LENGTH,
    MatchedLifecyclePair,
    ProtectedRoundTruth,
    PublicUpdateRound,
    realize_matched_pair,
)
from .experiments import MONITOR_SIZES, _clopper_pearson
from .identifiability_plots import (
    identifiability_restoration_svg,
    monitor_sample_complexity_svg,
)
from .identified_range_monitor import monitor_identified_range
from .lifecycle import WORST_GROUP_LIMIT
from .multiseed_experiments import _publish

DEFAULT_SEMANTIC_UNITS = MappingProxyType(
    {
        "authgate_v0": 100,
        "constraint_plan_v0": 100,
        "batch_triage_v0": 72,
    }
)
DEFAULT_EXACT_STREAMS = 20
DEFAULT_CONTROLLED_STREAMS = 500
MAX_WORKERS = 32
CONTROLLED_GAPS = (0.025, 0.05, 0.10)
RARE_PREVALENCES = (0.02, 0.10, 0.50)
RETENTIONS = (1.0, 0.5)
CORRECT_STREAM_FRACTION = 0.80


def run_identifiability_experiments(
    output_dir: Path,
    artifacts_dir: Path,
    data_dir: Path,
    *,
    seed: int = 20260821,
    workers: int = 1,
    semantic_units: Mapping[str, int] = DEFAULT_SEMANTIC_UNITS,
    exact_streams: int = DEFAULT_EXACT_STREAMS,
    controlled_streams: int = DEFAULT_CONTROLLED_STREAMS,
    monitor_sizes: tuple[int, ...] = MONITOR_SIZES,
) -> dict[str, object]:
    """Run matched-world and live-monitor identifiability experiments."""
    planning_templates, units, sizes = _validate_settings(
        output_dir,
        artifacts_dir,
        data_dir,
        seed,
        workers,
        semantic_units,
        exact_streams,
        controlled_streams,
        monitor_sizes,
    )
    pairs = _realize_pairs(units, planning_templates, workers)
    witnesses, method_names, uniqueness = _offline_witnesses(pairs, seed, units)
    expected_witnesses = sum(units.values()) * LIFECYCLE_LENGTH
    if len(witnesses) != expected_witnesses:
        raise AssertionError("matched witness count does not match the seed plan")

    exact_trials = _exact_trials(pairs, exact_streams, sizes, seed)
    controlled_trials = _controlled_trials(controlled_streams, sizes, seed)
    exact_truth = _exact_truth_summary(pairs)
    exact_summary = _summarize_exact(exact_trials, units, exact_streams, sizes)
    controlled_summary = _summarize_controlled(
        controlled_trials, controlled_streams, sizes
    )
    assumptions, invalid_ablations = _assumption_checks(sizes[-1])
    claim_status = _claim_status(exact_summary, assumptions, sizes[-1])
    summary: dict[str, object] = {
        "config": {
            "alpha": ALPHA,
            "controlled_gaps": list(CONTROLLED_GAPS),
            "controlled_streams": controlled_streams,
            "exact_streams_per_world_and_semantic_unit": exact_streams,
            "exact_semantic_success_rule": (
                "at_least_80_percent_of_streams_correct_at_the_look"
            ),
            "harm_threshold": float(WORST_GROUP_LIMIT),
            "monitor_sizes": list(sizes),
            "rare_prevalences": list(RARE_PREVALENCES),
            "retentions": list(RETENTIONS),
            "seed": seed,
            "semantic_units": dict(units),
            "workers": workers,
        },
        "exact_truth_scorer_only": exact_truth,
        "witnesses": {
            "expected_rows": expected_witnesses,
            "observed_rows": len(witnesses),
            "normalized_signature_uniqueness": uniqueness,
            "offline_methods": list(method_names),
            "all_public_observations_shared": True,
            "all_truth_answers_opposite": True,
            "all_offline_decisions_match": True,
        },
        "exact_family": exact_summary,
        "controlled_panel": controlled_summary,
        "claim_status": claim_status,
        "assumption_checks": assumptions,
        "invalid_ablations": invalid_ablations,
        "uncertainty": {
            "exact_family": "descriptive_fixed_semantic_panel_no_population_interval",
            "controlled_panel": "exact_one_sided_95_clopper_pearson_over_independent_monitor_streams",
        },
        "limitations": [
            "Exact-family streams repeat fixed semantic lifecycles and do not estimate prevalence outside the generated family.",
            "The monitor identifies only thresholded group harm under the declared collection design.",
            "No claim is made under outcome-dependent selection, missing required groups, or distribution shift.",
        ],
    }
    trial_rows = exact_trials + controlled_trials
    payloads = {
        output_dir / "identifiability_witnesses.jsonl": _jsonl(witnesses),
        output_dir / "monitor_trials.jsonl": _jsonl(trial_rows),
        output_dir / "identifiability_summary.json": _json(summary),
        artifacts_dir
        / "identifiability_restoration.svg": identifiability_restoration_svg(summary),
        artifacts_dir / "monitor_sample_complexity.svg": monitor_sample_complexity_svg(
            summary
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
    semantic_units: Mapping[str, int],
    exact_streams: int,
    controlled_streams: int,
    monitor_sizes: tuple[int, ...],
) -> tuple[Path, dict[str, int], tuple[int, ...]]:
    for name, path in {
        "output_dir": output_dir,
        "artifacts_dir": artifacts_dir,
        "data_dir": data_dir,
    }.items():
        if not isinstance(path, Path):
            raise TypeError(f"{name} must be a pathlib.Path")
        if path.exists() and not path.is_dir():
            raise ValueError(f"{name} must be a directory")
    if not data_dir.is_dir():
        raise ValueError(f"data_dir does not exist: {data_dir}")
    planning_templates = data_dir / "planning_templates.json"
    if not planning_templates.is_file():
        raise ValueError(f"missing planning templates: {planning_templates}")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer from 0 through 9223372036854775807")
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be an integer from 1 through {MAX_WORKERS}")
    if not isinstance(semantic_units, Mapping) or set(semantic_units) != set(FAMILIES):
        raise ValueError("semantic_units must exactly cover the three families")
    units = dict(semantic_units)
    if any(type(value) is not int or value < 1 for value in units.values()):
        raise ValueError("semantic unit counts must be positive integers")
    for name, value in {
        "exact_streams": exact_streams,
        "controlled_streams": controlled_streams,
    }.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        not isinstance(monitor_sizes, tuple)
        or not monitor_sizes
        or any(type(value) is not int or value < 1 for value in monitor_sizes)
        or tuple(sorted(set(monitor_sizes))) != monitor_sizes
    ):
        raise ValueError(
            "monitor_sizes must be a strictly increasing tuple of integers"
        )
    targets = (
        output_dir / "identifiability_witnesses.jsonl",
        output_dir / "monitor_trials.jsonl",
        output_dir / "identifiability_summary.json",
        artifacts_dir / "identifiability_restoration.svg",
        artifacts_dir / "monitor_sample_complexity.svg",
    )
    if any(target.exists() and not target.is_file() for target in targets):
        raise ValueError("every experiment output target must be a regular file")
    return planning_templates, units, monitor_sizes


def _realize_pairs(
    units: dict[str, int], planning_templates: Path, workers: int
) -> dict[str, tuple[MatchedLifecyclePair, ...]]:
    tasks = tuple(
        (family, semantic_unit, planning_templates)
        for family in FAMILIES
        for semantic_unit in range(units[family])
    )
    if workers == 1:
        realized = tuple(_realize_pair(task) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            realized = tuple(executor.map(_realize_pair, tasks, chunksize=1))
    grouped: dict[str, list[MatchedLifecyclePair]] = {family: [] for family in FAMILIES}
    for (family, _, _), pair in zip(tasks, realized, strict=True):
        grouped[family].append(pair)
    return {family: tuple(grouped[family]) for family in FAMILIES}


def _realize_pair(task: tuple[str, int, Path]) -> MatchedLifecyclePair:
    family, semantic_unit, planning_templates = task
    return realize_matched_pair(
        family, semantic_unit, planning_templates=planning_templates
    )


def _offline_witnesses(
    pairs: dict[str, tuple[MatchedLifecyclePair, ...]],
    seed: int,
    units: dict[str, int],
) -> tuple[list[dict[str, object]], tuple[str, ...], dict[str, dict[str, int]]]:
    witnesses: list[dict[str, object]] = []
    signatures: dict[str, set[tuple[object, ...]]] = {
        family: set() for family in FAMILIES
    }
    structures: dict[str, set[tuple[object, ...]]] = {
        family: set() for family in FAMILIES
    }
    method_names: tuple[str, ...] | None = None
    for family in FAMILIES:
        for semantic_unit, pair in enumerate(pairs[family]):
            if pair.safe.public_rounds is not pair.harmful.public_rounds:
                raise AssertionError("matched worlds must share public observations")
            if not all(
                safe.safe_to_deploy and not harmful.safe_to_deploy
                for safe, harmful in zip(
                    pair.safe.protected_truth,
                    pair.harmful.protected_truth,
                    strict=True,
                )
            ):
                raise AssertionError("matched worlds must have opposite truth answers")
            round_signatures = tuple(
                _method_signature(public) for public in pair.safe.public_rounds
            )
            signatures[family].add(round_signatures)
            structures[family].add(pair.semantic_structure)
            methods_seed = _stable_seed(seed, "offline", family, semantic_unit)
            safe_methods = tuple(
                method
                for method in build_public_methods(alpha=ALPHA, seed=methods_seed)
                if method.name != "monitor"
            )
            harmful_methods = tuple(
                method
                for method in build_public_methods(alpha=ALPHA, seed=methods_seed)
                if method.name != "monitor"
            )
            names = tuple(method.name for method in safe_methods)
            if names != tuple(method.name for method in harmful_methods):
                raise AssertionError("offline method registries differ")
            method_names = method_names or names
            if method_names != names:
                raise AssertionError("offline method registry changed")
            decisions: dict[str, tuple[object, ...]] = {}
            for safe_method, harmful_method in zip(
                safe_methods, harmful_methods, strict=True
            ):
                safe_decisions = tuple(
                    safe_method.decide(public.update)
                    for public in pair.safe.public_rounds
                )
                harmful_decisions = tuple(
                    harmful_method.decide(public.update)
                    for public in pair.harmful.public_rounds
                )
                if safe_decisions != harmful_decisions:
                    raise AssertionError("an offline method separated matched worlds")
                decisions[safe_method.name] = safe_decisions
            for round_index, (public, signature) in enumerate(
                zip(pair.safe.public_rounds, round_signatures, strict=True), start=1
            ):
                witnesses.append(
                    {
                        "family": family,
                        "semantic_unit": semantic_unit,
                        "round": round_index,
                        "public_component_count": len(signature),
                        "required_groups": list(public.monitor_groups),
                        "safe_answer": True,
                        "harmful_answer": False,
                        "public_observations_match": True,
                        "offline_decisions_match": True,
                        "offline_decisions": {
                            name: bool(decisions[name][round_index - 1].deploy)
                            for name in names
                        },
                        "sampling_scope": "semantic_lifecycle_round_witness",
                    }
                )
    uniqueness = {family: len(signatures[family]) for family in FAMILIES}
    structure_uniqueness = {family: len(structures[family]) for family in FAMILIES}
    if structure_uniqueness != units:
        raise AssertionError(
            "identifier-free environment structures are not unique: "
            f"{structure_uniqueness}"
        )
    if method_names is None:
        raise AssertionError("offline methods were not evaluated")
    return (
        witnesses,
        method_names,
        {
            "method_facing": uniqueness,
            "environment_structure": structure_uniqueness,
        },
    )


def _method_signature(public: PublicUpdateRound) -> tuple[object, ...]:
    return (
        tuple(
            (
                component.name,
                component.audit,
                component.holdout,
                component.require_all,
            )
            for component in public.update.components
        ),
        public.update.pace_outcomes,
    )


def _exact_trials(
    pairs: dict[str, tuple[MatchedLifecyclePair, ...]],
    streams: int,
    monitor_sizes: tuple[int, ...],
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        for semantic_unit, pair in enumerate(pairs[family]):
            roster = tuple(
                sorted(
                    {
                        group
                        for public in pair.safe.public_rounds
                        for group in public.monitor_groups
                    }
                )
            )
            for world, truths in (
                ("safe", pair.safe.protected_truth),
                ("harmful", pair.harmful.protected_truth),
            ):
                group_probabilities, harm_probabilities = _exact_distribution(
                    truths, roster
                )
                for stream in range(streams):
                    random_source = random.Random(
                        f"identifiability:{seed}:exact:{family}:{semantic_unit}:{world}:{stream}"
                    )
                    trajectory = _monitor_trajectory(
                        roster,
                        group_probabilities,
                        harm_probabilities,
                        1.0,
                        "representative",
                        monitor_sizes,
                        random_source,
                    )
                    rows.append(
                        _trial_row(
                            "exact_family",
                            world,
                            stream,
                            roster,
                            trajectory,
                            family=family,
                            semantic_unit=semantic_unit,
                        )
                    )
    return rows


def _exact_distribution(
    truths: tuple[ProtectedRoundTruth, ...], roster: tuple[str, ...]
) -> tuple[dict[str, float], dict[str, float]]:
    exposure, harm = _exact_totals(truths, roster)
    total = sum(exposure.values(), Fraction())
    return (
        {group: float(exposure[group] / total) for group in roster},
        {group: float(harm[group] / exposure[group]) for group in roster},
    )


def _exact_totals(
    truths: tuple[ProtectedRoundTruth, ...], roster: tuple[str, ...]
) -> tuple[dict[str, Fraction], dict[str, Fraction]]:
    exposure = {group: Fraction() for group in roster}
    harm = {group: Fraction() for group in roster}
    for truth in truths:
        if {record.group for record in truth.group_harm} != set(roster):
            raise AssertionError("protected group roster differs from public roster")
        for record in truth.group_harm:
            exposure[record.group] += record.exposure_mass
            harm[record.group] += record.harm_mass
    total = sum(exposure.values(), Fraction())
    if total <= 0 or any(value <= 0 for value in exposure.values()):
        raise AssertionError("every required group needs positive exact exposure")
    return exposure, harm


def _exact_truth_summary(
    pairs: dict[str, tuple[MatchedLifecyclePair, ...]],
) -> list[dict[str, object]]:
    rows = []
    for family in FAMILIES:
        for semantic_unit, pair in enumerate(pairs[family]):
            roster = tuple(
                sorted(
                    {
                        group
                        for public in pair.safe.public_rounds
                        for group in public.monitor_groups
                    }
                )
            )
            for world, truths in (
                ("safe", pair.safe.protected_truth),
                ("harmful", pair.harmful.protected_truth),
            ):
                exposure, harm = _exact_totals(truths, roster)
                total = sum(exposure.values(), Fraction())
                rows.extend(
                    {
                        "family": family,
                        "semantic_unit": semantic_unit,
                        "world": world,
                        "group": group,
                        "exposure_mass": _fraction_text(exposure[group]),
                        "harm_mass": _fraction_text(harm[group]),
                        "exposure_share": _fraction_text(exposure[group] / total),
                        "conditional_harm": _fraction_text(
                            harm[group] / exposure[group]
                        ),
                        "visibility": "simulator_scorer_only",
                    }
                    for group in roster
                )
    return rows


def _controlled_trials(
    streams: int, monitor_sizes: tuple[int, ...], seed: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    threshold = float(WORST_GROUP_LIMIT)
    roster = ("common", "rare")
    for gap in CONTROLLED_GAPS:
        for rare_prevalence in RARE_PREVALENCES:
            group_probabilities = {
                "common": 1 - rare_prevalence,
                "rare": rare_prevalence,
            }
            for retention in RETENTIONS:
                design = (
                    "representative"
                    if retention == 1
                    else "outcome_independent_missingness"
                )
                for world in ("safe", "harmful"):
                    harm_probabilities = {
                        "common": threshold - gap,
                        "rare": threshold - gap if world == "safe" else threshold + gap,
                    }
                    for stream in range(streams):
                        random_source = random.Random(
                            f"identifiability:{seed}:controlled:{gap}:{rare_prevalence}:{retention}:{world}:{stream}"
                        )
                        trajectory = _monitor_trajectory(
                            roster,
                            group_probabilities,
                            harm_probabilities,
                            retention,
                            design,
                            monitor_sizes,
                            random_source,
                        )
                        rows.append(
                            _trial_row(
                                "controlled",
                                world,
                                stream,
                                roster,
                                trajectory,
                                gap=gap,
                                rare_prevalence=rare_prevalence,
                                retention=retention,
                            )
                        )
    return rows


def _monitor_trajectory(
    roster: tuple[str, ...],
    group_probabilities: dict[str, float],
    harm_probabilities: dict[str, float],
    retention: float,
    collection_design: str,
    monitor_sizes: tuple[int, ...],
    random_source: random.Random,
) -> tuple[dict[str, object], ...]:
    harms = {group: 0 for group in roster}
    observations = {group: 0 for group in roster}
    previous = 0
    terminal_status = None
    trajectory = []
    for monitor_n in monitor_sizes:
        attempts = _multinomial_counts(
            roster,
            group_probabilities,
            monitor_n - previous,
            random_source,
        )
        for group in roster:
            observed = random_source.binomialvariate(attempts[group], retention)
            observations[group] += observed
            harms[group] += random_source.binomialvariate(
                observed, harm_probabilities[group]
            )
        previous = monitor_n
        if terminal_status is None:
            result = monitor_identified_range(
                frozenset((True, False)),
                roster,
                harms,
                observations,
                harm_threshold=float(WORST_GROUP_LIMIT),
                lifecycle_alpha=ALPHA,
                collection_design=collection_design,
            )
            status = result.status
            if status in {"deploy", "hold"}:
                terminal_status = status
        else:
            status = terminal_status
        trajectory.append(
            {
                "monitor_n": monitor_n,
                "status": status,
                "group_counts": [
                    {
                        "group": group,
                        "harms": harms[group],
                        "observations": observations[group],
                    }
                    for group in roster
                ],
            }
        )
    return tuple(trajectory)


def _multinomial_counts(
    roster: tuple[str, ...],
    probabilities: dict[str, float],
    trials: int,
    random_source: random.Random,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    remaining_trials = trials
    remaining_probability = sum(probabilities.values())
    for group in roster[:-1]:
        conditional = probabilities[group] / remaining_probability
        count = random_source.binomialvariate(remaining_trials, conditional)
        counts[group] = count
        remaining_trials -= count
        remaining_probability -= probabilities[group]
    counts[roster[-1]] = remaining_trials
    return counts


def _trial_row(
    panel: str,
    world: str,
    stream: int,
    roster: tuple[str, ...],
    trajectory: tuple[dict[str, object], ...],
    **fields: object,
) -> dict[str, object]:
    first = next(
        (
            {"monitor_n": look["monitor_n"], "status": look["status"]}
            for look in trajectory
            if look["status"] in {"deploy", "hold"}
        ),
        None,
    )
    return {
        "panel": panel,
        "world": world,
        "stream": stream,
        "required_groups": list(roster),
        "looks": [
            {"monitor_n": look["monitor_n"], "status": look["status"]}
            for look in trajectory
        ],
        "final_group_counts": trajectory[-1]["group_counts"],
        "first_decision": first,
        "final_status": trajectory[-1]["status"],
        **fields,
    }


def _summarize_exact(
    trials: list[dict[str, object]],
    units: dict[str, int],
    streams: int,
    monitor_sizes: tuple[int, ...],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in trials:
        grouped[
            (str(row["family"]), int(row["semantic_unit"]), str(row["world"]))
        ].append(row)
    summary = []
    for family in FAMILIES:
        for world in ("safe", "harmful"):
            correct_status = "deploy" if world == "safe" else "hold"
            for look_index, monitor_n in enumerate(monitor_sizes):
                correct_units = 0
                false_safe_units = 0
                for semantic_unit in range(units[family]):
                    cohort = grouped[(family, semantic_unit, world)]
                    if len(cohort) != streams:
                        raise AssertionError("exact stream cohort is incomplete")
                    correct = sum(
                        row["looks"][look_index]["status"] == correct_status
                        for row in cohort
                    )
                    correct_units += correct / streams >= CORRECT_STREAM_FRACTION
                    false_safe_units += world == "harmful" and any(
                        look["status"] == "deploy"
                        for row in cohort
                        for look in row["looks"][: look_index + 1]
                    )
                summary.append(
                    _fixed_panel_summary(
                        family,
                        world,
                        "correct",
                        monitor_n,
                        correct_units,
                        units[family],
                    )
                )
                summary.append(
                    _fixed_panel_summary(
                        family,
                        world,
                        "failure",
                        monitor_n,
                        units[family] - correct_units,
                        units[family],
                    )
                )
                summary.append(
                    _fixed_panel_summary(
                        family,
                        world,
                        "false_safe",
                        monitor_n,
                        false_safe_units,
                        units[family],
                    )
                )
    return summary


def _summarize_controlled(
    trials: list[dict[str, object]], streams: int, monitor_sizes: tuple[int, ...]
) -> list[dict[str, object]]:
    grouped: dict[tuple[float, float, float, str], list[dict[str, object]]] = (
        defaultdict(list)
    )
    for row in trials:
        grouped[
            (
                float(row["gap"]),
                float(row["rare_prevalence"]),
                float(row["retention"]),
                str(row["world"]),
            )
        ].append(row)
    cells = []
    for gap in CONTROLLED_GAPS:
        for prevalence in RARE_PREVALENCES:
            for retention in RETENTIONS:
                safe = grouped[(gap, prevalence, retention, "safe")]
                harmful = grouped[(gap, prevalence, retention, "harmful")]
                if len(safe) != streams or len(harmful) != streams:
                    raise AssertionError("controlled stream cohort is incomplete")
                looks = []
                first_qualifying = None
                for look_index, monitor_n in enumerate(monitor_sizes):
                    safe_correct = sum(
                        row["looks"][look_index]["status"] == "deploy" for row in safe
                    )
                    harmful_correct = sum(
                        row["looks"][look_index]["status"] == "hold" for row in harmful
                    )
                    false_safe = sum(
                        any(
                            look["status"] == "deploy"
                            for look in row["looks"][: look_index + 1]
                        )
                        for row in harmful
                    )
                    safe_rate = _rate_fields(safe_correct, streams)
                    harmful_rate = _rate_fields(harmful_correct, streams)
                    false_safe_rate = _rate_fields(false_safe, streams)
                    qualifies = (
                        safe_rate["ci_lower"] >= 0.80
                        and harmful_rate["ci_lower"] >= 0.80
                        and false_safe_rate["ci_upper"] <= 0.05
                    )
                    if qualifies and first_qualifying is None:
                        first_qualifying = monitor_n
                    looks.append(
                        {
                            "monitor_n": monitor_n,
                            "safe_correct": safe_rate,
                            "harmful_correct": harmful_rate,
                            "harmful_false_safe": false_safe_rate,
                            "qualifies": qualifies,
                        }
                    )
                cells.append(
                    {
                        "gap": gap,
                        "rare_prevalence": prevalence,
                        "retention": retention,
                        "first_qualifying_monitor_n": first_qualifying,
                        "looks": looks,
                    }
                )
    return cells


def _rate_summary(
    family: str,
    world: str,
    metric: str,
    monitor_n: int,
    successes: int,
    trials: int,
    sampling_scope: str,
) -> dict[str, object]:
    return {
        "family": family,
        "world": world,
        "metric": metric,
        "monitor_n": monitor_n,
        "sampling_scope": sampling_scope,
        **_rate_fields(successes, trials),
    }


def _fixed_panel_summary(
    family: str,
    world: str,
    metric: str,
    monitor_n: int,
    successes: int,
    trials: int,
) -> dict[str, object]:
    return {
        "family": family,
        "world": world,
        "metric": metric,
        "monitor_n": monitor_n,
        "sampling_scope": "fixed_semantic_lifecycle_panel",
        "successes": successes,
        "trials": trials,
        "estimate": successes / trials,
        "ci_lower": None,
        "ci_upper": None,
        "uncertainty": "descriptive_fixed_panel_no_population_interval",
    }


def _rate_fields(successes: int, trials: int) -> dict[str, object]:
    lower, upper = _clopper_pearson(successes, trials)
    return {
        "successes": successes,
        "trials": trials,
        "estimate": successes / trials,
        "ci_lower": lower,
        "ci_upper": upper,
        "uncertainty": "exact_one_sided_95_clopper_pearson",
    }


def _claim_status(
    exact_summary: list[dict[str, object]],
    assumptions: list[dict[str, object]],
    maximum_monitor_n: int,
) -> dict[str, object]:
    final_rows = [row for row in exact_summary if row["monitor_n"] == maximum_monitor_n]
    correct = [row for row in final_rows if row["metric"] == "correct"]
    harmful_false_safe = [
        row
        for row in final_rows
        if row["world"] == "harmful" and row["metric"] == "false_safe"
    ]
    criteria = {
        "all_three_families_represented": {row["family"] for row in correct}
        == set(FAMILIES),
        "final_descriptive_correct_rate_at_least_0_80": all(
            float(row["estimate"]) >= 0.80 for row in correct
        ),
        "no_harmful_false_safe_semantic_units": all(
            int(row["successes"]) == 0 for row in harmful_false_safe
        ),
        "fail_closed_assumption_checks_pass": all(
            bool(row["passed"]) for row in assumptions
        ),
    }
    return {
        "interpretation": "descriptive_fixed_semantic_panel_gate",
        "criteria": criteria,
        "status": "passed" if all(criteria.values()) else "failed",
    }


def _assumption_checks(
    maximum_monitor_n: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    roster = ("common", "rare")
    zero = {group: 0 for group in roster}
    no_monitoring = monitor_identified_range(
        frozenset((True, False)),
        roster,
        zero,
        zero,
        harm_threshold=float(WORST_GROUP_LIMIT),
        lifecycle_alpha=ALPHA,
        collection_design="representative",
    )
    missing_group = monitor_identified_range(
        frozenset((True, False)),
        roster,
        zero,
        {"common": maximum_monitor_n, "rare": 0},
        harm_threshold=float(WORST_GROUP_LIMIT),
        lifecycle_alpha=ALPHA,
        collection_design="outcome_independent_missingness",
    )
    unsupported = {}
    for design in ("outcome_dependent_selection", "distribution_shift"):
        unsupported[design] = monitor_identified_range(
            frozenset((True, False)),
            roster,
            zero,
            zero,
            harm_threshold=float(WORST_GROUP_LIMIT),
            lifecycle_alpha=ALPHA,
            collection_design=design,
        )
    assumptions = [
        {
            "case": "no_monitoring",
            "status": no_monitoring.status,
            "expected_status": "cannot_determine",
            "passed": no_monitoring.status == "cannot_determine",
        },
        {
            "case": "missing_required_group",
            "status": missing_group.status,
            "expected_status": "cannot_determine",
            "passed": missing_group.status == "cannot_determine",
        },
        *(
            {
                "case": design,
                "status": result.status,
                "expected_status": "unsupported",
                "passed": result.status == "unsupported",
            }
            for design, result in unsupported.items()
        ),
    ]
    naive = monitor_identified_range(
        frozenset((True, False)),
        roster,
        zero,
        {group: maximum_monitor_n for group in roster},
        harm_threshold=float(WORST_GROUP_LIMIT),
        lifecycle_alpha=ALPHA,
        collection_design="representative",
    )
    invalid_ablations = [
        {
            "name": "naive_outcome_dependent_selection_as_representative",
            "status": naive.status,
            "valid_method": False,
            "included_in_method_performance": False,
            "reason": "the declared design contradicts outcome-dependent observation selection",
            "selection_rule": "retain nonharm outcomes and discard harm outcomes",
            "observed_harms": dict(zero),
            "observations": {group: maximum_monitor_n for group in roster},
        }
    ]
    if not all(bool(row["passed"]) for row in assumptions):
        raise AssertionError("fail-closed assumption checks did not pass")
    return assumptions, invalid_ablations


def _stable_seed(seed: int, *parts: object) -> int:
    return random.Random(":".join(str(part) for part in (seed, *parts))).randrange(
        2**63
    )


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


def _jsonl(rows: list[dict[str, object]]) -> str:
    return "".join(_json(row) for row in rows)
