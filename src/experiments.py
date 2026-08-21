from __future__ import annotations

import json
import math
import os
import random
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from .decision_methods import ALPHA, ComponentEvidence, UpdateEvidence, build_methods
from .experiment_plots import impossibility_svg, landscape_svg, restoration_svg

FAMILIES = ("authgate_v0", "constraint_plan_v0")
SCENARIOS = ("null_only", "all_good", "mixed")
CONTROLLED_METHODS = ("shrinking_budget", "addis_spending", "online_closed_e")
OFFLINE_METHODS = (
    "always_hold",
    "greedy",
    "fixed_threshold",
    "shrinking_budget",
    "addis_spending",
    "online_closed_e",
    "pace_reset",
    "reused_holdout",
    "sgm_transferred",
)
MONITOR_SIZES = (5, 10, 20, 50, 100, 200, 500, 1000, 2500, 5000, 10000, 20000)
ONE_SIDED_95_Z = 1.6448536269514722


def run_experiments(
    output_dir: Path,
    *,
    artifacts_dir: Path | None = None,
    data_dir: Path | None = None,
    replications: int = 500,
    seed: int = 20260821,
    alpha: float = ALPHA,
) -> dict[str, object]:
    if replications < 1:
        raise ValueError("replications must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be within (0, 1)")

    data_dir = data_dir or Path(__file__).resolve().parents[1] / "data"
    monitor_settings = _monitor_settings(data_dir)
    landscape = _run_landscape(replications, seed, alpha)
    impossibility = _run_impossibility(replications, seed, alpha)
    restoration = _run_restoration(replications, seed, alpha, monitor_settings)
    rows = landscape + impossibility + restoration
    summary = {
        "config": {
            "alpha": alpha,
            "audit_batch": 64,
            "lifecycle_length": 50,
            "monitor_sizes": list(MONITOR_SIZES),
            "monitor_truth": monitor_settings,
            "replications": replications,
            "seed": seed,
        },
        "experiments": {
            "experiment_2": _landscape_status(landscape),
            "experiment_3": _impossibility_status(impossibility),
            "experiment_4": _restoration_status(restoration),
        },
    }
    artifacts_dir = artifacts_dir or output_dir
    payloads = {
        output_dir / "phase4_results.jsonl": "".join(_json_line(row) for row in rows),
        output_dir / "phase4_summary.json": _json_line(summary),
        artifacts_dir / "experiment_landscape.svg": landscape_svg(landscape, summary),
        artifacts_dir / "experiment_impossibility.svg": impossibility_svg(
            impossibility, summary
        ),
        artifacts_dir / "experiment_restoration.svg": restoration_svg(
            restoration, summary
        ),
    }
    _publish(payloads)
    return summary


def _run_landscape(
    replications: int,
    seed: int,
    alpha: float,
) -> list[dict[str, object]]:
    counts: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for family in FAMILIES:
        for scenario in SCENARIOS:
            for replication in range(replications):
                random_source = _random(
                    seed, "landscape", family, scenario, replication
                )
                updates: list[UpdateEvidence] = []
                safe: list[bool] = []
                for round_index in range(1, 51):
                    probability, is_safe = _scripted_round(scenario, round_index)
                    updates.append(
                        _sample_update(
                            family,
                            f"{scenario}-{round_index}",
                            probability,
                            random_source,
                        )
                    )
                    safe.append(is_safe)
                answers = {
                    (family, update.update_id): is_safe
                    for update, is_safe in zip(updates, safe, strict=True)
                }
                methods = build_methods(
                    answers,
                    alpha=alpha,
                    seed=_seed(seed, "thresholdout", family, scenario, replication),
                )
                for method in methods:
                    harmful_deployed = False
                    first_harm = 0
                    good_deployed = 0
                    good_total = 0
                    utility = 0
                    for round_index, (update, is_safe) in enumerate(
                        zip(updates, safe, strict=True), start=1
                    ):
                        decision = method.decide(update)
                        if is_safe:
                            good_total += 1
                            good_deployed += decision.deploy
                            utility += decision.deploy
                        elif decision.deploy:
                            harmful_deployed = True
                            first_harm = first_harm or round_index
                            utility -= 1
                    key = (family, scenario, method.name)
                    counts[key]["harm_lifecycles"] += harmful_deployed
                    counts[key]["good_deployed"] += good_deployed
                    counts[key]["good_total"] += good_total
                    counts[key]["first_harm_sum"] += first_harm
                    counts[key]["first_harm_count"] += bool(first_harm)
                    counts[key]["utility_sum"] += utility

    rows: list[dict[str, object]] = []
    for (family, scenario, method), values in sorted(counts.items()):
        rows.append(
            _rate_row(
                "experiment_2",
                family,
                scenario,
                method,
                "harmful_lifecycle",
                int(values["harm_lifecycles"]),
                replications,
            )
        )
        rows.append(
            _rate_row(
                "experiment_2",
                family,
                scenario,
                method,
                "genuine_acceptance",
                int(values["good_deployed"]),
                int(values["good_total"]),
            )
        )
        rows.append(
            _mean_row(
                "experiment_2",
                family,
                scenario,
                method,
                "first_harm_round",
                values["first_harm_sum"],
                int(values["first_harm_count"]),
            )
        )
        rows.append(
            _mean_row(
                "experiment_2",
                family,
                scenario,
                method,
                "final_utility",
                values["utility_sum"],
                replications,
            )
        )
    return rows


def _run_impossibility(
    replications: int,
    seed: int,
    alpha: float,
) -> list[dict[str, object]]:
    accepted: dict[tuple[str, str, str], int] = defaultdict(int)
    matched: dict[tuple[str, str], int] = defaultdict(int)
    abstained: dict[str, int] = defaultdict(int)
    for family in FAMILIES:
        for replication in range(replications):
            update = _sample_update(
                family,
                "matched-update",
                0.8,
                _random(seed, "impossibility", family, replication),
            )
            decisions: dict[str, dict[str, tuple[object, ...]]] = {}
            for world, answer in (("safe", True), ("harmful", False)):
                methods = build_methods(
                    {(family, update.update_id): answer},
                    alpha=alpha,
                    seed=_seed(seed, "impossibility-thresholdout", family, replication),
                )
                decisions[world] = {}
                for method in methods:
                    if method.name not in OFFLINE_METHODS:
                        continue
                    decision = method.decide(update)
                    decisions[world][method.name] = (
                        decision.deploy,
                        decision.reason,
                        decision.statistic,
                        decision.threshold,
                    )
                    accepted[(family, world, method.name)] += decision.deploy
            for method in OFFLINE_METHODS:
                matched[(family, method)] += (
                    decisions["safe"][method] == decisions["harmful"][method]
                )
            abstained[family] += (
                identified_range_status(frozenset({True, False})) == "cannot_determine"
            )

    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        for method in OFFLINE_METHODS:
            for world in ("safe", "harmful"):
                rows.append(
                    _rate_row(
                        "experiment_3",
                        family,
                        world,
                        method,
                        "acceptance",
                        accepted[(family, world, method)],
                        replications,
                    )
                )
            rows.append(
                _rate_row(
                    "experiment_3",
                    family,
                    "paired_worlds",
                    method,
                    "paired_decision_match",
                    matched[(family, method)],
                    replications,
                )
            )
        for world in ("safe", "harmful"):
            rows.append(
                _rate_row(
                    "experiment_3",
                    family,
                    world,
                    "identified_range",
                    "cannot_determine",
                    abstained[family],
                    replications,
                )
            )
    return rows


def _run_restoration(
    replications: int,
    seed: int,
    alpha: float,
    settings: dict[str, dict[str, tuple[float, float]]],
) -> list[dict[str, object]]:
    counts: dict[tuple[str, str, str, int, str], int] = defaultdict(int)
    estimates: dict[tuple[str, str, int], float] = defaultdict(float)
    false_safe_lifecycles: dict[tuple[str, str], int] = defaultdict(int)
    for family in FAMILIES:
        for world in ("safe", "harmful"):
            probability, limit = settings[family][world]
            for replication in range(replications):
                random_source = _random(seed, "restoration", family, world, replication)
                harm_count = 0
                previous_n = 0
                false_safe = {"correct_width": False, "too_narrow": False}
                for monitor_n in MONITOR_SIZES:
                    harm_count += sum(
                        random_source.random() < probability
                        for _ in range(monitor_n - previous_n)
                    )
                    previous_n = monitor_n
                    estimate = harm_count / monitor_n
                    estimates[(family, world, monitor_n)] += estimate
                    statuses = {
                        "correct_width": _confidence_sequence_status(
                            estimate, monitor_n, limit, alpha
                        ),
                        "too_narrow": _wald_status(estimate, monitor_n, limit),
                    }
                    for rule, status in statuses.items():
                        counts[(family, world, rule, monitor_n, status)] += 1
                        false_safe[rule] |= world == "harmful" and status == "safe"
                for rule, occurred in false_safe.items():
                    false_safe_lifecycles[(family, rule)] += occurred

    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        for world in ("safe", "harmful"):
            for rule in ("correct_width", "too_narrow"):
                for monitor_n in MONITOR_SIZES:
                    for status in ("safe", "harmful", "cannot_determine"):
                        row = _rate_row(
                            "experiment_4",
                            family,
                            world,
                            rule,
                            f"classified_{status}",
                            counts[(family, world, rule, monitor_n, status)],
                            replications,
                        )
                        row["monitor_n"] = monitor_n
                        rows.append(row)
                    estimate_row = _mean_row(
                        "experiment_4",
                        family,
                        world,
                        rule,
                        "estimated_harm",
                        estimates[(family, world, monitor_n)],
                        replications,
                    )
                    estimate_row["monitor_n"] = monitor_n
                    rows.append(estimate_row)
            for rule in ("correct_width", "too_narrow"):
                row = _rate_row(
                    "experiment_4",
                    family,
                    world,
                    rule,
                    "harmful_lifecycle_false_safe",
                    false_safe_lifecycles[(family, rule)] if world == "harmful" else 0,
                    replications,
                )
                row["monitor_n"] = 0
                rows.append(row)
    return rows


def _landscape_status(rows: list[dict[str, object]]) -> dict[str, object]:
    controlled_harm = [
        row
        for row in rows
        if row["metric"] == "harmful_lifecycle"
        and row["method"] in CONTROLLED_METHODS
        and row["scenario"] in {"null_only", "mixed"}
    ]
    controlled_acceptance = [
        row
        for row in rows
        if row["metric"] == "genuine_acceptance"
        and row["method"] in CONTROLLED_METHODS
        and row["scenario"] == "all_good"
    ]
    separated = []
    for family in FAMILIES:
        sgm = _find(rows, family, "mixed", "sgm_transferred", "harmful_lifecycle")
        best_controlled = max(
            float(_find(rows, family, "mixed", method, "harmful_lifecycle")["ci_upper"])
            for method in CONTROLLED_METHODS
        )
        separated.append(float(sgm["ci_lower"]) >= best_controlled + 0.20)
    criteria = {
        "controlled_harm_upper_at_most_0_05": all(
            float(row["ci_upper"]) <= 0.05 for row in controlled_harm
        ),
        "nontrivial_all_good_acceptance": all(
            any(
                float(row["ci_lower"]) > 0.30
                for row in controlled_acceptance
                if row["family"] == family
            )
            for family in FAMILIES
        ),
        "sgm_mixed_separation_at_least_0_20": all(separated),
    }
    return {
        "criteria": criteria,
        "status": "passed" if all(criteria.values()) else "failed",
    }


def _impossibility_status(rows: list[dict[str, object]]) -> dict[str, object]:
    pair_rows = [row for row in rows if row["metric"] == "paired_decision_match"]
    cannot_rows = [row for row in rows if row["metric"] == "cannot_determine"]
    nontrivial = []
    for family in FAMILIES:
        found = False
        for method in OFFLINE_METHODS:
            safe = _find(rows, family, "safe", method, "acceptance")
            harmful = _find(rows, family, "harmful", method, "acceptance")
            found |= (
                safe["successes"] == harmful["successes"]
                and float(safe["ci_lower"]) > 0.30
            )
        nontrivial.append(found)
    criteria = {
        "all_offline_decisions_match": all(
            row["successes"] == row["trials"] for row in pair_rows
        ),
        "nontrivial_matched_acceptance": all(nontrivial),
        "identified_range_always_abstains": all(
            row["successes"] == row["trials"] for row in cannot_rows
        ),
    }
    return {
        "criteria": criteria,
        "status": "passed" if all(criteria.values()) else "failed",
    }


def _restoration_status(rows: list[dict[str, object]]) -> dict[str, object]:
    correct_lifecycle_false_safe = [
        row
        for row in rows
        if row["method"] == "correct_width"
        and row["scenario"] == "harmful"
        and row["metric"] == "harmful_lifecycle_false_safe"
    ]
    safe_at_max = [
        row
        for row in rows
        if row["method"] == "correct_width"
        and row["scenario"] == "safe"
        and row["metric"] == "classified_safe"
        and row.get("monitor_n") == MONITOR_SIZES[-1]
    ]
    narrow_auth = _find(
        rows,
        "authgate_v0",
        "harmful",
        "too_narrow",
        "harmful_lifecycle_false_safe",
    )
    first_separation: dict[str, int | None] = {}
    for family in FAMILIES:
        first_separation[family] = next(
            (
                monitor_n
                for monitor_n in MONITOR_SIZES
                if float(
                    _find(
                        rows,
                        family,
                        "safe",
                        "correct_width",
                        "classified_safe",
                        monitor_n,
                    )["ci_lower"]
                )
                >= 0.80
                and float(
                    _find(
                        rows,
                        family,
                        "harmful",
                        "correct_width",
                        "classified_harmful",
                        monitor_n,
                    )["ci_lower"]
                )
                >= 0.80
            ),
            None,
        )
    criteria = {
        "correct_width_lifecycle_harm_upper_at_most_0_05": all(
            float(row["ci_upper"]) <= 0.05 for row in correct_lifecycle_false_safe
        ),
        "correct_width_safe_power_at_least_0_80": all(
            float(row["ci_lower"]) >= 0.80 for row in safe_at_max
        ),
        "families_separate": all(
            value is not None for value in first_separation.values()
        ),
        "too_narrow_reintroduces_authgate_harm": float(narrow_auth["ci_lower"]) > 0.05,
    }
    return {
        "criteria": criteria,
        "first_separation_monitor_n": first_separation,
        "status": "passed" if all(criteria.values()) else "failed",
    }


def _scripted_round(scenario: str, round_index: int) -> tuple[float, bool]:
    if scenario == "all_good":
        return 0.8, True
    if scenario == "mixed" and round_index <= 15:
        return 0.8, True
    return (0.5, False) if scenario == "null_only" else (0.2, False)


def _monitor_settings(
    data_dir: Path,
) -> dict[str, dict[str, tuple[float, float]]]:
    auth_rows = [
        row
        for row in _read_jsonl(data_dir / "truth.jsonl")
        if row.get("policy_id") == "candidate"
        and row.get("action") == "grant_temporary_access"
    ]
    planning_rows = [
        row
        for row in _read_jsonl(data_dir / "planning_truth.jsonl")
        if row.get("policy_id") == "p4"
    ]
    observed = {
        "authgate_v0": {
            world: {
                _fraction(row, "live_harm")
                for row in auth_rows
                if row.get("world") == world
            }
            for world in ("safe", "harmful")
        },
        "constraint_plan_v0": {
            world: {
                _fraction(row, "rare_group_harm")
                for row in planning_rows
                if row.get("world") == world
            }
            for world in ("safe", "harmful")
        },
    }
    expected = {
        "authgate_v0": {"safe": {Fraction(1, 50)}, "harmful": {Fraction(1, 5)}},
        "constraint_plan_v0": {"safe": {Fraction(0)}, "harmful": {Fraction(1)}},
    }
    if observed != expected:
        raise ValueError("committed monitor truth does not match the Phase 4 protocol")
    limits = {"authgate_v0": 0.05, "constraint_plan_v0": 0.15}
    return {
        family: {
            world: (float(next(iter(rates))), limits[family])
            for world, rates in worlds.items()
        }
        for family, worlds in observed.items()
    }


def _sample_update(
    family: str,
    update_id: str,
    win_probability: float,
    random_source: random.Random,
) -> UpdateEvidence:
    audit = tuple(
        1.0 if random_source.random() < win_probability else -1.0 for _ in range(64)
    )
    holdout = tuple(
        1.0 if random_source.random() < win_probability else -1.0 for _ in range(32)
    )
    names = (
        ("harm", "quality")
        if family == "authgate_v0"
        else ("public_feasibility", "objective")
    )
    components = tuple(ComponentEvidence(name, audit, holdout) for name in names)
    return UpdateEvidence(
        family=family,
        update_id=update_id,
        components=components,
        pace_outcomes=tuple(value > 0 for value in audit),
    )


def _confidence_sequence_status(
    estimate: float,
    sample_size: int,
    limit: float,
    alpha: float,
) -> str:
    allocated_alpha = alpha / (sample_size * (sample_size + 1))
    radius = math.sqrt(math.log(2 / allocated_alpha) / (2 * sample_size))
    return _interval_status(estimate, radius, limit)


def identified_range_status(compatible_answers: frozenset[bool]) -> str:
    if not compatible_answers:
        raise ValueError("at least one compatible answer is required")
    if len(compatible_answers) > 1:
        return "cannot_determine"
    return "deploy" if next(iter(compatible_answers)) else "hold"


def _wald_status(estimate: float, sample_size: int, limit: float) -> str:
    radius = 1.959963984540054 * math.sqrt(estimate * (1 - estimate) / sample_size)
    return _interval_status(estimate, radius, limit)


def _interval_status(estimate: float, radius: float, limit: float) -> str:
    if min(1.0, estimate + radius) <= limit:
        return "safe"
    if max(0.0, estimate - radius) > limit:
        return "harmful"
    return "cannot_determine"


def _rate_row(
    experiment: str,
    family: str,
    scenario: str,
    method: str,
    metric: str,
    successes: int,
    trials: int,
) -> dict[str, object]:
    lower, upper = _wilson(successes, trials)
    return {
        "ci_lower": lower,
        "ci_upper": upper,
        "estimate": successes / trials if trials else 0.0,
        "experiment": experiment,
        "family": family,
        "method": method,
        "metric": metric,
        "scenario": scenario,
        "successes": successes,
        "trials": trials,
    }


def _mean_row(
    experiment: str,
    family: str,
    scenario: str,
    method: str,
    metric: str,
    value_sum: float,
    trials: int,
) -> dict[str, object]:
    return {
        "ci_lower": None,
        "ci_upper": None,
        "estimate": value_sum / trials if trials else 0.0,
        "experiment": experiment,
        "family": family,
        "method": method,
        "metric": metric,
        "scenario": scenario,
        "trials": trials,
        "value_sum": value_sum,
    }


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    if trials < 1:
        return 0.0, 1.0
    proportion = successes / trials
    z_squared = ONE_SIDED_95_Z * ONE_SIDED_95_Z
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    radius = (
        ONE_SIDED_95_Z
        * math.sqrt(
            proportion * (1 - proportion) / trials + z_squared / (4 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _find(
    rows: Iterable[dict[str, object]],
    family: str,
    scenario: str,
    method: str,
    metric: str,
    monitor_n: int | None = None,
) -> dict[str, object]:
    matches = [
        row
        for row in rows
        if row["family"] == family
        and row["scenario"] == scenario
        and row["method"] == method
        and row["metric"] == metric
        and (monitor_n is None or row.get("monitor_n") == monitor_n)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one result row, found {len(matches)}")
    return matches[0]


def _random(seed: int, *parts: object) -> random.Random:
    return random.Random(":".join(str(part) for part in (seed,) + parts))


def _seed(seed: int, *parts: object) -> int:
    return _random(seed, *parts).randrange(2**63)


def _json_line(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name} must contain JSON objects")
    return rows


def _fraction(row: dict[str, object], field: str) -> Fraction:
    try:
        value = Fraction(str(row[field]))
    except (KeyError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"invalid exact fraction in {field}") from error
    if not 0 <= value <= 1:
        raise ValueError(f"{field} must be within [0, 1]")
    return value


def _publish(payloads: dict[Path, str]) -> None:
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
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
            os.chmod(temporary, 0o644)
            staged[target] = temporary
        for target in payloads:
            if target.exists():
                descriptor, backup_name = tempfile.mkstemp(dir=target.parent)
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
