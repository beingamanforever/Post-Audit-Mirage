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

from .decision_methods import ALPHA, Decision, Oracle, build_public_methods
from .environment_lifecycles import (
    MatchedLifecyclePair,
    ProtectedRoundTruth,
    RealizedLifecycle,
    realize_lifecycle,
    realize_matched_pair,
)
from .experiment_plots import impossibility_svg, landscape_svg, restoration_svg
from .lifecycle import WORST_GROUP_LIMIT

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
ONE_SIDED_CONFIDENCE = 0.95


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

    family_seeds = {
        family: _seed(seed, "environment", family) % (2**31) for family in FAMILIES
    }
    data_dir = data_dir or Path(__file__).resolve().parents[1] / "data"
    planning_templates = data_dir / "planning_templates.json"
    if not planning_templates.is_file():
        raise ValueError(f"missing planning templates: {planning_templates}")
    lifecycles = {
        (family, scenario): realize_lifecycle(
            family,
            scenario,
            family_seeds[family],
            planning_templates=planning_templates,
        )
        for family in FAMILIES
        for scenario in SCENARIOS
    }
    matched_pairs = {
        family: realize_matched_pair(
            family,
            family_seeds[family],
            planning_templates=planning_templates,
        )
        for family in FAMILIES
    }
    monitor_settings = _monitor_settings(matched_pairs)
    landscape = _run_landscape(lifecycles, seed, alpha)
    impossibility = _run_impossibility(matched_pairs, seed, alpha)
    restoration = _run_restoration(
        replications,
        seed,
        alpha,
        monitor_settings,
        matched_pairs,
    )
    rows = landscape + impossibility + restoration
    summary = {
        "config": {
            "alpha": alpha,
            "environment_updates": sum(
                len(lifecycle.public_rounds) for lifecycle in lifecycles.values()
            ),
            "lifecycle_length": 50,
            "monitor_sizes": list(MONITOR_SIZES),
            "monitor_truth": monitor_settings,
            "monitor_replications": replications,
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
        output_dir / "phase4_lifecycles.jsonl": "".join(
            _json_line(row) for row in _lifecycle_rows(lifecycles.values())
        ),
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
    lifecycles: dict[tuple[str, str], RealizedLifecycle],
    seed: int,
    alpha: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        for scenario in SCENARIOS:
            lifecycle = lifecycles[(family, scenario)]
            answers = {
                (family, public.update.update_id): truth.safe_to_deploy
                for public, truth in zip(
                    lifecycle.public_rounds,
                    lifecycle.protected_truth,
                    strict=True,
                )
            }
            methods = (
                *build_public_methods(
                    alpha=alpha,
                    seed=_seed(seed, "thresholdout", family, scenario),
                ),
                Oracle(answers),
            )
            for method in methods:
                harmful_deployed = False
                first_harm = 0
                good_deployed = 0
                good_total = 0
                utility = 0
                abstentions = 0
                group_harmful = False
                group_totals = _group_totals(lifecycle.protected_truth)
                for round_index, (public, truth) in enumerate(
                    zip(
                        lifecycle.public_rounds,
                        lifecycle.protected_truth,
                        strict=True,
                    ),
                    start=1,
                ):
                    monitor = truth.monitor if method.name == "monitor" else None
                    decision = method.decide(public.update, monitor)
                    abstentions += decision.reason == "live-style stream unavailable"
                    _add_deployed_group_truth(group_totals, truth, decision.deploy)
                    group_harmful |= decision.deploy and _has_harmful_group(truth)
                    is_safe = truth.safe_to_deploy
                    if is_safe:
                        good_total += 1
                        good_deployed += decision.deploy
                        utility += decision.deploy
                    elif decision.deploy:
                        harmful_deployed = True
                        first_harm = first_harm or round_index
                        utility -= 1
                rows.extend(
                    (
                        _descriptive_rate_row(
                            "experiment_2",
                            family,
                            scenario,
                            method.name,
                            "harmful_lifecycle",
                            int(harmful_deployed),
                            1,
                        ),
                        _descriptive_rate_row(
                            "experiment_2",
                            family,
                            scenario,
                            method.name,
                            "worst_group_harmful_lifecycle",
                            int(group_harmful),
                            1,
                        ),
                        _descriptive_rate_row(
                            "experiment_2",
                            family,
                            scenario,
                            method.name,
                            "genuine_acceptance",
                            good_deployed,
                            good_total,
                        ),
                        _descriptive_rate_row(
                            "experiment_2",
                            family,
                            scenario,
                            method.name,
                            "explicit_abstention",
                            abstentions,
                            len(lifecycle.public_rounds),
                        ),
                        _descriptive_mean_row(
                            "experiment_2",
                            family,
                            scenario,
                            method.name,
                            "first_harm_round",
                            first_harm,
                            int(bool(first_harm)),
                        ),
                        _descriptive_mean_row(
                            "experiment_2",
                            family,
                            scenario,
                            method.name,
                            "final_utility",
                            utility,
                            1,
                        ),
                    )
                )
                rows.extend(
                    _group_metric_rows(
                        "experiment_2",
                        family,
                        scenario,
                        method.name,
                        group_totals,
                    )
                )
    return rows


def _lifecycle_rows(
    lifecycles: Iterable[RealizedLifecycle],
) -> list[dict[str, object]]:
    rows = []
    for lifecycle in lifecycles:
        for public, truth in zip(
            lifecycle.public_rounds,
            lifecycle.protected_truth,
            strict=True,
        ):
            rows.append(
                {
                    "environment_ids": list(public.environment_ids),
                    "family": lifecycle.family,
                    "lifecycle_seed": lifecycle.seed,
                    "monitor_available": truth.monitor is not None,
                    "proposal_id": public.proposal_id,
                    "public_components": [
                        {
                            "audit": list(component.audit),
                            "holdout": list(component.holdout),
                            "name": component.name,
                            "require_all": component.require_all,
                        }
                        for component in public.update.components
                    ],
                    "public_failure_modes": list(public.failure_modes),
                    "public_monitor_groups": list(public.monitor_groups),
                    "round": public.index + 1,
                    "safe_to_deploy": truth.safe_to_deploy,
                    "scenario": lifecycle.scenario,
                    "truth_failure_modes": list(truth.failure_modes),
                    "truth_metrics": {
                        metric.name: (
                            f"{metric.value.numerator}/{metric.value.denominator}"
                        )
                        for metric in truth.metrics
                    },
                    "truth_group_harm": {
                        record.group: {
                            "exposure_mass": _fraction_text(record.exposure_mass),
                            "harm_mass": _fraction_text(record.harm_mass),
                            "rate": _fraction_text(
                                record.harm_mass / record.exposure_mass
                            ),
                        }
                        for record in truth.group_harm
                    },
                    "update_id": public.update.update_id,
                    "world": truth.world,
                }
            )
    return rows


def _run_impossibility(
    matched_pairs: dict[str, MatchedLifecyclePair],
    seed: int,
    alpha: float,
) -> list[dict[str, object]]:
    matched: dict[tuple[str, str], int] = defaultdict(int)
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        pair = matched_pairs[family]
        decisions: dict[str, dict[str, tuple[object, ...]]] = {}
        for world, lifecycle in (("safe", pair.safe), ("harmful", pair.harmful)):
            methods = build_public_methods(
                alpha=alpha,
                seed=_seed(seed, "impossibility-thresholdout", family),
            )
            decisions[world] = {}
            for method in methods:
                if method.name not in OFFLINE_METHODS:
                    continue
                lifecycle_decisions = tuple(
                    method.decide(public.update) for public in lifecycle.public_rounds
                )
                decisions[world][method.name] = tuple(
                    (
                        decision.deploy,
                        decision.reason,
                        decision.statistic,
                        decision.threshold,
                    )
                    for decision in lifecycle_decisions
                )
                rows.extend(
                    _offline_lifecycle_rows(
                        family,
                        world,
                        method.name,
                        lifecycle,
                        lifecycle_decisions,
                    )
                )
        for method in OFFLINE_METHODS:
            matched[(family, method)] += (
                decisions["safe"][method] == decisions["harmful"][method]
            )
        for method in OFFLINE_METHODS:
            rows.append(
                _descriptive_rate_row(
                    "experiment_3",
                    family,
                    "paired_worlds",
                    method,
                    "paired_decision_match",
                    matched[(family, method)],
                    1,
                )
            )
        for world in ("safe", "harmful"):
            cannot_determine = int(
                identified_range_status(frozenset({True, False})) == "cannot_determine"
            )
            rows.extend(
                (
                    _descriptive_rate_row(
                        "experiment_3",
                        family,
                        world,
                        "identified_range",
                        "cannot_determine",
                        cannot_determine,
                        1,
                    ),
                    _descriptive_rate_row(
                        "experiment_3",
                        family,
                        world,
                        "identified_range",
                        "explicit_abstention",
                        cannot_determine,
                        1,
                    ),
                )
            )
    return rows


def _offline_lifecycle_rows(
    family: str,
    world: str,
    method: str,
    lifecycle: RealizedLifecycle,
    decisions: tuple[Decision, ...],
) -> list[dict[str, object]]:
    harmful_deployed = False
    group_harmful = False
    good_deployed = 0
    good_total = 0
    first_harm = 0
    utility = 0
    group_totals = _group_totals(lifecycle.protected_truth)
    for round_index, (decision, truth) in enumerate(
        zip(decisions, lifecycle.protected_truth, strict=True), start=1
    ):
        deploy = bool(decision.deploy)
        _add_deployed_group_truth(group_totals, truth, deploy)
        group_harmful |= deploy and _has_harmful_group(truth)
        if truth.safe_to_deploy:
            good_total += 1
            good_deployed += deploy
            utility += deploy
        elif deploy:
            harmful_deployed = True
            first_harm = first_harm or round_index
            utility -= 1
    rows = [
        _descriptive_rate_row(
            "experiment_3",
            family,
            world,
            method,
            "acceptance",
            int(any(bool(decision.deploy) for decision in decisions)),
            1,
        ),
        _descriptive_rate_row(
            "experiment_3",
            family,
            world,
            method,
            "harmful_lifecycle",
            int(harmful_deployed),
            1,
        ),
        _descriptive_rate_row(
            "experiment_3",
            family,
            world,
            method,
            "worst_group_harmful_lifecycle",
            int(group_harmful),
            1,
        ),
        _descriptive_rate_row(
            "experiment_3",
            family,
            world,
            method,
            "genuine_acceptance",
            good_deployed,
            good_total,
        ),
        _descriptive_mean_row(
            "experiment_3",
            family,
            world,
            method,
            "first_harm_round",
            first_harm,
            int(bool(first_harm)),
        ),
        _descriptive_mean_row(
            "experiment_3",
            family,
            world,
            method,
            "final_utility",
            utility,
            1,
        ),
    ]
    rows.extend(_group_metric_rows("experiment_3", family, world, method, group_totals))
    return rows


def _group_totals(
    truths: tuple[ProtectedRoundTruth, ...],
) -> dict[str, list[Fraction]]:
    totals: dict[str, list[Fraction]] = {}
    for truth in truths:
        for record in truth.group_harm:
            values = totals.setdefault(
                record.group, [Fraction(), Fraction(), Fraction()]
            )
            values[2] += record.exposure_mass
    return totals


def _add_deployed_group_truth(
    totals: dict[str, list[Fraction]],
    truth: ProtectedRoundTruth,
    deployed: bool,
) -> None:
    if not deployed:
        return
    for record in truth.group_harm:
        values = totals[record.group]
        values[0] += record.harm_mass
        values[1] += record.exposure_mass


def _has_harmful_group(truth: ProtectedRoundTruth) -> bool:
    return any(
        record.harm_mass / record.exposure_mass > WORST_GROUP_LIMIT
        for record in truth.group_harm
    )


def _group_metric_rows(
    experiment: str,
    family: str,
    scenario: str,
    method: str,
    totals: dict[str, list[Fraction]],
) -> list[dict[str, object]]:
    rates = [
        harm_mass / exposure_mass
        for harm_mass, exposure_mass, _ in totals.values()
        if exposure_mass
    ]
    rows = [
        _descriptive_value_row(
            experiment,
            family,
            scenario,
            method,
            "worst_group_harm",
            float(max(rates)) if rates else None,
        ),
        _descriptive_value_row(
            experiment,
            family,
            scenario,
            method,
            "group_disparity",
            float(max(rates) - min(rates)) if rates else None,
        ),
    ]
    for group, (harm_mass, exposure_mass, eligible_exposure_mass) in sorted(
        totals.items()
    ):
        row = _descriptive_value_row(
            experiment,
            family,
            scenario,
            method,
            "group_harm",
            float(harm_mass / exposure_mass) if exposure_mass else None,
        )
        row.update(
            {
                "eligible_exposure_mass": _fraction_text(eligible_exposure_mass),
                "exposure_mass": _fraction_text(exposure_mass),
                "group": group,
                "harm_mass": _fraction_text(harm_mass),
            }
        )
        rows.append(row)
    return rows


def _run_restoration(
    replications: int,
    seed: int,
    alpha: float,
    settings: dict[str, dict[str, tuple[float, float]]],
    matched_pairs: dict[str, MatchedLifecyclePair],
) -> list[dict[str, object]]:
    counts: dict[tuple[str, str, str, int, str], int] = defaultdict(int)
    estimates: dict[tuple[str, str, int], float] = defaultdict(float)
    observed_harm: dict[tuple[str, str, int], int] = defaultdict(int)
    false_safe_lifecycles: dict[tuple[str, str], int] = defaultdict(int)
    genuine_acceptance: dict[tuple[str, str], int] = defaultdict(int)
    explicit_abstention: dict[tuple[str, str, str], int] = defaultdict(int)
    first_harm_sum: dict[tuple[str, str], int] = defaultdict(int)
    first_harm_count: dict[tuple[str, str], int] = defaultdict(int)
    utility_sum: dict[tuple[str, str, str], int] = defaultdict(int)
    for family in FAMILIES:
        for world in ("safe", "harmful"):
            probability, limit = settings[family][world]
            for replication in range(replications):
                random_source = _random(seed, "restoration", family, world, replication)
                harm_count = 0
                previous_n = 0
                false_safe = {"correct_width": False, "too_narrow": False}
                accepted_safe = {"correct_width": False, "too_narrow": False}
                determined = {"correct_width": False, "too_narrow": False}
                first_harm = {"correct_width": 0, "too_narrow": 0}
                for monitor_n in MONITOR_SIZES:
                    harm_count += sum(
                        random_source.random() < probability
                        for _ in range(monitor_n - previous_n)
                    )
                    previous_n = monitor_n
                    estimate = harm_count / monitor_n
                    observed_harm[(family, world, monitor_n)] += harm_count
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
                        accepted_safe[rule] |= world == "safe" and status == "safe"
                        determined[rule] |= status != "cannot_determine"
                        if world == "harmful" and status == "safe":
                            first_harm[rule] = first_harm[rule] or monitor_n
                for rule, occurred in false_safe.items():
                    false_safe_lifecycles[(family, rule)] += occurred
                    genuine_acceptance[(family, rule)] += accepted_safe[rule]
                    explicit_abstention[(family, world, rule)] += not determined[rule]
                    first_harm_sum[(family, rule)] += first_harm[rule]
                    first_harm_count[(family, rule)] += bool(first_harm[rule])
                    utility_sum[(family, world, rule)] += int(
                        accepted_safe[rule]
                    ) - int(occurred)

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
                group_harm_row = _rate_row(
                    "experiment_4",
                    family,
                    world,
                    rule,
                    "worst_group_harmful_lifecycle",
                    false_safe_lifecycles[(family, rule)] if world == "harmful" else 0,
                    replications,
                )
                group_harm_row["monitor_n"] = 0
                rows.append(group_harm_row)
                genuine_row = _rate_row(
                    "experiment_4",
                    family,
                    world,
                    rule,
                    "genuine_acceptance_lifecycle",
                    genuine_acceptance[(family, rule)] if world == "safe" else 0,
                    replications,
                )
                genuine_row["monitor_n"] = 0
                rows.append(genuine_row)
                abstention_row = _rate_row(
                    "experiment_4",
                    family,
                    world,
                    rule,
                    "explicit_abstention_lifecycle",
                    explicit_abstention[(family, world, rule)],
                    replications,
                )
                abstention_row["monitor_n"] = 0
                rows.append(abstention_row)
                first_harm_row = _mean_row(
                    "experiment_4",
                    family,
                    world,
                    rule,
                    "first_harm_monitor_n",
                    first_harm_sum[(family, rule)] if world == "harmful" else 0,
                    first_harm_count[(family, rule)] if world == "harmful" else 0,
                )
                first_harm_row["monitor_n"] = 0
                first_harm_row["sampling_scope"] = "independent_monitor_streams"
                first_harm_row["uncertainty"] = (
                    "descriptive_conditional_on_harmful_acceptance"
                )
                rows.append(first_harm_row)
                utility_row = _mean_row(
                    "experiment_4",
                    family,
                    world,
                    rule,
                    "final_utility",
                    utility_sum[(family, world, rule)],
                    replications,
                )
                utility_row["monitor_n"] = 0
                utility_row["sampling_scope"] = "independent_monitor_streams"
                utility_row["uncertainty"] = "descriptive_monitor_replications"
                rows.append(utility_row)
                lifecycle = (
                    matched_pairs[family].safe
                    if world == "safe"
                    else matched_pairs[family].harmful
                )
                accepted_lifecycles = (
                    genuine_acceptance[(family, rule)]
                    if world == "safe"
                    else false_safe_lifecycles[(family, rule)]
                )
                group_totals = _replicated_group_totals(
                    lifecycle.protected_truth,
                    accepted_lifecycles,
                    replications,
                )
                rows.extend(
                    _group_metric_rows(
                        "experiment_4",
                        family,
                        world,
                        rule,
                        group_totals,
                    )
                )
            for monitor_n in MONITOR_SIZES:
                observed_row = _rate_row(
                    "experiment_4",
                    family,
                    world,
                    "monitor_stream",
                    "observed_monitor_harm",
                    observed_harm[(family, world, monitor_n)],
                    replications * monitor_n,
                )
                observed_row["estimand"] = "pooled_worst_group_harm"
                observed_row["monitor_n"] = monitor_n
                observed_row["sampling_scope"] = "independent_monitor_observations"
                rows.append(observed_row)
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
            float(_find(rows, family, "mixed", method, "harmful_lifecycle")["estimate"])
            for method in CONTROLLED_METHODS
        )
        separated.append(float(sgm["estimate"]) >= best_controlled + 0.20)
    criteria = {
        "controlled_harm_at_most_0_05": all(
            float(row["estimate"]) <= 0.05 for row in controlled_harm
        ),
        "nontrivial_all_good_acceptance": all(
            any(
                float(row["estimate"]) > 0.30
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
                and float(safe["estimate"]) > 0.30
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


def _monitor_settings(
    matched_pairs: dict[str, MatchedLifecyclePair],
) -> dict[str, dict[str, tuple[float, float]]]:
    settings = {}
    for family in FAMILIES:
        pair = matched_pairs[family]
        settings[family] = {}
        for world, lifecycle in (("safe", pair.safe), ("harmful", pair.harmful)):
            settings[family][world] = (
                float(_pooled_worst_group_harm(lifecycle.protected_truth)),
                float(WORST_GROUP_LIMIT),
            )
    return settings


def _pooled_worst_group_harm(
    truths: tuple[ProtectedRoundTruth, ...],
) -> Fraction:
    totals = _group_totals(truths)
    for truth in truths:
        _add_deployed_group_truth(totals, truth, True)
    return max(harm / exposure for harm, exposure, _ in totals.values())


def _replicated_group_totals(
    truths: tuple[ProtectedRoundTruth, ...],
    deployed_lifecycles: int,
    eligible_lifecycles: int,
) -> dict[str, list[Fraction]]:
    totals = _group_totals(truths)
    for truth in truths:
        _add_deployed_group_truth(totals, truth, True)
    for values in totals.values():
        values[0] *= deployed_lifecycles
        values[1] *= deployed_lifecycles
        values[2] *= eligible_lifecycles
    return totals


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
    lower, upper = _clopper_pearson(successes, trials)
    return {
        "ci_lower": lower,
        "ci_upper": upper,
        "estimate": successes / trials,
        "experiment": experiment,
        "family": family,
        "method": method,
        "metric": metric,
        "scenario": scenario,
        "successes": successes,
        "trials": trials,
        "uncertainty": "exact_one_sided_95_clopper_pearson",
        "sampling_scope": "independent_monitor_streams",
    }


def _descriptive_rate_row(
    experiment: str,
    family: str,
    scenario: str,
    method: str,
    metric: str,
    successes: int,
    trials: int,
) -> dict[str, object]:
    return {
        "ci_lower": None,
        "ci_upper": None,
        "estimate": successes / trials if trials else None,
        "experiment": experiment,
        "family": family,
        "method": method,
        "metric": metric,
        "scenario": scenario,
        "successes": successes,
        "trials": trials,
        "uncertainty": "descriptive_fixed_lifecycle",
        "sampling_scope": "fixed_lifecycle_witness",
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
        "estimate": value_sum / trials if trials else None,
        "experiment": experiment,
        "family": family,
        "method": method,
        "metric": metric,
        "scenario": scenario,
        "trials": trials,
        "value_sum": value_sum,
    }


def _descriptive_mean_row(
    experiment: str,
    family: str,
    scenario: str,
    method: str,
    metric: str,
    value_sum: float,
    trials: int,
) -> dict[str, object]:
    row = _mean_row(
        experiment,
        family,
        scenario,
        method,
        metric,
        value_sum,
        trials,
    )
    row["uncertainty"] = "descriptive_fixed_lifecycle"
    row["sampling_scope"] = "fixed_lifecycle_witness"
    return row


def _descriptive_value_row(
    experiment: str,
    family: str,
    scenario: str,
    method: str,
    metric: str,
    estimate: float | None,
) -> dict[str, object]:
    return {
        "ci_lower": None,
        "ci_upper": None,
        "estimate": estimate,
        "experiment": experiment,
        "family": family,
        "method": method,
        "metric": metric,
        "sampling_scope": "fixed_lifecycle_witness",
        "scenario": scenario,
        "trials": 1,
        "uncertainty": "descriptive_fixed_lifecycle",
    }


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("exact binomial bounds require 0 <= successes <= trials")
    upper = _clopper_pearson_upper(successes, trials)
    lower = 1 - _clopper_pearson_upper(trials - successes, trials)
    return max(0.0, lower), min(1.0, upper)


def _clopper_pearson_upper(successes: int, trials: int) -> float:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("exact binomial bounds require 0 <= successes <= trials")
    if successes == trials:
        return 1.0
    alpha = 1 - ONE_SIDED_CONFIDENCE
    if successes == 0:
        value = -math.expm1(math.log(alpha) / trials)
        return math.nextafter(value, 1.0)

    target = math.log(alpha)
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        if _log_binomial_cdf(successes, trials, middle) > target:
            low = middle
        else:
            high = middle
    return math.nextafter(high, 1.0)


def _log_binomial_cdf(successes: int, trials: int, probability: float) -> float:
    if probability <= 0:
        return 0.0
    if probability >= 1:
        return -math.inf if successes < trials else 0.0
    value = _regularized_beta(
        1 - probability,
        trials - successes,
        successes + 1,
    )
    return math.log(value) if value else -math.inf


def _regularized_beta(value: float, first: float, second: float) -> float:
    if value <= 0:
        return 0.0
    if value >= 1:
        return 1.0
    log_factor = (
        math.lgamma(first + second)
        - math.lgamma(first)
        - math.lgamma(second)
        + first * math.log(value)
        + second * math.log1p(-value)
    )
    direct = value < (first + 1) / (first + second + 2)
    if log_factor < -745:
        return 0.0 if direct else 1.0
    if direct:
        return (
            math.exp(log_factor)
            * _beta_continued_fraction(first, second, value)
            / first
        )
    complement = (
        math.exp(log_factor)
        * _beta_continued_fraction(second, first, 1 - value)
        / second
    )
    return max(0.0, 1 - complement)


def _beta_continued_fraction(first: float, second: float, value: float) -> float:
    total = first + second
    first_plus_one = first + 1
    first_minus_one = first - 1
    fraction = 1 - total * value / first_plus_one
    fraction = fraction if abs(fraction) > 1e-300 else 1e-300
    fraction = 1 / fraction
    numerator = 1.0
    result = fraction
    for index in range(1, 1001):
        doubled = 2 * index
        coefficient = (
            index
            * (second - index)
            * value
            / ((first_minus_one + doubled) * (first + doubled))
        )
        fraction = 1 + coefficient * fraction
        fraction = fraction if abs(fraction) > 1e-300 else 1e-300
        numerator = 1 + coefficient / numerator
        numerator = numerator if abs(numerator) > 1e-300 else 1e-300
        fraction = 1 / fraction
        result *= fraction * numerator
        coefficient = (
            -(first + index)
            * (total + index)
            * value
            / ((first + doubled) * (first_plus_one + doubled))
        )
        fraction = 1 + coefficient * fraction
        fraction = fraction if abs(fraction) > 1e-300 else 1e-300
        numerator = 1 + coefficient / numerator
        numerator = numerator if abs(numerator) > 1e-300 else 1e-300
        fraction = 1 / fraction
        change = fraction * numerator
        result *= change
        if abs(change - 1) <= 3e-14:
            return result
    raise ArithmeticError("exact binomial bound did not converge")


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
