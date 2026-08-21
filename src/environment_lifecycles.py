from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from .authgate import (
    Policy,
    World,
    generate_authgate_instance,
    generated_audit_record,
    generated_truth_record,
)
from .batch_triage import (
    BATCH_TRIAGE_FAMILY,
    Policy as BatchPolicy,
    evaluate_all as evaluate_batch_triage,
    evaluator_input as batch_triage_evaluator_input,
    generate_batch_triage_instance,
)
from .constraint_plan import (
    evaluate_all,
    evaluator_input,
    generate_instances,
    load_templates,
)
from .decision_methods import ComponentEvidence, MonitorEvidence, UpdateEvidence
from .lifecycle import (
    AUTH_FAMILY,
    HARM_TOLERANCE,
    PLAN_FAMILY,
    QUALITY_FLOOR,
    WORST_GROUP_LIMIT,
)

LIFECYCLE_LENGTH = 50
SCENARIOS = ("null_only", "all_good", "mixed")
FAMILIES = (AUTH_FAMILY, PLAN_FAMILY, BATCH_TRIAGE_FAMILY)
_PLANNING_TEMPLATES = (
    Path(__file__).resolve().parents[1] / "data" / "planning_templates.json"
)


@dataclass(frozen=True)
class ExactMetric:
    name: str
    value: Fraction


@dataclass(frozen=True)
class PublicUpdateRound:
    index: int
    environment_ids: tuple[str, ...]
    proposal_id: str
    update: UpdateEvidence
    failure_modes: tuple[str, ...]


@dataclass(frozen=True)
class ProtectedRoundTruth:
    world: str
    safe_to_deploy: bool
    monitor: MonitorEvidence | None
    metrics: tuple[ExactMetric, ...]
    failure_modes: tuple[str, ...]


@dataclass(frozen=True)
class RealizedLifecycle:
    family: str
    scenario: str
    seed: int
    public_rounds: tuple[PublicUpdateRound, ...]
    protected_truth: tuple[ProtectedRoundTruth, ...]

    def __post_init__(self) -> None:
        if len(self.public_rounds) != LIFECYCLE_LENGTH:
            raise ValueError("a lifecycle must contain exactly 50 public rounds")
        if len(self.protected_truth) != LIFECYCLE_LENGTH:
            raise ValueError("a lifecycle must contain exactly 50 truth records")


@dataclass(frozen=True)
class MatchedLifecyclePair:
    safe: RealizedLifecycle
    harmful: RealizedLifecycle

    def __post_init__(self) -> None:
        if self.safe.public_rounds is not self.harmful.public_rounds:
            raise ValueError("matched lifecycles must share their public rounds")


@dataclass(frozen=True)
class _BuiltRound:
    public: PublicUpdateRound
    safe: ProtectedRoundTruth
    harmful: ProtectedRoundTruth


def realize_lifecycle(
    family: str,
    scenario: str,
    seed: int,
    *,
    planning_templates: Path = _PLANNING_TEMPLATES,
) -> RealizedLifecycle:
    """Realize one deterministic 50-update exact-environment lifecycle."""
    _validate_inputs(family, scenario, seed)
    template_text = (
        planning_templates.read_text(encoding="utf-8") if family == PLAN_FAMILY else ""
    )
    rounds = _build_rounds(
        family,
        scenario,
        seed,
        str(planning_templates),
        template_text,
    )
    truths = tuple(
        item.safe
        if _round_world(scenario, item.public.index) == "safe"
        else item.harmful
        for item in rounds
    )
    expected_safe = 50 if scenario == "all_good" else 15 if scenario == "mixed" else 0
    if sum(truth.safe_to_deploy for truth in truths) != expected_safe:
        raise ValueError(
            f"{family}/{scenario} did not realize its intended truth schedule"
        )
    return RealizedLifecycle(
        family,
        scenario,
        seed,
        tuple(item.public for item in rounds),
        truths,
    )


def realize_matched_pair(
    family: str,
    seed: int,
    *,
    planning_templates: Path = _PLANNING_TEMPLATES,
) -> MatchedLifecyclePair:
    """Realize audit-identical safe and harmful lifecycles for Experiment 3."""
    _validate_inputs(family, "all_good", seed)
    template_text = (
        planning_templates.read_text(encoding="utf-8") if family == PLAN_FAMILY else ""
    )
    rounds = _build_rounds(
        family,
        "matched",
        seed,
        str(planning_templates),
        template_text,
    )
    public = tuple(item.public for item in rounds)
    safe = RealizedLifecycle(
        family,
        "matched_safe",
        seed,
        public,
        tuple(item.safe for item in rounds),
    )
    harmful = RealizedLifecycle(
        family,
        "matched_harmful",
        seed,
        public,
        tuple(item.harmful for item in rounds),
    )
    if not all(truth.safe_to_deploy for truth in safe.protected_truth):
        raise ValueError(f"{family} matched safe truth is not deployable")
    if any(truth.safe_to_deploy for truth in harmful.protected_truth):
        raise ValueError(f"{family} matched harmful truth is deployable")
    return MatchedLifecyclePair(safe, harmful)


def _validate_inputs(family: str, scenario: str, seed: int) -> None:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    if type(seed) is not int:
        raise ValueError("lifecycle seed must be an integer")


def _round_world(scenario: str, index: int) -> str:
    if scenario == "all_good" or scenario == "mixed" and index < 15:
        return "safe"
    return "harmful"


@lru_cache(maxsize=64)
def _build_rounds(
    family: str,
    scenario: str,
    seed: int,
    planning_templates: str,
    template_text: str,
) -> tuple[_BuiltRound, ...]:
    builders = {
        AUTH_FAMILY: _authgate_round,
        BATCH_TRIAGE_FAMILY: _batch_triage_round,
        PLAN_FAMILY: _planning_round,
    }
    builder = builders[family]
    return tuple(
        builder(scenario, seed, index, planning_templates, template_text)
        for index in range(LIFECYCLE_LENGTH)
    )


def _authgate_round(
    scenario: str,
    seed: int,
    index: int,
    _planning_templates: str,
    _template_text: str,
) -> _BuiltRound:
    combined_null = (
        _round_world(scenario, index) == "harmful"
        and index % 9 == 7
        and scenario != "matched"
    )
    repeated = index % 5 != 0
    bank_index = index // 5
    start = _bounded_seed(seed, bank_index, 101)
    candidates = []
    candidate_seed = start
    while len(candidates) < 32:
        instance = generate_authgate_instance("development", candidate_seed)
        candidate = generated_audit_record(instance, Policy.CANDIDATE, World.SAFE)
        incumbent = generated_audit_record(instance, Policy.INCUMBENT, World.SAFE)
        harm_margin = HARM_TOLERANCE - Fraction(str(candidate["audit_harm"]))
        quality_margin = Fraction(str(candidate["task_quality"])) - QUALITY_FLOOR
        passing = harm_margin > 0 and quality_margin > 0
        if passing != combined_null:
            candidates.append(
                (instance, candidate, incumbent, harm_margin, quality_margin)
            )
        candidate_seed = (candidate_seed + 1) % (2**31)
        if candidate_seed == start:
            raise ValueError("no suitable generated AuthGate instances")
    candidates.sort(
        key=lambda item: (min(item[3], item[4]), item[0]["instance_id"]), reverse=True
    )
    batch = candidates[:32]
    update_id = f"auth-{seed}-{index}"
    update = UpdateEvidence(
        AUTH_FAMILY,
        update_id,
        (
            _component("harm", [_signed(item[3]) for item in batch]),
            _component("quality", [_signed(item[4]) for item in batch]),
        ),
        tuple(
            int(harm_margin > 0 and quality_margin > 0)
            for _, _, incumbent, harm_margin, quality_margin in batch
            if (harm_margin > 0 and quality_margin > 0) != _authgate_passes(incumbent)
        ),
    )
    public_modes = ["correlated_updates", "adaptive_public_selection"]
    if repeated:
        public_modes.extend(("repeated_similar_updates", "partial_evidence_transfer"))
    if combined_null:
        public_modes.append("incorrect_combined_null_stress")
    public = PublicUpdateRound(
        index,
        tuple(str(item[0]["instance_id"]) for item in batch),
        Policy.CANDIDATE.value,
        update,
        tuple(public_modes),
    )
    safe_rows = [
        generated_truth_record(item[0], Policy.CANDIDATE, World.SAFE) for item in batch
    ]
    harmful_rows = [
        generated_truth_record(item[0], Policy.CANDIDATE, World.HARMFUL)
        for item in batch
    ]
    return _BuiltRound(
        public,
        _authgate_truth(update, safe_rows, index),
        _authgate_truth(update, harmful_rows, index),
    )


def _authgate_passes(record: dict[str, object]) -> bool:
    return (
        Fraction(str(record["audit_harm"])) <= HARM_TOLERANCE
        and Fraction(str(record["task_quality"])) >= QUALITY_FLOOR
    )


def _authgate_truth(
    update: UpdateEvidence,
    rows: list[dict[str, object]],
    index: int,
) -> ProtectedRoundTruth:
    live_harm = [Fraction(str(row["live_harm"])) for row in rows]
    audit_harm = [Fraction(str(row["audit_harm"])) for row in rows]
    quality = [Fraction(str(row["task_quality"])) for row in rows]
    group_harm = [
        Fraction(str(value))
        for row in rows
        for value in dict(row["group_live_harm"]).values()
    ]
    safe = (
        all(
            harm <= HARM_TOLERANCE and score >= QUALITY_FLOOR
            for harm, score in zip(live_harm, quality, strict=True)
        )
        and max(group_harm) <= WORST_GROUP_LIMIT
    )
    monitor_rows = list(range(len(rows)))
    modes = []
    if index % 11 == 6:
        monitor_rows = sorted(
            monitor_rows,
            key=lambda item: HARM_TOLERANCE - live_harm[item],
            reverse=True,
        )[:2]
        modes.append("selection_biased_monitoring")
    if index >= 40:
        monitor_rows = [item for item in monitor_rows if item % 2] or monitor_rows
        modes.append("monitor_shift")
    monitor = MonitorEvidence(
        update.family,
        update.update_id,
        (
            ComponentEvidence(
                "harm",
                tuple(
                    _signed(HARM_TOLERANCE - live_harm[item]) for item in monitor_rows
                ),
            ),
            ComponentEvidence(
                "quality",
                tuple(_signed(quality[item] - QUALITY_FLOOR) for item in monitor_rows),
            ),
            ComponentEvidence(
                "worst_group",
                tuple(
                    _signed(
                        WORST_GROUP_LIMIT
                        - max(
                            Fraction(str(value))
                            for value in dict(rows[item]["group_live_harm"]).values()
                        )
                    )
                    for item in monitor_rows
                ),
            ),
        ),
    )
    if index % 11 == 4:
        monitor = None
        modes.append("missing_monitoring")
    if any(harm > audit for harm, audit in zip(live_harm, audit_harm, strict=True)):
        modes.append("delayed_harm")
    if group_harm and max(group_harm) > min(group_harm):
        modes.append("group_dependent_drift")
    return ProtectedRoundTruth(
        str(rows[0]["world"]),
        safe,
        monitor,
        (
            ExactMetric("audit_harm", sum(audit_harm) / len(audit_harm)),
            ExactMetric("live_harm", sum(live_harm) / len(live_harm)),
            ExactMetric("task_quality", sum(quality) / len(quality)),
            ExactMetric("worst_group_harm", max(group_harm)),
        ),
        tuple(modes),
    )


@lru_cache(maxsize=32)
def _batch_triage_bank(
    seed: int,
) -> tuple[tuple[dict[str, object], list[dict[str, object]]], ...]:
    bank = []
    for index in range(LIFECYCLE_LENGTH):
        split = "diagnostic" if index >= 40 else "development"
        generation_seed = _bounded_seed(seed, index, 211)
        instance = generate_batch_triage_instance(split, generation_seed)
        rows = evaluate_batch_triage(batch_triage_evaluator_input(instance))
        bank.append((instance, rows))
    return tuple(bank)


def _batch_triage_round(
    _scenario: str,
    seed: int,
    index: int,
    _planning_templates: str,
    _template_text: str,
) -> _BuiltRound:
    instance, rows = _batch_triage_bank(seed)[index]
    candidate = _result(rows, BatchPolicy.CANDIDATE.value, "safe")
    incumbent = _result(rows, BatchPolicy.INCUMBENT.value, "safe")
    candidate_audit = dict(candidate["audit"])
    incumbent_audit = dict(incumbent["audit"])
    candidate_completion = Fraction(str(candidate_audit["completion_rate"]))
    incumbent_completion = Fraction(str(incumbent_audit["completion_rate"]))
    candidate_cost = Fraction(str(candidate_audit["service_cost"]))
    incumbent_cost = Fraction(str(incumbent_audit["service_cost"]))
    completion_noninferior = candidate_completion >= incumbent_completion
    cost_improvement = (incumbent_cost - candidate_cost) / incumbent_cost
    update = UpdateEvidence(
        BATCH_TRIAGE_FAMILY,
        f"batch-{seed}-{index}",
        (
            _component(
                "completion",
                [1.0 if completion_noninferior else -1.0],
                require_all=True,
            ),
            _component("service_cost", [float(cost_improvement)]),
        ),
        (int(completion_noninferior and cost_improvement > 0),),
    )
    modes = ["correlated_updates"]
    if index % 5:
        modes.extend(("repeated_similar_updates", "partial_evidence_transfer"))
    public = PublicUpdateRound(
        index,
        (str(instance["instance_id"]),),
        BatchPolicy.CANDIDATE.value,
        update,
        tuple(modes),
    )
    safe = _result(rows, BatchPolicy.CANDIDATE.value, "safe")
    harmful = _result(rows, BatchPolicy.CANDIDATE.value, "harmful")
    safe_incumbent = _result(rows, BatchPolicy.INCUMBENT.value, "safe")
    harmful_incumbent = _result(rows, BatchPolicy.INCUMBENT.value, "harmful")
    return _BuiltRound(
        public,
        _batch_triage_truth(update, safe, safe_incumbent),
        _batch_triage_truth(update, harmful, harmful_incumbent),
    )


def _batch_triage_truth(
    update: UpdateEvidence,
    row: dict[str, object],
    incumbent: dict[str, object],
) -> ProtectedRoundTruth:
    overall_harm = Fraction(str(row["overall_harm"]))
    group_harm = [Fraction(str(value)) for value in dict(row["group_harm"]).values()]
    worst_group_harm = max(group_harm)
    backlog_probability = Fraction(str(row["backlog_probability"]))
    completed = Fraction(str(row["completed_per_batch"]))
    incumbent_completed = Fraction(str(incumbent["completed_per_batch"]))
    completion_noninferior = completed >= incumbent_completed
    safe = (
        overall_harm <= HARM_TOLERANCE
        and worst_group_harm <= WORST_GROUP_LIMIT
        and completion_noninferior
    )
    monitor = MonitorEvidence(
        update.family,
        update.update_id,
        (
            ComponentEvidence(
                "overall_harm",
                (1.0 if overall_harm <= HARM_TOLERANCE else -1.0,),
                require_all=True,
            ),
            ComponentEvidence(
                "worst_group",
                (1.0 if worst_group_harm <= WORST_GROUP_LIMIT else -1.0,),
                require_all=True,
            ),
            ComponentEvidence(
                "completion",
                (1.0 if completion_noninferior else -1.0,),
                require_all=True,
            ),
        ),
    )
    metrics = [
        ExactMetric("overall_harm", overall_harm),
        ExactMetric("worst_group_harm", worst_group_harm),
        ExactMetric("backlog_probability", backlog_probability),
        ExactMetric("completed_per_batch", completed),
    ]
    first_harm_time = row["first_harm_time"]
    modes = []
    if first_harm_time is not None:
        metrics.append(ExactMetric("first_harm_time", Fraction(first_harm_time)))
        modes.append("delayed_harm")
    if max(group_harm) > min(group_harm):
        modes.append("group_dependent_drift")
    if backlog_probability > 0:
        modes.append("shared_capacity_interference")
    return ProtectedRoundTruth(
        str(row["world"]),
        safe,
        monitor,
        tuple(metrics),
        tuple(modes),
    )


@lru_cache(maxsize=16)
def _templates(path: str, _template_text: str) -> tuple[dict[str, object], ...]:
    return tuple(load_templates(Path(path)))


def _planning_round(
    scenario: str,
    seed: int,
    index: int,
    planning_templates: str,
    template_text: str,
) -> _BuiltRound:
    repeated = index % 5 != 0
    evaluated = list(
        _planning_bank(seed, index // 5, planning_templates, template_text)
    )
    harmful_round = _round_world(scenario, index) == "harmful"
    combined_null = harmful_round and index % 9 == 7 and scenario != "matched"
    if scenario == "matched":
        proposal, adaptive = "p4", False
    else:
        proposal, adaptive = _planning_proposal(
            evaluated, index, harmful_round, combined_null
        )
    update_id = f"plan-{seed}-{index}"
    public_results = [
        (
            _result(rows, proposal, "safe")["public"],
            _result(rows, "p0", "safe")["public"],
        )
        for _, rows in evaluated
    ]
    comparisons = [
        _comparison(candidate, incumbent) for candidate, incumbent in public_results
    ]
    update = UpdateEvidence(
        PLAN_FAMILY,
        update_id,
        (
            _component(
                "public_feasibility",
                [
                    1.0 if bool(candidate["feasible"]) else -1.0
                    for candidate, _ in public_results
                ],
                require_all=True,
            ),
            _component("objective", [float(value) for value in comparisons]),
        ),
        tuple(int(value > 0) for value in comparisons if value),
    )
    public_modes = ["correlated_updates"]
    if adaptive:
        public_modes.append("adaptive_public_selection")
    if repeated:
        public_modes.extend(("repeated_similar_updates", "partial_evidence_transfer"))
    if combined_null:
        public_modes.append("incorrect_combined_null_stress")
    public = PublicUpdateRound(
        index,
        tuple(str(instance["instance_id"]) for instance, _ in evaluated),
        proposal,
        update,
        tuple(public_modes),
    )
    safe_rows = [_result(rows, proposal, "safe") for _, rows in evaluated]
    harmful_rows = [_result(rows, proposal, "harmful") for _, rows in evaluated]
    safe_incumbents = [_result(rows, "p0", "safe") for _, rows in evaluated]
    harmful_incumbents = [_result(rows, "p0", "harmful") for _, rows in evaluated]
    splits = [str(instance["split"]) for instance, _ in evaluated]
    return _BuiltRound(
        public,
        _planning_truth(update, safe_rows, safe_incumbents, splits, index),
        _planning_truth(update, harmful_rows, harmful_incumbents, splits, index),
    )


@lru_cache(maxsize=64)
def _planning_bank(
    seed: int,
    bank_index: int,
    planning_templates: str,
    template_text: str,
) -> tuple[tuple[dict[str, object], list[dict[str, object]]], ...]:
    evaluated = []
    for batch in range(2):
        generation_seed = seed + bank_index * 101 + batch * 10_007
        instances = generate_instances(
            list(_templates(planning_templates, template_text)),
            seed=generation_seed,
            instances_per_template=5,
        )
        evaluated.extend(
            (instance, evaluate_all(evaluator_input(instance)))
            for instance in instances
        )
    return tuple(evaluated)


def _planning_proposal(
    evaluated: list[tuple[dict[str, object], list[dict[str, object]]]],
    index: int,
    harmful_round: bool,
    combined_null: bool,
) -> tuple[str, bool]:
    if not harmful_round:
        return "p1", False
    if combined_null:
        return "p3", False
    forced = {3: "p7", 5: "p4", 6: "p5"}.get(index % 9)
    if forced:
        return forced, False
    candidates = ("p4", "p5", "p6", "p7")
    scores = {}
    for proposal in candidates:
        values = [
            _comparison(
                _result(rows, proposal, "safe")["public"],
                _result(rows, "p0", "safe")["public"],
            )
            for _, rows in evaluated
        ]
        scores[proposal] = (sum(values), min(values))
    return max(candidates, key=lambda proposal: (scores[proposal], proposal)), True


def _planning_truth(
    update: UpdateEvidence,
    rows: list[dict[str, object]],
    incumbents: list[dict[str, object]],
    splits: list[str],
    index: int,
) -> ProtectedRoundTruth:
    comparisons = [
        _comparison(row["full"], incumbent["full"])
        for row, incumbent in zip(rows, incumbents, strict=True)
    ]
    feasible = [bool(dict(row["full"])["feasible"]) for row in rows]
    rare_harm = [Fraction(str(row["rare_group_harm"])) for row in rows]
    safe = all(
        valid and comparison >= 0 and harm <= WORST_GROUP_LIMIT
        for valid, comparison, harm in zip(
            feasible, comparisons, rare_harm, strict=True
        )
    )
    monitor_rows = list(range(len(rows)))
    modes = []
    if index % 11 == 6:
        monitor_rows = sorted(
            monitor_rows,
            key=lambda item: (feasible[item], comparisons[item], -rare_harm[item]),
            reverse=True,
        )[:2]
        modes.append("selection_biased_monitoring")
    if index >= 40:
        shifted = [item for item, split in enumerate(splits) if split == "diagnostic"]
        monitor_rows = shifted or monitor_rows
        modes.append("monitor_shift")
    monitor = MonitorEvidence(
        update.family,
        update.update_id,
        (
            ComponentEvidence(
                "full_feasibility",
                tuple(1.0 if feasible[item] else -1.0 for item in monitor_rows),
                require_all=True,
            ),
            ComponentEvidence(
                "objective", tuple(float(comparisons[item]) for item in monitor_rows)
            ),
            ComponentEvidence(
                "worst_group",
                tuple(
                    _signed(WORST_GROUP_LIMIT - rare_harm[item])
                    for item in monitor_rows
                ),
            ),
        ),
    )
    if index % 11 == 4:
        monitor = None
        modes.append("missing_monitoring")
    delayed = sum(row["delayed_violation_slot"] is not None for row in rows)
    if delayed:
        modes.append("delayed_harm")
    if any(rare_harm):
        modes.append("group_dependent_drift")
    return ProtectedRoundTruth(
        str(rows[0]["world"]),
        safe,
        monitor,
        (
            ExactMetric("full_feasible_rate", Fraction(sum(feasible), len(feasible))),
            ExactMetric(
                "objective_noninferior_rate",
                Fraction(sum(value >= 0 for value in comparisons), len(comparisons)),
            ),
            ExactMetric("rare_group_harm", sum(rare_harm) / len(rare_harm)),
            ExactMetric("delayed_violation_rate", Fraction(delayed, len(rows))),
        ),
        tuple(modes),
    )


def _component(
    name: str,
    values: list[float],
    *,
    require_all: bool = False,
) -> ComponentEvidence:
    split = max(1, 2 * len(values) // 3)
    return ComponentEvidence(
        name,
        tuple(values[:split]),
        tuple(values[split:]),
        require_all,
    )


def _result(
    rows: list[dict[str, object]], proposal: str, world: str
) -> dict[str, object]:
    return next(
        row for row in rows if row["policy_id"] == proposal and row["world"] == world
    )


def _comparison(candidate: object, incumbent: object) -> int:
    candidate_result = dict(candidate)  # type: ignore[arg-type]
    incumbent_result = dict(incumbent)  # type: ignore[arg-type]
    candidate_feasible = bool(candidate_result["feasible"])
    incumbent_feasible = bool(incumbent_result["feasible"])
    if candidate_feasible != incumbent_feasible:
        return 1 if candidate_feasible else -1
    if not candidate_feasible:
        return 0
    candidate_objective = tuple(candidate_result["objective"])
    incumbent_objective = tuple(incumbent_result["objective"])
    return (candidate_objective < incumbent_objective) - (
        candidate_objective > incumbent_objective
    )


def _bounded_seed(seed: int, index: int, stride: int) -> int:
    return (seed + index * stride) % (2**31)


def _signed(value: Fraction) -> float:
    return 1.0 if value > 0 else -1.0
