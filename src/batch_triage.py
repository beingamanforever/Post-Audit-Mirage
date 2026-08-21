from __future__ import annotations

import copy
import re
from collections import defaultdict
from enum import StrEnum
from fractions import Fraction

BATCH_TRIAGE_FAMILY = "batch_triage_v0"

_EVALUATOR_FIELDS = {
    "capacity",
    "family",
    "groups",
    "harm_delay",
    "horizon",
    "policies",
    "worlds",
}
_PROVENANCE_FIELDS = _EVALUATOR_FIELDS | {
    "instance_id",
    "request_text",
    "seed",
    "split",
}
_GROUP_FIELDS = {"arrival_probability", "deadline", "group_id"}
_POLICY_FIELDS = {"policy_id", "priority", "service_cost"}
_SCENARIO_FIELDS = {"common", "probability", "rare"}
_FRACTION = re.compile(r"(0|[1-9][0-9]*)/([1-9][0-9]*)\Z")
_MAX_VALUE = 2**31 - 1
_MAX_FRACTION_DIGITS = 40


class World(StrEnum):
    SAFE = "safe"
    HARMFUL = "harmful"


class Policy(StrEnum):
    CANDIDATE = "candidate"
    INCUMBENT = "incumbent"


def generate_batch_triage_instance(
    split: str | int,
    seed: int | None = None,
) -> dict[str, object]:
    if seed is None:
        seed = split
        split = "development"
    if split not in {"development", "diagnostic"}:
        raise ValueError("BatchTriage split must be development or diagnostic")
    if type(seed) is not int or not 0 <= seed <= _MAX_VALUE:
        raise ValueError(
            "BatchTriage seed must be an integer from 0 through 2147483647"
        )

    if split == "development":
        common_probabilities = (Fraction(1, 2), Fraction(3, 5), Fraction(2, 5))
        rare_probabilities = (Fraction(1, 20), Fraction(1, 10), Fraction(1, 8))
        common_deadline = 2 + (seed // 9) % 2
        harm_delay = 1 + (seed // 18) % 2
        candidate_costs = (Fraction(1), Fraction(3, 2))
    else:
        common_probabilities = (Fraction(4, 7), Fraction(5, 8), Fraction(7, 10))
        rare_probabilities = (Fraction(1, 16), Fraction(1, 12), Fraction(1, 7))
        common_deadline = 3 + (seed // 9) % 2
        harm_delay = 2 + (seed // 18) % 2
        candidate_costs = (Fraction(4, 3), Fraction(5, 3))
    common_probability = common_probabilities[seed % 3]
    rare_probability = rare_probabilities[(seed // 3) % 3]
    candidate_cost = candidate_costs[(seed // 36) % 2]
    candidate_priority = "random" if split == "development" else "common_first"
    incumbent_cost = candidate_cost + 1

    safe = _joint_scenarios(
        common_only=common_probability,
        rare_only=rare_probability,
        both=Fraction(0),
        neither=1 - common_probability - rare_probability,
    )
    harmful = _joint_scenarios(
        common_only=common_probability - rare_probability,
        rare_only=Fraction(0),
        both=rare_probability,
        neither=1 - common_probability,
    )
    if seed % 2:
        safe.reverse()
    if (seed // 2) % 2:
        harmful = harmful[2:] + harmful[:2]

    evaluator = {
        "capacity": 1,
        "family": BATCH_TRIAGE_FAMILY,
        "groups": [
            {
                "arrival_probability": _fraction_text(common_probability),
                "deadline": common_deadline,
                "group_id": "common",
            },
            {
                "arrival_probability": _fraction_text(rare_probability),
                "deadline": 1,
                "group_id": "rare",
            },
        ],
        "harm_delay": harm_delay,
        "horizon": common_deadline + harm_delay,
        "policies": [
            {
                "policy_id": Policy.INCUMBENT.value,
                "priority": "earliest_deadline",
                "service_cost": _fraction_text(incumbent_cost),
            },
            {
                "policy_id": Policy.CANDIDATE.value,
                "priority": candidate_priority,
                "service_cost": _fraction_text(candidate_cost),
            },
        ],
        "worlds": {
            World.SAFE.value: safe,
            World.HARMFUL.value: harmful,
        },
    }
    instance = {
        **evaluator,
        "instance_id": f"batch-triage-{split}-{seed}",
        "request_text": f"Generated shared-capacity triage batch {seed}.",
        "seed": seed,
        "split": split,
    }
    _validate_evaluator(evaluator)
    return instance


def anchor_instance() -> dict[str, object]:
    return generate_batch_triage_instance("development", 0)


def evaluator_input(instance: dict[str, object]) -> dict[str, object]:
    if not isinstance(instance, dict) or set(instance) != _PROVENANCE_FIELDS:
        raise ValueError("BatchTriage provenance has unexpected fields")
    try:
        expected = generate_batch_triage_instance(instance["split"], instance["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("BatchTriage provenance is invalid") from error
    if instance != expected:
        raise ValueError("BatchTriage instance does not match its provenance")
    evaluator = {field: copy.deepcopy(instance[field]) for field in _EVALUATOR_FIELDS}
    _validate_evaluator(evaluator)
    return evaluator


def audit_record(
    instance: dict[str, object], policy: Policy | str
) -> dict[str, object]:
    evaluator = (
        evaluator_input(instance)
        if isinstance(instance, dict) and set(instance) == _PROVENANCE_FIELDS
        else instance
    )
    groups, policies, _ = _validate_evaluator(evaluator)
    selected = _select_policy(policies, policy)
    return {
        "audit": _audit(groups, selected),
        "policy_id": selected["policy_id"],
    }


def evaluate_all(evaluator: dict[str, object]) -> list[dict[str, object]]:
    groups, policies, worlds = _validate_evaluator(evaluator)
    rows = []
    for policy in policies:
        audit = _audit(groups, policy)
        for world in World:
            metrics = _evaluate_world(evaluator, groups, policy, worlds[world.value])
            rows.append(
                {
                    "audit": audit,
                    **metrics,
                    "policy_id": policy["policy_id"],
                    "world": world.value,
                }
            )
    return sorted(rows, key=lambda row: (row["policy_id"], row["world"]))


def _joint_scenarios(
    *,
    common_only: Fraction,
    rare_only: Fraction,
    both: Fraction,
    neither: Fraction,
) -> list[dict[str, object]]:
    return [
        {
            "common": True,
            "probability": _fraction_text(common_only),
            "rare": False,
        },
        {
            "common": False,
            "probability": _fraction_text(rare_only),
            "rare": True,
        },
        {
            "common": True,
            "probability": _fraction_text(both),
            "rare": True,
        },
        {
            "common": False,
            "probability": _fraction_text(neither),
            "rare": False,
        },
    ]


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _parse_fraction(
    value: object,
    field: str,
    *,
    maximum: Fraction | None = Fraction(1),
) -> Fraction:
    if not isinstance(value, str) or not _FRACTION.fullmatch(value):
        raise ValueError(f"{field} must be a canonical exact fraction string")
    numerator, denominator = value.split("/")
    if max(len(numerator), len(denominator)) > _MAX_FRACTION_DIGITS:
        raise ValueError(f"{field} exceeds the fraction size bound")
    fraction = Fraction(value)
    if _fraction_text(fraction) != value:
        raise ValueError(f"{field} must be a reduced exact fraction string")
    if fraction < 0 or maximum is not None and fraction > maximum:
        bound = "non-negative" if maximum is None else f"within [0, {maximum}]"
        raise ValueError(f"{field} must be {bound}")
    return fraction


def _integer(value: object, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer from {minimum} to {maximum}")
    return value


def _validate_evaluator(
    evaluator: object,
) -> tuple[
    dict[str, dict[str, object]],
    tuple[dict[str, object], ...],
    dict[str, tuple[tuple[bool, bool, Fraction], ...]],
]:
    if not isinstance(evaluator, dict) or set(evaluator) != _EVALUATOR_FIELDS:
        raise ValueError("BatchTriage evaluator input has unexpected fields")
    if evaluator["family"] != BATCH_TRIAGE_FAMILY:
        raise ValueError("BatchTriage family is invalid")
    capacity = _integer(evaluator["capacity"], "capacity", 1, 2)
    harm_delay = _integer(evaluator["harm_delay"], "harm_delay", 1, 32)
    horizon = _integer(evaluator["horizon"], "horizon", 2, 96)

    raw_groups = evaluator["groups"]
    if not isinstance(raw_groups, list) or len(raw_groups) != 2:
        raise ValueError("BatchTriage must contain common and rare groups")
    groups: dict[str, dict[str, object]] = {}
    for group in raw_groups:
        if not isinstance(group, dict) or set(group) != _GROUP_FIELDS:
            raise ValueError("BatchTriage group has unexpected fields")
        group_id = group["group_id"]
        if group_id not in {"common", "rare"} or group_id in groups:
            raise ValueError("BatchTriage groups must be unique common and rare groups")
        groups[group_id] = {
            "arrival_probability": _parse_fraction(
                group["arrival_probability"], f"{group_id} arrival_probability"
            ),
            "deadline": _integer(group["deadline"], f"{group_id} deadline", 1, 32),
            "group_id": group_id,
        }
    if set(groups) != {"common", "rare"}:
        raise ValueError("BatchTriage groups must be common and rare")
    if groups["rare"]["deadline"] > groups["common"]["deadline"]:
        raise ValueError("rare deadline must not exceed common deadline")
    if max(group["deadline"] for group in groups.values()) + harm_delay > horizon:
        raise ValueError("horizon must cover every deadline plus harm_delay")
    if capacity == 2 and horizon < 2:
        raise ValueError("BatchTriage horizon is too short")

    raw_policies = evaluator["policies"]
    if not isinstance(raw_policies, list) or len(raw_policies) != 2:
        raise ValueError("BatchTriage must contain incumbent and candidate policies")
    policies = []
    policy_ids: set[str] = set()
    for policy in raw_policies:
        if not isinstance(policy, dict) or set(policy) != _POLICY_FIELDS:
            raise ValueError("BatchTriage policy has unexpected fields")
        policy_id = policy["policy_id"]
        if policy_id not in {item.value for item in Policy} or policy_id in policy_ids:
            raise ValueError(
                "BatchTriage policies must be unique incumbent and candidate policies"
            )
        priority = policy["priority"]
        if priority not in {"common_first", "earliest_deadline", "random"}:
            raise ValueError("BatchTriage policy priority is invalid")
        service_cost = _parse_fraction(
            policy["service_cost"],
            f"{policy_id} service_cost",
            maximum=Fraction(1000),
        )
        if service_cost == 0:
            raise ValueError("BatchTriage service_cost must be positive")
        policies.append(
            {
                "policy_id": policy_id,
                "priority": priority,
                "service_cost": service_cost,
            }
        )
        policy_ids.add(policy_id)
    if policy_ids != {item.value for item in Policy}:
        raise ValueError("BatchTriage policies must be incumbent and candidate")

    raw_worlds = evaluator["worlds"]
    if not isinstance(raw_worlds, dict) or set(raw_worlds) != {
        item.value for item in World
    }:
        raise ValueError("BatchTriage worlds must be safe and harmful")
    worlds = {
        world.value: _validate_world(raw_worlds[world.value], world.value, groups)
        for world in World
    }
    safe_marginals = _marginals(worlds[World.SAFE.value])
    harmful_marginals = _marginals(worlds[World.HARMFUL.value])
    declared = {
        group_id: group["arrival_probability"] for group_id, group in groups.items()
    }
    if safe_marginals != declared or harmful_marginals != declared:
        raise ValueError(
            "BatchTriage world marginals must match each group arrival probability"
        )
    if safe_marginals != harmful_marginals:
        raise ValueError("BatchTriage worlds must have identical group marginals")
    if sum(declared.values()) == 0:
        raise ValueError("BatchTriage must contain positive arrival mass")
    return groups, tuple(policies), worlds


def _validate_world(
    raw_scenarios: object,
    world: str,
    groups: dict[str, dict[str, object]],
) -> tuple[tuple[bool, bool, Fraction], ...]:
    if not isinstance(raw_scenarios, list) or len(raw_scenarios) != 4:
        raise ValueError(f"BatchTriage {world} world must contain four scenarios")
    scenarios = []
    support: set[tuple[bool, bool]] = set()
    for scenario in raw_scenarios:
        if not isinstance(scenario, dict) or set(scenario) != _SCENARIO_FIELDS:
            raise ValueError("BatchTriage scenario has unexpected fields")
        common = scenario["common"]
        rare = scenario["rare"]
        if type(common) is not bool or type(rare) is not bool:
            raise ValueError("BatchTriage scenario arrivals must be booleans")
        outcome = (common, rare)
        if outcome in support:
            raise ValueError(f"BatchTriage {world} world has duplicate support")
        scenarios.append(
            (
                common,
                rare,
                _parse_fraction(scenario["probability"], "scenario probability"),
            )
        )
        support.add(outcome)
    if support != {(False, False), (False, True), (True, False), (True, True)}:
        raise ValueError(f"BatchTriage {world} world has incomplete support")
    if sum(scenario[2] for scenario in scenarios) != 1:
        raise ValueError(f"BatchTriage {world} probabilities must sum to one")
    if any(group["arrival_probability"] == 0 for group in groups.values()):
        raise ValueError("BatchTriage group arrival probabilities must be positive")
    return tuple(scenarios)


def _marginals(
    scenarios: tuple[tuple[bool, bool, Fraction], ...],
) -> dict[str, Fraction]:
    return {
        "common": sum(probability for common, _, probability in scenarios if common),
        "rare": sum(probability for _, rare, probability in scenarios if rare),
    }


def _select_policy(
    policies: tuple[dict[str, object], ...], policy: Policy | str
) -> dict[str, object]:
    try:
        policy_id = Policy(policy).value
    except (TypeError, ValueError) as error:
        raise ValueError("BatchTriage policy must be incumbent or candidate") from error
    return next(item for item in policies if item["policy_id"] == policy_id)


def _audit(
    groups: dict[str, dict[str, object]], policy: dict[str, object]
) -> dict[str, object]:
    total_arrivals = sum(group["arrival_probability"] for group in groups.values())
    completed = sum(
        group["arrival_probability"]
        for group in groups.values()
        if 1 <= group["deadline"]
    )
    return {
        "completion_rate": _fraction_text(completed / total_arrivals),
        "deadlines": {
            group_id: group["deadline"] for group_id, group in sorted(groups.items())
        },
        "marginals": {
            group_id: _fraction_text(group["arrival_probability"])
            for group_id, group in sorted(groups.items())
        },
        "service_cost": _fraction_text(policy["service_cost"]),
    }


def _service_orders(
    arrivals: tuple[str, ...],
    capacity: int,
    groups: dict[str, dict[str, object]],
    priority: str,
) -> tuple[tuple[Fraction, dict[str, int]], ...]:
    if not arrivals:
        return ((Fraction(1), {}),)
    if capacity == 2 or len(arrivals) == 1:
        return ((Fraction(1), {group_id: 0 for group_id in arrivals}),)
    if priority == "random":
        first, second = arrivals
        return (
            (Fraction(1, 2), {first: 0, second: 1}),
            (Fraction(1, 2), {second: 0, first: 1}),
        )
    if priority == "common_first":
        ordered = sorted(arrivals, key=lambda group_id: group_id != "common")
    else:
        if groups[arrivals[0]]["deadline"] == groups[arrivals[1]]["deadline"]:
            first, second = arrivals
            return (
                (Fraction(1, 2), {first: 0, second: 1}),
                (Fraction(1, 2), {second: 0, first: 1}),
            )
        ordered = sorted(
            arrivals,
            key=lambda group_id: (groups[group_id]["deadline"], group_id),
        )
    return ((Fraction(1), {group_id: index for index, group_id in enumerate(ordered)}),)


def _evaluate_world(
    evaluator: dict[str, object],
    groups: dict[str, dict[str, object]],
    policy: dict[str, object],
    scenarios: tuple[tuple[bool, bool, Fraction], ...],
) -> dict[str, object]:
    capacity = evaluator["capacity"]
    group_arrivals: defaultdict[str, Fraction] = defaultdict(Fraction)
    group_harm: defaultdict[str, Fraction] = defaultdict(Fraction)
    expected_arrivals = Fraction(0)
    completed = Fraction(0)
    backlog = Fraction(0)
    first_harm_time: int | None = None

    for common, rare, scenario_probability in scenarios:
        arrivals = tuple(
            group_id
            for group_id, present in (("common", common), ("rare", rare))
            if present
        )
        expected_arrivals += scenario_probability * len(arrivals)
        if len(arrivals) > capacity:
            backlog += scenario_probability
        for group_id in arrivals:
            group_arrivals[group_id] += scenario_probability
        for order_probability, starts in _service_orders(
            arrivals, capacity, groups, policy["priority"]
        ):
            branch_probability = scenario_probability * order_probability
            if branch_probability == 0:
                continue
            completed += branch_probability * sum(
                start + 1 <= evaluator["horizon"] for start in starts.values()
            )
            for group_id, start in starts.items():
                if start < groups[group_id]["deadline"]:
                    continue
                group_harm[group_id] += branch_probability
                harm_time = groups[group_id]["deadline"] + evaluator["harm_delay"]
                if first_harm_time is None or harm_time < first_harm_time:
                    first_harm_time = harm_time

    unconditional_harm = sum(group_harm.values())
    conditional_harm = {
        group_id: _fraction_text(group_harm[group_id] / group_arrivals[group_id])
        for group_id in sorted(groups)
    }
    return {
        "backlog_probability": _fraction_text(backlog),
        "completed_per_batch": _fraction_text(completed),
        "first_harm_time": first_harm_time,
        "group_harm": conditional_harm,
        "overall_harm": _fraction_text(unconditional_harm / expected_arrivals),
        "unconditional_harm": _fraction_text(unconditional_harm),
    }
