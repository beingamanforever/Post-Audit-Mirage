from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

Status = Literal["deploy", "hold", "cannot_determine", "unsupported"]

VALID_COLLECTION_DESIGNS = frozenset(
    ("representative", "outcome_independent_missingness")
)
UNSUPPORTED_COLLECTION_DESIGNS = frozenset(
    ("outcome_dependent_selection", "distribution_shift")
)


@dataclass(frozen=True, slots=True)
class GroupBounds:
    group: str
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class IdentifiedRangeResult:
    status: Status
    bounds: tuple[GroupBounds, ...]
    reason: str


def monitor_identified_range(
    compatible_answers: frozenset[bool],
    required_groups: Iterable[str],
    harm_counts: Mapping[str, int],
    observation_counts: Mapping[str, int],
    *,
    harm_threshold: float,
    lifecycle_alpha: float,
    collection_design: str,
) -> IdentifiedRangeResult:
    answers = _validate_answers(compatible_answers)
    groups = _validate_groups(required_groups)
    harms, observations = _validate_counts(groups, harm_counts, observation_counts)
    threshold = _validate_probability(harm_threshold, "harm_threshold", closed=True)
    alpha = _validate_probability(lifecycle_alpha, "lifecycle_alpha", closed=False)
    _validate_collection_design(collection_design)

    if len(answers) == 1:
        answer = next(iter(answers))
        status: Status = "deploy" if answer else "hold"
        reason = (
            "offline answers identify deployment"
            if answer
            else "offline answers identify harm"
        )
        return IdentifiedRangeResult(status, (), reason)

    if collection_design in UNSUPPORTED_COLLECTION_DESIGNS:
        return IdentifiedRangeResult(
            "unsupported",
            (),
            f"{collection_design} does not identify live harm",
        )

    bounds = tuple(
        _group_bounds(group, harms[group], observations[group], len(groups), alpha)
        for group in groups
    )
    for bound in bounds:
        if bound.lower > threshold:
            return IdentifiedRangeResult(
                "hold",
                bounds,
                f"{bound.group} lower bound exceeds the harm threshold",
            )
    if all(bound.upper <= threshold for bound in bounds):
        return IdentifiedRangeResult(
            "deploy",
            bounds,
            "all group upper bounds are within the harm threshold",
        )
    return IdentifiedRangeResult(
        "cannot_determine",
        bounds,
        "monitoring bounds do not identify deploy or hold",
    )


def _validate_answers(answers: object) -> frozenset[bool]:
    if (
        not isinstance(answers, frozenset)
        or not answers
        or any(type(answer) is not bool for answer in answers)
    ):
        raise ValueError("compatible_answers must be a nonempty frozenset of booleans")
    return answers


def _validate_groups(required_groups: Iterable[str]) -> tuple[str, ...]:
    if isinstance(required_groups, str):
        raise ValueError("required_groups must be a nonempty sequence of unique names")
    try:
        groups = tuple(required_groups)
    except TypeError as error:
        raise ValueError(
            "required_groups must be a nonempty sequence of unique names"
        ) from error
    if (
        not groups
        or any(not isinstance(group, str) or not group for group in groups)
        or len(groups) != len(set(groups))
    ):
        raise ValueError("required_groups must be a nonempty sequence of unique names")
    return groups


def _validate_counts(
    groups: tuple[str, ...],
    harm_counts: Mapping[str, int],
    observation_counts: Mapping[str, int],
) -> tuple[Mapping[str, int], Mapping[str, int]]:
    if not isinstance(harm_counts, Mapping) or set(harm_counts) != set(groups):
        raise ValueError("harm_counts must exactly match required_groups")
    if not isinstance(observation_counts, Mapping) or set(observation_counts) != set(
        groups
    ):
        raise ValueError("observation_counts must exactly match required_groups")
    for group in groups:
        harms = harm_counts[group]
        observations = observation_counts[group]
        if type(harms) is not int or type(observations) is not int:
            raise ValueError("group counts must be integers")
        if not 0 <= harms <= observations:
            raise ValueError("group counts must satisfy 0 <= harms <= observations")
    return harm_counts, observation_counts


def _validate_probability(value: object, name: str, *, closed: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    probability = float(value)
    valid = 0 <= probability <= 1 if closed else 0 < probability < 1
    if not math.isfinite(probability) or not valid:
        interval = "[0, 1]" if closed else "(0, 1)"
        raise ValueError(f"{name} must be finite and within {interval}")
    return probability


def _validate_collection_design(collection_design: object) -> None:
    designs = VALID_COLLECTION_DESIGNS | UNSUPPORTED_COLLECTION_DESIGNS
    if not isinstance(collection_design, str) or collection_design not in designs:
        raise ValueError("collection_design is not recognized")


def _group_bounds(
    group: str,
    harms: int,
    observations: int,
    group_count: int,
    alpha: float,
) -> GroupBounds:
    if observations == 0:
        return GroupBounds(group, 0.0, 1.0)
    # Each group spends alpha / G because sum 1 / (n(n + 1)) over n is one.
    allocated_alpha = alpha / (group_count * observations * (observations + 1))
    radius = math.sqrt(math.log(2 / allocated_alpha) / (2 * observations))
    estimate = harms / observations
    return GroupBounds(
        group,
        max(0.0, estimate - radius),
        min(1.0, estimate + radius),
    )
