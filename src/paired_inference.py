from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

from .environment_lifecycles import FAMILIES

CANDIDATE = "online_closed_e"
COMPARATORS = ("addis_spending", "shrinking_budget")
METHODS = frozenset((CANDIDATE, *COMPARATORS))
BENCHMARK_FAMILIES = frozenset(FAMILIES)
SCENARIOS = frozenset(("null_only", "all_good", "mixed"))
ENVIRONMENTS_PER_CELL = 100
DEFAULT_ALPHA = 0.05
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260821


@dataclass(frozen=True)
class LifecycleOutcome:
    family: str
    scenario: str
    environment_seed: int
    lifecycle_seed: int
    method_seed: int
    method: str
    harmful_lifecycle: bool
    genuine_acceptance: float
    final_utility: float

    def __post_init__(self) -> None:
        for name in ("family", "scenario", "method"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if self.scenario not in SCENARIOS:
            raise ValueError(f"unsupported scenario: {self.scenario}")
        if self.method not in METHODS:
            raise ValueError(f"unsupported method: {self.method}")
        for name in ("environment_seed", "lifecycle_seed", "method_seed"):
            if isinstance(getattr(self, name), bool) or not isinstance(
                getattr(self, name), int
            ):
                raise ValueError(f"{name} must be an integer")
        if not isinstance(self.harmful_lifecycle, bool):
            raise ValueError("harmful_lifecycle must be boolean")
        if isinstance(self.genuine_acceptance, bool) or not isinstance(
            self.genuine_acceptance, (int, float)
        ):
            raise ValueError("genuine_acceptance must be numeric")
        if not math.isfinite(self.genuine_acceptance):
            raise ValueError("genuine_acceptance must be finite")
        if not 0 <= self.genuine_acceptance <= 1:
            raise ValueError("genuine_acceptance must be within [0, 1]")
        if isinstance(self.final_utility, bool) or not isinstance(
            self.final_utility, (int, float)
        ):
            raise ValueError("final_utility must be numeric")
        if not math.isfinite(self.final_utility):
            raise ValueError("final_utility must be finite")
        if not -1 <= self.final_utility <= 1:
            raise ValueError("final_utility must be normalized to [-1, 1]")

    @property
    def pair_key(self) -> tuple[str, str, int, int, int]:
        return (
            self.family,
            self.scenario,
            self.environment_seed,
            self.lifecycle_seed,
            self.method_seed,
        )


@dataclass(frozen=True)
class PairedHypothesis:
    hypothesis_id: str
    comparator: str
    metric: str
    direction: str
    scenarios: frozenset[str]


HYPOTHESES = (
    PairedHypothesis(
        "H1",
        "addis_spending",
        "harmful_lifecycle",
        "lower",
        frozenset(("null_only", "mixed")),
    ),
    PairedHypothesis(
        "H2",
        "addis_spending",
        "genuine_acceptance",
        "higher",
        frozenset(("all_good", "mixed")),
    ),
    PairedHypothesis(
        "H3",
        "addis_spending",
        "final_utility",
        "higher",
        SCENARIOS,
    ),
    PairedHypothesis(
        "H4",
        "shrinking_budget",
        "harmful_lifecycle",
        "lower",
        frozenset(("null_only", "mixed")),
    ),
    PairedHypothesis(
        "H5",
        "shrinking_budget",
        "genuine_acceptance",
        "higher",
        frozenset(("all_good", "mixed")),
    ),
    PairedHypothesis(
        "H6",
        "shrinking_budget",
        "final_utility",
        "higher",
        SCENARIOS,
    ),
)


def run_paired_inference(
    outcomes: Iterable[LifecycleOutcome],
    *,
    alpha: float = DEFAULT_ALPHA,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    if alpha != DEFAULT_ALPHA:
        raise ValueError(f"alpha must equal the prespecified value {DEFAULT_ALPHA}")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int):
        raise ValueError("bootstrap_samples must be an integer")
    if bootstrap_samples != DEFAULT_BOOTSTRAP_SAMPLES:
        raise ValueError(
            "bootstrap_samples must equal the prespecified value "
            f"{DEFAULT_BOOTSTRAP_SAMPLES}"
        )
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("bootstrap_seed must be an integer")
    if bootstrap_seed != DEFAULT_BOOTSTRAP_SEED:
        raise ValueError(
            f"bootstrap_seed must equal the prespecified value {DEFAULT_BOOTSTRAP_SEED}"
        )

    indexed = _index_outcomes(outcomes)
    rows = [
        _hypothesis_row(
            hypothesis,
            indexed,
            alpha,
            bootstrap_samples,
            bootstrap_seed,
        )
        for hypothesis in HYPOTHESES
    ]
    adjusted = holm_adjust(
        [(str(row["hypothesis_id"]), float(row["raw_p"])) for row in rows]
    )
    for row in rows:
        row["adjusted_p"] = adjusted[str(row["hypothesis_id"])]
        failure = _hypothesis_failure(row, alpha)
        row["criterion_met"] = failure is None
        row["failure_reason"] = failure

    family_effects = _family_effect_rows(indexed)
    comparator_results = [
        _comparator_result(comparator, rows, family_effects)
        for comparator in COMPARATORS
    ]
    passing = [
        str(result["comparator"])
        for result in comparator_results
        if result["all_prespecified_criteria_met"]
    ]

    return {
        "analysis": "secondary_synthetic_protocol_characterization",
        "candidate": CANDIDATE,
        "comparisons": rows,
        "comparator_results": comparator_results,
        "criteria_met_for": passing,
        "family_effects": family_effects,
        "settings": {
            "alpha": alpha,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "interpretation": "secondary_characterization_not_method_ranking",
            "interval": "paired_stratified_percentile_bootstrap",
            "known_semantic_limit": (
                "BatchTriage nominal seeds repeat 72 evaluator-distinct semantics"
            ),
            "multiplicity": "holm_over_six_prespecified_hypotheses",
        },
    }


def holm_adjust(p_values: Iterable[tuple[str, float]]) -> dict[str, float]:
    values = list(p_values)
    identifiers = [identifier for identifier, _ in values]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("hypothesis identifiers must be unique")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for _, value in values):
        raise ValueError("p-values must be finite and within [0, 1]")

    ordered = sorted(values, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (identifier, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[identifier] = running
    return {identifier: adjusted[identifier] for identifier in identifiers}


def exact_one_sided_sign_p(wins: int, losses: int) -> float:
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (wins, losses)
    ):
        raise ValueError("wins and losses must be integers")
    if wins < 0 or losses < 0:
        raise ValueError("wins and losses must be nonnegative")
    informative = wins + losses
    if informative == 0:
        return 1.0
    numerator = sum(
        math.comb(informative, count) for count in range(wins, informative + 1)
    )
    return numerator / (1 << informative)


def exact_one_sided_mcnemar_p(
    candidate_only_harm: int, comparator_only_harm: int
) -> float:
    return exact_one_sided_sign_p(comparator_only_harm, candidate_only_harm)


def _index_outcomes(
    outcomes: Iterable[LifecycleOutcome],
) -> dict[tuple[str, str, int, int, int], dict[str, LifecycleOutcome]]:
    indexed: dict[tuple[str, str, int, int, int], dict[str, LifecycleOutcome]] = {}
    for outcome in outcomes:
        if not isinstance(outcome, LifecycleOutcome):
            raise TypeError("outcomes must contain LifecycleOutcome records")
        methods = indexed.setdefault(outcome.pair_key, {})
        if outcome.method in methods:
            raise ValueError(
                f"duplicate outcome for {outcome.pair_key}/{outcome.method}"
            )
        methods[outcome.method] = outcome
    if not indexed:
        raise ValueError("at least one paired lifecycle is required")
    for key, methods in indexed.items():
        missing = METHODS - methods.keys()
        if missing:
            raise ValueError(f"missing paired methods for {key}: {sorted(missing)}")
    families = {key[0] for key in indexed}
    if families != BENCHMARK_FAMILIES:
        raise ValueError(
            "benchmark families must match the prespecified set: "
            f"{sorted(BENCHMARK_FAMILIES)}"
        )
    independent_units: set[tuple[str, str, int]] = set()
    for family, scenario, environment_seed, _, _ in indexed:
        unit = (family, scenario, environment_seed)
        if unit in independent_units:
            raise ValueError(f"repeated independent lifecycle unit: {unit}")
        independent_units.add(unit)
    seed_owners: dict[int, tuple[str, str]] = {}
    for family, scenario, environment_seed in sorted(independent_units):
        owner = (family, scenario)
        previous = seed_owners.setdefault(environment_seed, owner)
        if previous != owner:
            raise ValueError(
                f"environment seed reused across benchmark cells: {environment_seed}"
            )
    for family in sorted(BENCHMARK_FAMILIES):
        for scenario in sorted(SCENARIOS):
            count = sum(
                unit[0] == family and unit[1] == scenario for unit in independent_units
            )
            if count != ENVIRONMENTS_PER_CELL:
                raise ValueError(
                    "benchmark cell must contain exactly "
                    f"{ENVIRONMENTS_PER_CELL} environments: "
                    f"{family}/{scenario} has {count}"
                )
    return indexed


def _hypothesis_row(
    hypothesis: PairedHypothesis,
    indexed: dict[tuple[str, str, int, int, int], dict[str, LifecycleOutcome]],
    alpha: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    pairs = _metric_pairs(hypothesis, indexed)
    differences = [candidate - comparator for _, candidate, comparator in pairs]
    wins, losses, ties = _win_counts(differences, hypothesis.direction)
    interval_lower, interval_upper = _paired_bootstrap_interval(
        [
            (key, difference)
            for (key, _, _), difference in zip(pairs, differences, strict=True)
        ],
        alpha,
        bootstrap_samples,
        bootstrap_seed,
    )
    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "candidate": CANDIDATE,
        "comparator": hypothesis.comparator,
        "metric": hypothesis.metric,
        "direction": hypothesis.direction,
        "scenarios": sorted(hypothesis.scenarios),
        "pairs": len(pairs),
        "informative_pairs": wins + losses,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "effect": sum(differences) / len(differences),
        "ci_lower": interval_lower,
        "ci_upper": interval_upper,
        "interval": "paired_stratified_percentile_bootstrap",
        "test": (
            "exact_one_sided_mcnemar"
            if hypothesis.metric == "harmful_lifecycle"
            else "exact_one_sided_sign"
        ),
        "raw_p": (
            exact_one_sided_mcnemar_p(losses, wins)
            if hypothesis.metric == "harmful_lifecycle"
            else exact_one_sided_sign_p(wins, losses)
        ),
        "adjusted_p": None,
        "criterion_met": False,
        "failure_reason": None,
    }


def _metric_pairs(
    hypothesis: PairedHypothesis,
    indexed: dict[tuple[str, str, int, int, int], dict[str, LifecycleOutcome]],
) -> list[tuple[tuple[str, str, int, int, int], float, float]]:
    pairs = []
    for key in sorted(indexed):
        if key[1] not in hypothesis.scenarios:
            continue
        methods = indexed[key]
        candidate = methods[CANDIDATE]
        comparator = methods[hypothesis.comparator]
        pairs.append(
            (
                key,
                float(getattr(candidate, hypothesis.metric)),
                float(getattr(comparator, hypothesis.metric)),
            )
        )
    if not pairs:
        raise ValueError(f"no paired outcomes for {hypothesis.hypothesis_id}")
    return pairs


def _win_counts(differences: list[float], direction: str) -> tuple[int, int, int]:
    if direction == "lower":
        wins = sum(difference < 0 for difference in differences)
        losses = sum(difference > 0 for difference in differences)
    else:
        wins = sum(difference > 0 for difference in differences)
        losses = sum(difference < 0 for difference in differences)
    ties = len(differences) - wins - losses
    return wins, losses, ties


def _paired_bootstrap_interval(
    paired_differences: list[tuple[tuple[str, str, int, int, int], float]],
    alpha: float,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    strata: dict[tuple[str, str], list[float]] = {}
    for key, difference in paired_differences:
        strata.setdefault((key[0], key[1]), []).append(difference)
    ordered_strata = [strata[key] for key in sorted(strata)]
    if all(len(set(values)) == 1 for values in ordered_strata):
        mean = sum(sum(values) for values in ordered_strata) / sum(
            len(values) for values in ordered_strata
        )
        return mean, mean
    random_source = random.Random(seed)
    total = sum(len(values) for values in ordered_strata)
    estimates = sorted(
        sum(
            random_source.choice(values)
            for values in ordered_strata
            for _ in range(len(values))
        )
        / total
        for _ in range(samples)
    )
    return (
        _percentile(estimates, alpha / 2),
        _percentile(estimates, 1 - alpha / 2),
    )


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _family_effect_rows(
    indexed: dict[tuple[str, str, int, int, int], dict[str, LifecycleOutcome]],
) -> list[dict[str, object]]:
    rows = []
    families = sorted({key[0] for key in indexed})
    for hypothesis in HYPOTHESES:
        pairs = _metric_pairs(hypothesis, indexed)
        for family in families:
            family_pairs = [pair for pair in pairs if pair[0][0] == family]
            if not family_pairs:
                raise ValueError(
                    f"missing family cohort for {hypothesis.hypothesis_id}/{family}"
                )
            differences = [
                candidate - comparator for _, candidate, comparator in family_pairs
            ]
            rows.append(
                {
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "family": family,
                    "candidate": CANDIDATE,
                    "comparator": hypothesis.comparator,
                    "metric": hypothesis.metric,
                    "direction": hypothesis.direction,
                    "pairs": len(family_pairs),
                    "effect": sum(differences) / len(differences),
                    "uncertainty": "descriptive",
                }
            )
    return rows


def _hypothesis_failure(row: dict[str, object], alpha: float) -> str | None:
    if float(row["adjusted_p"]) > alpha:
        return "holm_adjusted_p_above_alpha"
    if row["direction"] == "lower":
        if float(row["effect"]) >= 0:
            return "effect_not_lower"
        if float(row["ci_upper"]) >= 0:
            return "bootstrap_interval_not_below_zero"
    else:
        if float(row["effect"]) <= 0:
            return "effect_not_higher"
        if float(row["ci_lower"]) <= 0:
            return "bootstrap_interval_not_above_zero"
    return None


def _comparator_result(
    comparator: str,
    rows: list[dict[str, object]],
    family_effects: list[dict[str, object]],
) -> dict[str, object]:
    comparison_rows = [row for row in rows if row["comparator"] == comparator]
    reversal = any(
        row["comparator"] == comparator
        and row["metric"] == "harmful_lifecycle"
        and float(row["effect"]) > 0
        for row in family_effects
    )
    failures = [
        f"{row['hypothesis_id']}:{row['failure_reason']}"
        for row in comparison_rows
        if not row["criterion_met"]
    ]
    if reversal:
        failures.append("family_harm_direction_reversal")
    return {
        "comparator": comparator,
        "all_prespecified_criteria_met": not failures,
        "failure_reasons": failures,
    }
