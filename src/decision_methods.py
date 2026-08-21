from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Protocol

ALPHA = 0.05
BET = 0.5
ADDIS_LAMBDA = 0.25
ADDIS_TAU = 0.5
ADDIS_GAMMA_SCALE = 0.4374901658

METHOD_NAMES = (
    "always_hold",
    "greedy",
    "fixed_threshold",
    "shrinking_budget",
    "addis_spending",
    "online_closed_e",
    "pace_reset",
    "reused_holdout",
    "sgm_transferred",
    "monitor",
    "oracle",
)


@dataclass(frozen=True)
class ComponentEvidence:
    name: str
    audit: tuple[float, ...]
    holdout: tuple[float, ...] = ()
    require_all: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.audit:
            raise ValueError("an evidence component needs a name and audit values")
        for value in self.audit + self.holdout:
            if not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError("evidence values must be finite and within [-1, 1]")


@dataclass(frozen=True)
class UpdateEvidence:
    family: str
    update_id: str
    components: tuple[ComponentEvidence, ...]
    pace_outcomes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.family or not self.update_id or not self.components:
            raise ValueError("an update needs a family, identifier, and components")
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError("component names must be unique")
        if any(outcome not in (0, 1) for outcome in self.pace_outcomes):
            raise ValueError("PACE outcomes must be binary discordant wins")


@dataclass(frozen=True)
class MonitorEvidence:
    family: str
    update_id: str
    components: tuple[ComponentEvidence, ...]

    def __post_init__(self) -> None:
        if not self.family or not self.update_id or not self.components:
            raise ValueError(
                "monitor evidence needs a family, identifier, and components"
            )
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError("monitor component names must be unique")


@dataclass(frozen=True)
class Decision:
    deploy: bool
    statistic: float
    threshold: float
    reason: str


class DecisionMethod(Protocol):
    name: str

    def decide(
        self,
        update: UpdateEvidence,
        monitor: MonitorEvidence | None = None,
    ) -> Decision: ...


def _mean(values: tuple[float, ...]) -> float:
    return math.fsum(values) / len(values)


def _component_p(component: ComponentEvidence) -> float:
    if component.require_all and any(value <= 0 for value in component.audit):
        return 1.0
    mean = max(_mean(component.audit), 0.0)
    return math.exp(-len(component.audit) * mean * mean / 2)


def _combined_p(components: tuple[ComponentEvidence, ...]) -> float:
    return max(_component_p(component) for component in components)


def _component_e(component: ComponentEvidence, bet: float = BET) -> float:
    if component.require_all and any(value <= 0 for value in component.audit):
        return 0.0
    return math.prod(1 + bet * value for value in component.audit)


def _combined_e(components: tuple[ComponentEvidence, ...]) -> float:
    # The unsafe null is a union, so the minimum is valid without assuming
    # independence between requirement components.
    return min(_component_e(component) for component in components)


def _fixed_decision(
    components: tuple[ComponentEvidence, ...],
    level: float,
    reason: str,
) -> Decision:
    p_value = _combined_p(components)
    return Decision(p_value <= level, p_value, level, reason)


class AlwaysHold:
    name = "always_hold"

    def decide(
        self,
        update: UpdateEvidence,
        monitor: MonitorEvidence | None = None,
    ) -> Decision:
        return Decision(False, 0.0, 0.0, "baseline never deploys")


class Greedy:
    name = "greedy"

    def decide(
        self,
        update: UpdateEvidence,
        monitor: MonitorEvidence | None = None,
    ) -> Decision:
        deploy = all(
            _mean(component.audit) > 0
            and (
                not component.require_all or all(value > 0 for value in component.audit)
            )
            for component in update.components
        )
        statistic = min(_mean(component.audit) for component in update.components)
        return Decision(deploy, statistic, 0.0, "all observable margins are positive")


class FixedThreshold:
    name = "fixed_threshold"

    def __init__(self, alpha: float = ALPHA) -> None:
        self.alpha = _valid_alpha(alpha)

    def decide(
        self,
        update: UpdateEvidence,
        monitor: MonitorEvidence | None = None,
    ) -> Decision:
        return _fixed_decision(update.components, self.alpha, "fixed offline level")


class ShrinkingBudget:
    name = "shrinking_budget"

    def __init__(self, alpha: float = ALPHA) -> None:
        self.alpha = _valid_alpha(alpha)
        self.round = 0

    def decide(
        self,
        update: UpdateEvidence,
        monitor: MonitorEvidence | None = None,
    ) -> Decision:
        self.round += 1
        level = self.alpha * 6 / (math.pi * math.pi * self.round * self.round)
        return _fixed_decision(update.components, level, "summable alpha spending")


class AddisSpending:
    name = "addis_spending"

    def __init__(
        self,
        alpha: float = ALPHA,
        candidate: float = ADDIS_LAMBDA,
        discard: float = ADDIS_TAU,
    ) -> None:
        self.alpha = _valid_alpha(alpha)
        if not 0 < candidate < discard < 1:
            raise ValueError("ADDIS requires 0 < candidate < discard < 1")
        self.candidate = candidate
        self.discard = discard
        self.selected = 0
        self.candidates = 0

    def decide(
        self,
        update: UpdateEvidence,
        monitor: MonitorEvidence | None = None,
    ) -> Decision:
        p_value = _combined_p(update.components)
        index = self.selected - self.candidates
        gamma = ADDIS_GAMMA_SCALE / ((index + 1) ** 1.6)
        level = self.alpha * (self.discard - self.candidate) * gamma
        decision = Decision(p_value <= level, p_value, level, "ADDIS spending")
        self.selected += p_value <= self.discard
        self.candidates += p_value <= self.candidate
        return decision


class OnlineClosedE:
    name = "online_closed_e"

    def __init__(self, alpha: float = ALPHA) -> None:
        self.alpha = _valid_alpha(alpha)
        self.previous: list[float] = []

    def decide(
        self,
        update: UpdateEvidence,
        monitor: MonitorEvidence | None = None,
    ) -> Decision:
        e_value = _combined_e(update.components)
        # For the singleton claim about the current update, the least favorable
        # intersection includes every earlier e-value below one and excludes
        # every earlier e-value above one.
        wealth = e_value * math.prod(value for value in self.previous if value < 1)
        deploy = wealth >= 1 / self.alpha
        self.previous.append(e_value)
        return Decision(deploy, wealth, 1 / self.alpha, "singleton online closure")


class PaceReset:
    name = "pace_reset"

    def __init__(self, alpha: float = ALPHA, bet: float = BET) -> None:
        self.alpha = _valid_alpha(alpha)
        if not 0 <= bet < 1:
            raise ValueError("PACE bet must be within [0, 1)")
        self.bet = bet

    def decide(
        self,
        update: UpdateEvidence,
        monitor: MonitorEvidence | None = None,
    ) -> Decision:
        threshold = 1 / self.alpha
        wealth = 1.0
        for outcome in update.pace_outcomes:
            wealth *= 1 + self.bet * (2 * outcome - 1)
            if wealth >= threshold:
                return Decision(
                    True,
                    wealth,
                    threshold,
                    "per-update PACE first crossing",
                )
        return Decision(False, wealth, threshold, "per-update PACE wealth reset")


class ReusedHoldout:
    name = "reused_holdout"

    def __init__(
        self,
        threshold: float = 0.1,
        noise: float = 0.01,
        budget: int = 20,
        seed: int = 0,
    ) -> None:
        if threshold <= 0 or noise <= 0 or budget < 1:
            raise ValueError("Thresholdout parameters must be positive")
        self.threshold = threshold
        self.noise = noise
        self.budget = budget
        self.random = random.Random(seed)
        self.noisy_threshold = threshold + self._laplace(2 * noise)

    def _laplace(self, scale: float) -> float:
        magnitude = self.random.expovariate(1 / scale)
        return magnitude if self.random.getrandbits(1) else -magnitude

    def decide(
        self,
        update: UpdateEvidence,
        monitor: MonitorEvidence | None = None,
    ) -> Decision:
        answers: list[float] = []
        for component in update.components:
            if not component.holdout or self.budget < 1:
                return Decision(False, 0.0, 0.0, "holdout unavailable or exhausted")
            if component.require_all and any(
                value <= 0 for value in component.audit + component.holdout
            ):
                return Decision(False, -1.0, 0.0, "exact holdout requirement failed")
            audit_mean = (_mean(component.audit) + 1) / 2
            holdout_mean = (_mean(component.holdout) + 1) / 2
            noisy_gap = self.noisy_threshold + self._laplace(4 * self.noise)
            if abs(holdout_mean - audit_mean) > noisy_gap:
                answer = holdout_mean + self._laplace(self.noise)
                self.budget -= 1
                self.noisy_threshold = self.threshold + self._laplace(2 * self.noise)
            else:
                answer = audit_mean
            answers.append(2 * answer - 1)
        statistic = min(answers)
        return Decision(statistic > 0, statistic, 0.0, "Thresholdout baseline")


class SgmTransferred:
    name = "sgm_transferred"

    def __init__(self, alpha: float = ALPHA) -> None:
        self.alpha = _valid_alpha(alpha)
        self.wealth = 1.0

    def decide(
        self,
        update: UpdateEvidence,
        monitor: MonitorEvidence | None = None,
    ) -> Decision:
        self.wealth *= _combined_e(update.components)
        return Decision(
            self.wealth >= 1 / self.alpha,
            self.wealth,
            1 / self.alpha,
            "known-bad cross-update wealth transfer",
        )


class Monitor:
    name = "monitor"

    def __init__(self, alpha: float = ALPHA) -> None:
        self.alpha = _valid_alpha(alpha)

    def decide(
        self,
        update: UpdateEvidence,
        monitor: MonitorEvidence | None = None,
    ) -> Decision:
        if monitor is None:
            return Decision(False, 1.0, self.alpha, "live-style stream unavailable")
        if (monitor.family, monitor.update_id) != (update.family, update.update_id):
            raise ValueError("monitor evidence does not match the update")
        return _fixed_decision(monitor.components, self.alpha, "live-style monitor")


class Oracle:
    name = "oracle"

    def __init__(self, answers: Mapping[tuple[str, str], bool]) -> None:
        self.answers = dict(answers)

    def decide(
        self,
        update: UpdateEvidence,
        monitor: MonitorEvidence | None = None,
    ) -> Decision:
        key = (update.family, update.update_id)
        if key not in self.answers:
            raise ValueError(
                f"missing oracle answer for {update.family}/{update.update_id}"
            )
        deploy = self.answers[key]
        return Decision(deploy, float(deploy), 1.0, "protected true answer")


def build_methods(
    answers: Mapping[tuple[str, str], bool],
    *,
    alpha: float = ALPHA,
    seed: int = 0,
) -> tuple[DecisionMethod, ...]:
    methods: tuple[DecisionMethod, ...] = (
        AlwaysHold(),
        Greedy(),
        FixedThreshold(alpha),
        ShrinkingBudget(alpha),
        AddisSpending(alpha),
        OnlineClosedE(alpha),
        PaceReset(alpha),
        ReusedHoldout(seed=seed),
        SgmTransferred(alpha),
        Monitor(alpha),
        Oracle(answers),
    )
    if tuple(method.name for method in methods) != METHOD_NAMES:
        raise AssertionError("method registry is incomplete")
    return methods


def _valid_alpha(alpha: float) -> float:
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and within (0, 1)")
    return alpha
