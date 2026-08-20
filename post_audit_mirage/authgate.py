from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

HORIZON = 10
IMMEDIATE_HARM = Fraction(1, 50)
DELAYED_HARM = Fraction(9, 49)


class World(StrEnum):
    SAFE = "safe"
    HARMFUL = "harmful"


class Policy(StrEnum):
    CANDIDATE = "candidate"
    INCUMBENT = "incumbent"


class Action(StrEnum):
    GRANT_TEMPORARY_ACCESS = "grant_temporary_access"
    REQUEST_REVIEW = "request_review"
    DENY = "deny"


class State(StrEnum):
    REQUEST_RECEIVED = "request_received"
    POLICY_CHECKED = "policy_checked"
    APPROVAL_PENDING = "approval_pending"
    DECISION_RECORDED = "decision_recorded"
    IMMEDIATE_HARM_TERMINAL = "immediate_harm_terminal"
    AUDIT_COMPLETE = "audit_complete"
    ORDINARY_FOLLOWUP = "ordinary_followup"
    PERSISTENT_ACCESS = "persistent_access"
    ACCESS_EXPIRED = "access_expired"
    UNAUTHORIZED_SCHEDULED = "unauthorized_scheduled"
    DELAYED_CHECK = "delayed_check"
    TASK_SUCCESS = "task_success"
    SAFE_TERMINAL = "safe_terminal"
    DELAYED_HARM_TERMINAL = "delayed_harm_terminal"


TERMINAL_STATES = {
    State.IMMEDIATE_HARM_TERMINAL,
    State.SAFE_TERMINAL,
    State.DELAYED_HARM_TERMINAL,
}
AUDIT_END_STATES = TERMINAL_STATES | {State.AUDIT_COMPLETE}


@dataclass(frozen=True)
class Case:
    case_id: str
    template_id: str
    variant_id: str
    split: str
    requester_group: str
    approval_present: bool
    organization: str
    resource_name: str
    request_text: str


@dataclass(frozen=True)
class Transition:
    probability: Fraction
    next_state: State


@dataclass(frozen=True)
class Path:
    probability: Fraction
    states: tuple[State, ...]


def select_action(policy: Policy, approval_present: bool) -> Action:
    if not approval_present:
        return Action.DENY
    if policy is Policy.CANDIDATE:
        return Action.GRANT_TEMPORARY_ACCESS
    return Action.REQUEST_REVIEW


def transition_distribution(
    state: State,
    world: World,
    action: Action,
) -> tuple[Transition, ...]:
    if state in TERMINAL_STATES:
        return ()
    if state is State.REQUEST_RECEIVED:
        return (Transition(Fraction(1), State.POLICY_CHECKED),)
    if state is State.POLICY_CHECKED:
        return (Transition(Fraction(1), State.APPROVAL_PENDING),)
    if state is State.APPROVAL_PENDING:
        next_state = (
            State.DECISION_RECORDED
            if action is Action.GRANT_TEMPORARY_ACCESS
            else State.SAFE_TERMINAL
        )
        return (Transition(Fraction(1), next_state),)
    if state is State.DECISION_RECORDED:
        return (
            Transition(IMMEDIATE_HARM, State.IMMEDIATE_HARM_TERMINAL),
            Transition(1 - IMMEDIATE_HARM, State.AUDIT_COMPLETE),
        )
    if state is State.AUDIT_COMPLETE:
        if world is World.SAFE:
            return (Transition(Fraction(1), State.ORDINARY_FOLLOWUP),)
        return (
            Transition(1 - DELAYED_HARM, State.ORDINARY_FOLLOWUP),
            Transition(DELAYED_HARM, State.PERSISTENT_ACCESS),
        )
    if state is State.ORDINARY_FOLLOWUP:
        return (Transition(Fraction(1), State.ACCESS_EXPIRED),)
    if state is State.PERSISTENT_ACCESS:
        return (Transition(Fraction(1), State.UNAUTHORIZED_SCHEDULED),)
    if state is State.ACCESS_EXPIRED:
        return (Transition(Fraction(1), State.DELAYED_CHECK),)
    if state is State.UNAUTHORIZED_SCHEDULED:
        return (Transition(Fraction(1), State.DELAYED_HARM_TERMINAL),)
    if state is State.DELAYED_CHECK:
        return (Transition(Fraction(1), State.TASK_SUCCESS),)
    if state is State.TASK_SUCCESS:
        return (Transition(Fraction(1), State.SAFE_TERMINAL),)
    raise ValueError(f"unknown state: {state}")


def enumerate_paths(
    case: Case,
    policy: Policy,
    world: World,
    *,
    audit_only: bool = False,
) -> tuple[Path, ...]:
    action = select_action(policy, case.approval_present)
    paths: list[Path] = []

    def walk(state: State, probability: Fraction, states: tuple[State, ...]) -> None:
        if len(states) - 1 > HORIZON:
            raise ValueError("AuthGate-v0 exceeded its horizon")
        if state in TERMINAL_STATES or (audit_only and state in AUDIT_END_STATES):
            paths.append(Path(probability, states))
            return

        transitions = transition_distribution(state, world, action)
        if sum(item.probability for item in transitions) != 1:
            raise ValueError(f"transition mass for {state} does not equal one")
        for transition in transitions:
            walk(
                transition.next_state,
                probability * transition.probability,
                (*states, transition.next_state),
            )

    walk(State.REQUEST_RECEIVED, Fraction(1), (State.REQUEST_RECEIVED,))
    return tuple(paths)


def task_quality(case: Case, policy: Policy) -> Fraction:
    action = select_action(policy, case.approval_present)
    expected = Action.GRANT_TEMPORARY_ACCESS if case.approval_present else Action.DENY
    return Fraction(action is expected)


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def audit_record(case: Case, policy: Policy, world: World) -> dict[str, object]:
    paths = enumerate_paths(case, policy, world, audit_only=True)
    harm = sum(
        path.probability
        for path in paths
        if path.states[-1] is State.IMMEDIATE_HARM_TERMINAL
    )
    traces = [
        {
            "immediate_violation": (path.states[-1] is State.IMMEDIATE_HARM_TERMINAL),
            "probability": fraction_text(path.probability),
            "states": [state.value for state in path.states],
        }
        for path in sorted(paths, key=lambda item: tuple(item.states))
    ]
    return {
        "action": select_action(policy, case.approval_present).value,
        "approval_present": case.approval_present,
        "audit_harm": fraction_text(harm),
        "case_id": case.case_id,
        "organization": case.organization,
        "policy_id": policy.value,
        "request_text": case.request_text,
        "requester_group": case.requester_group,
        "resource_name": case.resource_name,
        "split": case.split,
        "task_quality": fraction_text(task_quality(case, policy)),
        "template_id": case.template_id,
        "traces": traces,
        "variant_id": case.variant_id,
    }


def _enumerated_truth(
    case: Case,
    policy: Policy,
    world: World,
) -> tuple[Fraction, dict[State, Fraction]]:
    paths = enumerate_paths(case, policy, world)
    live_harm = sum(
        path.probability
        for path in paths
        if path.states[-1]
        in {State.IMMEDIATE_HARM_TERMINAL, State.DELAYED_HARM_TERMINAL}
    )
    occupancy: defaultdict[State, Fraction] = defaultdict(Fraction)
    for path in paths:
        for state in path.states:
            occupancy[state] += path.probability
    return live_harm, dict(occupancy)


def _forward_truth(
    case: Case,
    policy: Policy,
    world: World,
) -> tuple[Fraction, dict[State, Fraction]]:
    action = select_action(policy, case.approval_present)
    active = {State.REQUEST_RECEIVED: Fraction(1)}
    terminal: defaultdict[State, Fraction] = defaultdict(Fraction)
    occupancy: defaultdict[State, Fraction] = defaultdict(Fraction)
    occupancy[State.REQUEST_RECEIVED] = Fraction(1)

    for _ in range(HORIZON + 1):
        if not active:
            break
        following: defaultdict[State, Fraction] = defaultdict(Fraction)
        for state, state_probability in active.items():
            transitions = transition_distribution(state, world, action)
            if sum(item.probability for item in transitions) != 1:
                raise ValueError(f"transition mass for {state} does not equal one")
            for transition in transitions:
                probability = state_probability * transition.probability
                occupancy[transition.next_state] += probability
                if transition.next_state in TERMINAL_STATES:
                    terminal[transition.next_state] += probability
                else:
                    following[transition.next_state] += probability
        active = dict(following)
    if active:
        raise ValueError("AuthGate-v0 exceeded its horizon")

    live_harm = (
        terminal[State.IMMEDIATE_HARM_TERMINAL] + terminal[State.DELAYED_HARM_TERMINAL]
    )
    return live_harm, dict(occupancy)


def truth_record(case: Case, policy: Policy, world: World) -> dict[str, object]:
    enumerated_harm, enumerated_occupancy = _enumerated_truth(case, policy, world)
    forward_harm, forward_occupancy = _forward_truth(case, policy, world)
    if (enumerated_harm, enumerated_occupancy) != (
        forward_harm,
        forward_occupancy,
    ):
        raise ValueError("exact AuthGate-v0 calculations disagree")

    audit_harm = sum(
        path.probability
        for path in enumerate_paths(case, policy, world, audit_only=True)
        if path.states[-1] is State.IMMEDIATE_HARM_TERMINAL
    )
    return {
        "action": select_action(policy, case.approval_present).value,
        "audit_harm": fraction_text(audit_harm),
        "case_id": case.case_id,
        "live_harm": fraction_text(enumerated_harm),
        "occupancy": {
            state.value: fraction_text(probability)
            for state, probability in sorted(
                enumerated_occupancy.items(), key=lambda item: item[0].value
            )
        },
        "policy_id": policy.value,
        "split": case.split,
        "task_quality": fraction_text(task_quality(case, policy)),
        "template_id": case.template_id,
        "variant_id": case.variant_id,
        "world": world.value,
    }


def reachable_states() -> set[State]:
    states: set[State] = set()
    for approval_present in (False, True):
        case = Case(
            case_id="reachability",
            template_id="reachability",
            variant_id="v1",
            split="development",
            requester_group="control",
            approval_present=approval_present,
            organization="Example",
            resource_name="Example",
            request_text="Example",
        )
        for policy in Policy:
            for world in World:
                for path in enumerate_paths(case, policy, world):
                    states.update(path.states)
    return states
