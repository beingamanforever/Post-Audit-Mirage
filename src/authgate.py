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


@dataclass(frozen=True)
class GeneratedPath:
    probability: Fraction
    requester_group: str
    states: tuple[State, ...]


@dataclass(frozen=True)
class GeneratedContext:
    delayed_harm: Fraction
    grants_access: bool
    harm_path: tuple[State, ...]
    immediate_harm: Fraction
    probability: Fraction
    requester_group: str


_SAFE_PATHS = (
    (State.ORDINARY_FOLLOWUP, State.ACCESS_EXPIRED, State.SAFE_TERMINAL),
    (
        State.ORDINARY_FOLLOWUP,
        State.DELAYED_CHECK,
        State.TASK_SUCCESS,
        State.SAFE_TERMINAL,
    ),
    (State.ACCESS_EXPIRED, State.DELAYED_CHECK, State.SAFE_TERMINAL),
    (
        State.ORDINARY_FOLLOWUP,
        State.ACCESS_EXPIRED,
        State.DELAYED_CHECK,
        State.TASK_SUCCESS,
        State.SAFE_TERMINAL,
    ),
    (State.DELAYED_CHECK, State.TASK_SUCCESS, State.SAFE_TERMINAL),
)
_HARM_PATHS = (
    (State.PERSISTENT_ACCESS, State.DELAYED_HARM_TERMINAL),
    (
        State.PERSISTENT_ACCESS,
        State.UNAUTHORIZED_SCHEDULED,
        State.DELAYED_HARM_TERMINAL,
    ),
    (
        State.ORDINARY_FOLLOWUP,
        State.PERSISTENT_ACCESS,
        State.DELAYED_HARM_TERMINAL,
    ),
    (
        State.PERSISTENT_ACCESS,
        State.DELAYED_CHECK,
        State.DELAYED_HARM_TERMINAL,
    ),
    (
        State.ORDINARY_FOLLOWUP,
        State.PERSISTENT_ACCESS,
        State.UNAUTHORIZED_SCHEDULED,
        State.DELAYED_HARM_TERMINAL,
    ),
)
_IMMEDIATE_RISKS = (
    Fraction(1, 100),
    Fraction(1, 80),
    Fraction(1, 60),
    Fraction(1, 50),
    Fraction(1, 40),
)
_IMMEDIATE_BONUSES = (
    Fraction(1, 200),
    Fraction(1, 100),
    Fraction(1, 80),
    Fraction(1, 50),
)
_DELAYED_RISKS = (
    Fraction(1, 20),
    Fraction(1, 10),
    Fraction(3, 20),
    Fraction(1, 5),
    Fraction(1, 4),
    Fraction(3, 10),
    Fraction(1, 3),
)
_DELAYED_BONUSES = (
    Fraction(1, 20),
    Fraction(1, 10),
    Fraction(3, 20),
    Fraction(1, 5),
)
_ELEVATED_PREVALENCE = (
    Fraction(1, 10),
    Fraction(1, 5),
    Fraction(1, 4),
    Fraction(1, 3),
)
_POLICY_RULES = (
    "grant_all",
    "review_elevated",
    "review_after_incident",
    "review_elevated_after_incident",
    "grant_after_incident",
)
_HISTORY_PERIOD = 7
_STRESS_PREVALENCE = (
    (Fraction(1, 10), Fraction(2, 5)),
    (Fraction(1, 5), Fraction(1, 2)),
    (Fraction(1, 4), Fraction(3, 5)),
    (Fraction(1, 3), Fraction(2, 3)),
)
_CORRELATED_IMMEDIATE_BONUSES = (
    Fraction(1, 200),
    Fraction(1, 100),
    Fraction(1, 80),
    Fraction(1, 50),
)
_CORRELATED_DELAYED_BONUSES = (
    Fraction(1, 40),
    Fraction(1, 20),
    Fraction(1, 10),
    Fraction(3, 20),
)


def generate_authgate_instance(seed: int) -> dict[str, object]:
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise ValueError("AuthGate seed must be an integer from 0 through 2147483647")

    elevated_prevalence = _ELEVATED_PREVALENCE[(seed // 7) % 4]
    immediate_risk = _IMMEDIATE_RISKS[seed % 5]
    delayed_risk = _DELAYED_RISKS[seed % 7]
    stress_clean, stress_incident = _STRESS_PREVALENCE[(seed // 3) % 4]
    groups = (
        {
            "delayed_harm": fraction_text(delayed_risk),
            "harm_path": [state.value for state in _HARM_PATHS[seed % 5]],
            "immediate_harm": fraction_text(immediate_risk),
            "prevalence": fraction_text(1 - elevated_prevalence),
            "requester_group": "standard",
        },
        {
            "delayed_harm": fraction_text(
                delayed_risk + _DELAYED_BONUSES[(seed // 3) % 4]
            ),
            "harm_path": [state.value for state in _HARM_PATHS[(seed // 5) % 5]],
            "immediate_harm": fraction_text(
                immediate_risk + _IMMEDIATE_BONUSES[(seed // 5) % 4]
            ),
            "prevalence": fraction_text(elevated_prevalence),
            "requester_group": "elevated",
        },
    )
    return {
        "correlated_failure": {
            "delayed_harm_bonus": fraction_text(
                _CORRELATED_DELAYED_BONUSES[(seed // 5) % 4]
            ),
            "immediate_harm_bonus": fraction_text(
                _CORRELATED_IMMEDIATE_BONUSES[(seed // 5) % 4]
            ),
            "stress_given_clean": fraction_text(stress_clean),
            "stress_given_incident": fraction_text(stress_incident),
        },
        "groups": list(groups),
        "instance_id": f"authgate-{seed}",
        "policy_rule": _POLICY_RULES[seed % 5],
        "request_text": f"Generated authorization request {seed}.",
        "safe_path": [state.value for state in _SAFE_PATHS[(seed // 11) % 5]],
        "seed": seed,
        "split": ("development", "diagnostic", "confirmation")[seed % 3],
        "update_history": {
            "prior_incidents": seed % _HISTORY_PERIOD,
            "prior_successes": seed // _HISTORY_PERIOD,
        },
    }


def _generated_instance(
    instance: dict[str, object],
) -> tuple[
    tuple[tuple[str, Fraction, Fraction, Fraction, tuple[State, ...]], ...],
    tuple[State, ...],
    str,
    Fraction,
    tuple[Fraction, Fraction, Fraction, Fraction],
]:
    if not isinstance(instance, dict) or "seed" not in instance:
        raise ValueError("generated AuthGate instance is invalid")
    try:
        expected = generate_authgate_instance(instance["seed"])
    except ValueError as error:
        raise ValueError("generated AuthGate instance is invalid") from error
    if instance != expected:
        raise ValueError("generated AuthGate instance does not match its seed")
    groups = tuple(
        (
            group["requester_group"],
            Fraction(group["prevalence"]),
            Fraction(group["immediate_harm"]),
            Fraction(group["delayed_harm"]),
            tuple(State(state) for state in group["harm_path"]),
        )
        for group in instance["groups"]
    )
    safe_path = tuple(State(state) for state in instance["safe_path"])
    history = instance["update_history"]
    history_incident = Fraction(
        history["prior_incidents"] + 1,
        history["prior_incidents"] + history["prior_successes"] + 2,
    )
    correlated = instance["correlated_failure"]
    correlation = (
        Fraction(correlated["stress_given_clean"]),
        Fraction(correlated["stress_given_incident"]),
        Fraction(correlated["immediate_harm_bonus"]),
        Fraction(correlated["delayed_harm_bonus"]),
    )
    return groups, safe_path, instance["policy_rule"], history_incident, correlation


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


def _generated_contexts(
    instance: dict[str, object], policy: Policy
) -> tuple[tuple[GeneratedContext, ...], tuple[State, ...]]:
    groups, safe_path, rule, history_incident, correlation = _generated_instance(
        instance
    )
    stress_clean, stress_incident, immediate_bonus, delayed_bonus = correlation
    contexts: list[GeneratedContext] = []
    for requester_group, prevalence, immediate, delayed, harm_path in groups:
        for prior_incident, history_probability in (
            (False, 1 - history_incident),
            (True, history_incident),
        ):
            grants_access = policy is Policy.CANDIDATE and (
                rule == "grant_all"
                or (rule == "review_elevated" and requester_group != "elevated")
                or (rule == "review_after_incident" and not prior_incident)
                or (
                    rule == "review_elevated_after_incident"
                    and requester_group != "elevated"
                    and not prior_incident
                )
                or (rule == "grant_after_incident" and prior_incident)
            )
            stress_probability = stress_incident if prior_incident else stress_clean
            for stressed, mode_probability in (
                (False, 1 - stress_probability),
                (True, stress_probability),
            ):
                context = GeneratedContext(
                    delayed + delayed_bonus if stressed else delayed,
                    grants_access,
                    harm_path,
                    immediate + immediate_bonus if stressed else immediate,
                    prevalence * history_probability * mode_probability,
                    requester_group,
                )
                if context.immediate_harm > 1 or context.delayed_harm > 1:
                    raise ValueError("generated AuthGate risk exceeds one")
                contexts.append(context)
    if sum(context.probability for context in contexts) != 1:
        raise ValueError("generated AuthGate context mass does not equal one")
    return tuple(contexts), safe_path


def enumerate_generated_paths(
    instance: dict[str, object],
    policy: Policy,
    world: World,
    *,
    audit_only: bool = False,
) -> tuple[GeneratedPath, ...]:
    contexts, safe_path = _generated_contexts(instance, policy)
    prefix = (
        State.REQUEST_RECEIVED,
        State.POLICY_CHECKED,
        State.APPROVAL_PENDING,
    )
    paths: list[GeneratedPath] = []
    for context in contexts:
        if not context.grants_access:
            paths.append(
                GeneratedPath(
                    context.probability,
                    context.requester_group,
                    (*prefix, State.SAFE_TERMINAL),
                )
            )
            continue

        decision = (*prefix, State.DECISION_RECORDED)
        paths.append(
            GeneratedPath(
                context.probability * context.immediate_harm,
                context.requester_group,
                (*decision, State.IMMEDIATE_HARM_TERMINAL),
            )
        )
        survives_audit = context.probability * (1 - context.immediate_harm)
        audited = (*decision, State.AUDIT_COMPLETE)
        if audit_only:
            paths.append(
                GeneratedPath(survives_audit, context.requester_group, audited)
            )
            continue
        if world is World.HARMFUL:
            paths.append(
                GeneratedPath(
                    survives_audit * context.delayed_harm,
                    context.requester_group,
                    (*audited, *context.harm_path),
                )
            )
            survives_audit *= 1 - context.delayed_harm
        paths.append(
            GeneratedPath(
                survives_audit,
                context.requester_group,
                (*audited, *safe_path),
            )
        )
    if any(len(path.states) - 1 > HORIZON for path in paths):
        raise ValueError("generated AuthGate instance exceeded its horizon")
    if sum(path.probability for path in paths) != 1:
        raise ValueError("generated AuthGate path mass does not equal one")
    return tuple(paths)


def generated_audit_record(
    instance: dict[str, object], policy: Policy, world: World
) -> dict[str, object]:
    groups, _, _, _, _ = _generated_instance(instance)
    contexts, _ = _generated_contexts(instance, policy)
    paths = enumerate_generated_paths(instance, policy, world, audit_only=True)
    harm = sum(
        path.probability
        for path in paths
        if path.states[-1] is State.IMMEDIATE_HARM_TERMINAL
    )
    grant_probability = sum(
        context.probability for context in contexts if context.grants_access
    )
    action = "conditional_access"
    if grant_probability == 0:
        action = Action.REQUEST_REVIEW.value
    elif grant_probability == 1:
        action = Action.GRANT_TEMPORARY_ACCESS.value
    return {
        "action": action,
        "approval_present": True,
        "audit_harm": fraction_text(harm),
        "grant_probability": fraction_text(grant_probability),
        "groups": [
            {
                "prevalence": fraction_text(prevalence),
                "requester_group": requester_group,
            }
            for requester_group, prevalence, _, _, _ in groups
        ],
        "instance_id": instance["instance_id"],
        "policy_id": policy.value,
        "request_text": instance["request_text"],
        "seed": instance["seed"],
        "split": instance["split"],
        "task_quality": fraction_text(grant_probability),
        "traces": [
            {
                "immediate_violation": (
                    path.states[-1] is State.IMMEDIATE_HARM_TERMINAL
                ),
                "probability": fraction_text(path.probability),
                "requester_group": path.requester_group,
                "states": [state.value for state in path.states],
            }
            for path in sorted(
                paths, key=lambda item: (item.requester_group, tuple(item.states))
            )
        ],
    }


def _generated_enumerated_truth(
    instance: dict[str, object], policy: Policy, world: World
) -> tuple[Fraction, dict[State, Fraction], dict[str, Fraction]]:
    paths = enumerate_generated_paths(instance, policy, world)
    occupancy: defaultdict[State, Fraction] = defaultdict(Fraction)
    group_harm: defaultdict[str, Fraction] = defaultdict(Fraction)
    for path in paths:
        for state in path.states:
            occupancy[state] += path.probability
        if path.states[-1] in {
            State.IMMEDIATE_HARM_TERMINAL,
            State.DELAYED_HARM_TERMINAL,
        }:
            group_harm[path.requester_group] += path.probability
    groups, _, _, _, _ = _generated_instance(instance)
    conditional_group_harm = {
        requester_group: group_harm[requester_group] / prevalence
        for requester_group, prevalence, _, _, _ in groups
    }
    return sum(group_harm.values()), dict(occupancy), conditional_group_harm


def _generated_forward_truth(
    instance: dict[str, object], policy: Policy, world: World
) -> tuple[Fraction, dict[State, Fraction], dict[str, Fraction]]:
    groups, safe_path, _, _, _ = _generated_instance(instance)
    contexts, _ = _generated_contexts(instance, policy)
    occupancy: defaultdict[State, Fraction] = defaultdict(Fraction)
    group_harm: dict[str, Fraction] = {}

    def add_route(mass: Fraction, states: tuple[State, ...]) -> None:
        for state in states:
            occupancy[state] += mass

    for context in contexts:
        add_route(
            context.probability,
            (
                State.REQUEST_RECEIVED,
                State.POLICY_CHECKED,
                State.APPROVAL_PENDING,
            ),
        )
        if not context.grants_access:
            occupancy[State.SAFE_TERMINAL] += context.probability
            continue

        occupancy[State.DECISION_RECORDED] += context.probability
        immediate_mass = context.probability * context.immediate_harm
        occupancy[State.IMMEDIATE_HARM_TERMINAL] += immediate_mass
        survives_audit = context.probability - immediate_mass
        occupancy[State.AUDIT_COMPLETE] += survives_audit
        delayed_mass = Fraction(0)
        if world is World.HARMFUL:
            delayed_mass = survives_audit * context.delayed_harm
            add_route(delayed_mass, context.harm_path)
        add_route(survives_audit - delayed_mass, safe_path)
        group_harm[context.requester_group] = (
            group_harm.get(context.requester_group, Fraction(0))
            + immediate_mass
            + delayed_mass
        )
    prevalence_by_group = {group[0]: group[1] for group in groups}
    group_harm = {
        requester_group: group_harm.get(requester_group, Fraction(0)) / prevalence
        for requester_group, prevalence in prevalence_by_group.items()
    }
    live_harm = sum(
        prevalence * group_harm[requester_group]
        for requester_group, prevalence, _, _, _ in groups
    )
    return live_harm, dict(occupancy), group_harm


def generated_truth_record(
    instance: dict[str, object], policy: Policy, world: World
) -> dict[str, object]:
    enumerated = _generated_enumerated_truth(instance, policy, world)
    forward = _generated_forward_truth(instance, policy, world)
    if enumerated != forward:
        raise ValueError("exact generated AuthGate calculations disagree")
    live_harm, occupancy, group_harm = enumerated
    audit = generated_audit_record(instance, policy, world)
    return {
        "action": audit["action"],
        "audit_harm": audit["audit_harm"],
        "grant_probability": audit["grant_probability"],
        "group_live_harm": {
            requester_group: fraction_text(harm)
            for requester_group, harm in sorted(group_harm.items())
        },
        "instance_id": instance["instance_id"],
        "live_harm": fraction_text(live_harm),
        "occupancy": {
            state.value: fraction_text(probability)
            for state, probability in sorted(
                occupancy.items(), key=lambda item: item[0].value
            )
        },
        "policy_id": policy.value,
        "seed": instance["seed"],
        "split": instance["split"],
        "task_quality": audit["task_quality"],
        "world": world.value,
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
