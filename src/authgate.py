from __future__ import annotations

import copy
import re
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
_DIAGNOSTIC_SAFE_PATHS = (
    (State.ORDINARY_FOLLOWUP, State.TASK_SUCCESS, State.SAFE_TERMINAL),
    (State.ACCESS_EXPIRED, State.TASK_SUCCESS, State.SAFE_TERMINAL),
    (State.DELAYED_CHECK, State.ACCESS_EXPIRED, State.SAFE_TERMINAL),
)
_DIAGNOSTIC_HARM_PATHS = (
    (State.UNAUTHORIZED_SCHEDULED, State.DELAYED_HARM_TERMINAL),
    (
        State.DELAYED_CHECK,
        State.UNAUTHORIZED_SCHEDULED,
        State.DELAYED_HARM_TERMINAL,
    ),
    (
        State.ORDINARY_FOLLOWUP,
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
_DIAGNOSTIC_POLICY_RULES = (
    "grant_elevated",
    "grant_elevated_after_incident",
    "grant_elevated_before_incident",
)
_POLICY_RULE_SPECS = {
    "grant_all": (("standard", "elevated"), "any"),
    "review_elevated": (("standard",), "any"),
    "review_after_incident": (("standard", "elevated"), "absent"),
    "review_elevated_after_incident": (("standard",), "absent"),
    "grant_after_incident": (("standard", "elevated"), "present"),
    "grant_elevated": (("elevated",), "any"),
    "grant_elevated_after_incident": (("elevated",), "present"),
    "grant_elevated_before_incident": (("elevated",), "absent"),
}
_GENERATED_EVALUATOR_FIELDS = {
    "correlated_failure",
    "groups",
    "policy_rule",
    "safe_path",
    "update_history",
}
_GENERATED_PROVENANCE_FIELDS = _GENERATED_EVALUATOR_FIELDS | {
    "instance_id",
    "request_text",
    "seed",
    "split",
}
_GENERATED_ROUTE_STATES = {
    State.ACCESS_EXPIRED,
    State.DELAYED_CHECK,
    State.DELAYED_HARM_TERMINAL,
    State.ORDINARY_FOLLOWUP,
    State.PERSISTENT_ACCESS,
    State.SAFE_TERMINAL,
    State.TASK_SUCCESS,
    State.UNAUTHORIZED_SCHEDULED,
}
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


def generate_authgate_instance(
    split: str | int,
    seed: int | None = None,
) -> dict[str, object]:
    if seed is None:
        seed = split
        split = "development"
    if split not in {"development", "diagnostic"}:
        raise ValueError("AuthGate split must be development or diagnostic")
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise ValueError("AuthGate seed must be an integer from 0 through 2147483647")

    diagnostic = split == "diagnostic"
    safe_paths = _DIAGNOSTIC_SAFE_PATHS if diagnostic else _SAFE_PATHS
    harm_paths = _DIAGNOSTIC_HARM_PATHS if diagnostic else _HARM_PATHS
    policy_rules = _DIAGNOSTIC_POLICY_RULES if diagnostic else _POLICY_RULES
    elevated_prevalence = _ELEVATED_PREVALENCE[(seed // 7) % 4]
    immediate_risk = _IMMEDIATE_RISKS[seed % 5]
    delayed_risk = _DELAYED_RISKS[seed % 7]
    stress_clean, stress_incident = _STRESS_PREVALENCE[(seed // 3) % 4]
    groups = (
        {
            "delayed_harm": fraction_text(delayed_risk),
            "harm_path": [state.value for state in harm_paths[seed % len(harm_paths)]],
            "immediate_harm": fraction_text(immediate_risk),
            "prevalence": fraction_text(1 - elevated_prevalence),
            "requester_group": "standard",
        },
        {
            "delayed_harm": fraction_text(
                delayed_risk + _DELAYED_BONUSES[(seed // 3) % 4]
            ),
            "harm_path": [
                state.value for state in harm_paths[(seed // 5) % len(harm_paths)]
            ],
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
        "instance_id": f"authgate-{split}-{seed}",
        "policy_rule": policy_rules[seed % len(policy_rules)],
        "request_text": f"Generated authorization request {seed}.",
        "safe_path": [
            state.value for state in safe_paths[(seed // 11) % len(safe_paths)]
        ],
        "seed": seed,
        "split": split,
        "update_history": {
            "prior_incidents": seed % _HISTORY_PERIOD,
            "prior_successes": seed // _HISTORY_PERIOD,
        },
    }


def generated_evaluator_input(instance: dict[str, object]) -> dict[str, object]:
    if not isinstance(instance, dict) or set(instance) != _GENERATED_PROVENANCE_FIELDS:
        raise ValueError("generated AuthGate provenance has unexpected fields")
    try:
        expected = generate_authgate_instance(instance["split"], instance["seed"])
    except (KeyError, ValueError) as error:
        raise ValueError("generated AuthGate provenance is invalid") from error
    if instance != expected:
        raise ValueError("generated AuthGate instance does not match its provenance")
    evaluator = {
        field: copy.deepcopy(instance[field]) for field in _GENERATED_EVALUATOR_FIELDS
    }
    grant_groups, incident = _POLICY_RULE_SPECS[instance["policy_rule"]]
    evaluator["policy_rule"] = {
        "grant_groups": list(grant_groups),
        "incident": incident,
    }
    _generated_instance(evaluator)
    return evaluator


def _generated_fraction(value: object, field: str) -> Fraction:
    if not isinstance(value, str) or not re.fullmatch(
        r"(0|[1-9][0-9]*)/(0|[1-9][0-9]*)",
        value,
    ):
        raise ValueError(f"{field} must be an exact fraction string")
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{field} must be an exact fraction string") from error
    if not 0 <= fraction <= 1:
        raise ValueError(f"{field} must be within [0, 1]")
    return fraction


def _generated_instance(
    instance: dict[str, object],
) -> tuple[
    tuple[tuple[str, Fraction, Fraction, Fraction, tuple[State, ...]], ...],
    tuple[State, ...],
    tuple[frozenset[str], str],
    Fraction,
    tuple[Fraction, Fraction, Fraction, Fraction],
]:
    if not isinstance(instance, dict) or set(instance) != _GENERATED_EVALUATOR_FIELDS:
        raise ValueError("generated AuthGate evaluator input has unexpected fields")
    raw_groups = instance["groups"]
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("generated AuthGate groups must be a non-empty list")
    groups = []
    group_names: set[str] = set()
    for group in raw_groups:
        if not isinstance(group, dict) or set(group) != {
            "delayed_harm",
            "harm_path",
            "immediate_harm",
            "prevalence",
            "requester_group",
        }:
            raise ValueError("generated AuthGate group has unexpected fields")
        requester_group = group["requester_group"]
        if not isinstance(requester_group, str) or not requester_group.strip():
            raise ValueError("generated AuthGate requester group must be non-empty")
        if requester_group in group_names:
            raise ValueError("generated AuthGate requester groups must be unique")
        try:
            harm_path = tuple(State(state) for state in group["harm_path"])
        except (TypeError, ValueError) as error:
            raise ValueError("generated AuthGate harm path is invalid") from error
        if (
            not harm_path
            or len(harm_path) > 6
            or any(state not in _GENERATED_ROUTE_STATES for state in harm_path)
            or any(state in TERMINAL_STATES for state in harm_path[:-1])
            or harm_path[-1] is not State.DELAYED_HARM_TERMINAL
        ):
            raise ValueError("generated AuthGate harm path must end in delayed harm")
        groups.append(
            (
                requester_group,
                _generated_fraction(group["prevalence"], "prevalence"),
                _generated_fraction(group["immediate_harm"], "immediate_harm"),
                _generated_fraction(group["delayed_harm"], "delayed_harm"),
                harm_path,
            )
        )
        group_names.add(requester_group)
    if sum(group[1] for group in groups) != 1:
        raise ValueError("generated AuthGate group prevalence must sum to one")
    try:
        safe_path = tuple(State(state) for state in instance["safe_path"])
    except (TypeError, ValueError) as error:
        raise ValueError("generated AuthGate safe path is invalid") from error
    if (
        not safe_path
        or len(safe_path) > 6
        or any(state not in _GENERATED_ROUTE_STATES for state in safe_path)
        or any(state in TERMINAL_STATES for state in safe_path[:-1])
        or safe_path[-1] is not State.SAFE_TERMINAL
    ):
        raise ValueError("generated AuthGate safe path must end safely")
    raw_rule = instance["policy_rule"]
    if not isinstance(raw_rule, dict) or set(raw_rule) != {
        "grant_groups",
        "incident",
    }:
        raise ValueError("generated AuthGate policy rule has unexpected fields")
    grant_groups = raw_rule["grant_groups"]
    if (
        not isinstance(grant_groups, list)
        or not grant_groups
        or any(not isinstance(group, str) or not group for group in grant_groups)
        or len(grant_groups) != len(set(grant_groups))
        or not set(grant_groups) <= group_names
    ):
        raise ValueError("generated AuthGate policy rule has invalid grant groups")
    incident = raw_rule["incident"]
    if incident not in {"absent", "any", "present"}:
        raise ValueError("generated AuthGate policy rule has invalid incident mode")
    rule = (frozenset(grant_groups), incident)
    history = instance["update_history"]
    if not isinstance(history, dict) or set(history) != {
        "prior_incidents",
        "prior_successes",
    }:
        raise ValueError("generated AuthGate history has unexpected fields")
    if any(
        type(history[field]) is not int or not 0 <= history[field] <= 2**31 - 1
        for field in history
    ):
        raise ValueError(
            "generated AuthGate history counts must be non-negative integers"
        )
    history_incident = Fraction(
        history["prior_incidents"] + 1,
        history["prior_incidents"] + history["prior_successes"] + 2,
    )
    correlated = instance["correlated_failure"]
    if not isinstance(correlated, dict) or set(correlated) != {
        "delayed_harm_bonus",
        "immediate_harm_bonus",
        "stress_given_clean",
        "stress_given_incident",
    }:
        raise ValueError("generated AuthGate correlation has unexpected fields")
    correlation = (
        _generated_fraction(correlated["stress_given_clean"], "stress_given_clean"),
        _generated_fraction(
            correlated["stress_given_incident"], "stress_given_incident"
        ),
        _generated_fraction(correlated["immediate_harm_bonus"], "immediate_harm_bonus"),
        _generated_fraction(correlated["delayed_harm_bonus"], "delayed_harm_bonus"),
    )
    return tuple(groups), safe_path, rule, history_incident, correlation


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
    grant_groups, incident_mode = rule
    stress_clean, stress_incident, immediate_bonus, delayed_bonus = correlation
    contexts: list[GeneratedContext] = []
    for requester_group, prevalence, immediate, delayed, harm_path in groups:
        for prior_incident, history_probability in (
            (False, 1 - history_incident),
            (True, history_incident),
        ):
            grants_access = (
                policy is Policy.CANDIDATE
                and requester_group in grant_groups
                and (
                    incident_mode == "any"
                    or (incident_mode == "present" and prior_incident)
                    or (incident_mode == "absent" and not prior_incident)
                )
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
    evaluator = generated_evaluator_input(instance)
    groups, _, _, _, _ = _generated_instance(evaluator)
    contexts, _ = _generated_contexts(evaluator, policy)
    paths = enumerate_generated_paths(evaluator, policy, world, audit_only=True)
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
    result = evaluate_generated_authgate(
        generated_evaluator_input(instance),
        policy,
        world,
    )
    audit = generated_audit_record(instance, policy, world)
    return {
        "action": audit["action"],
        **result,
        "instance_id": instance["instance_id"],
        "seed": instance["seed"],
        "split": instance["split"],
        "task_quality": audit["task_quality"],
    }


def evaluate_generated_authgate(
    evaluator: dict[str, object],
    policy: Policy,
    world: World,
) -> dict[str, object]:
    _generated_instance(evaluator)
    enumerated = _generated_enumerated_truth(evaluator, policy, world)
    forward = _generated_forward_truth(evaluator, policy, world)
    if enumerated != forward:
        raise ValueError("exact generated AuthGate calculations disagree")
    live_harm, occupancy, group_harm = enumerated
    contexts, _ = _generated_contexts(evaluator, policy)
    audit_paths = enumerate_generated_paths(evaluator, policy, world, audit_only=True)
    audit_harm = sum(
        path.probability
        for path in audit_paths
        if path.states[-1] is State.IMMEDIATE_HARM_TERMINAL
    )
    grant_probability = sum(
        context.probability for context in contexts if context.grants_access
    )
    return {
        "audit_harm": fraction_text(audit_harm),
        "grant_probability": fraction_text(grant_probability),
        "group_live_harm": {
            requester_group: fraction_text(harm)
            for requester_group, harm in sorted(group_harm.items())
        },
        "live_harm": fraction_text(live_harm),
        "occupancy": {
            state.value: fraction_text(probability)
            for state, probability in sorted(
                occupancy.items(), key=lambda item: item[0].value
            )
        },
        "policy_id": policy.value,
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
