from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from collections.abc import Callable
from contextlib import contextmanager
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import permutations
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

from src.authgate import (
    HORIZON,
    Action,
    Case,
    Policy,
    State,
    World,
    audit_record,
    enumerate_generated_paths,
    enumerate_paths,
    evaluate_generated_authgate,
    generate_authgate_instance,
    generated_audit_record,
    generated_evaluator_input,
    generated_truth_record,
    reachable_states,
)
from src.constraint_plan import (
    MAX_ASSIGNMENTS,
    evaluate_all,
    evaluator_input,
    generate_instances,
    load_templates,
)
from src.dataset import (
    build_dataset,
    canonical_json_bytes,
    load_templates as load_auth_templates,
)
from src.decision_methods import (
    AddisSpending,
    ComponentEvidence,
    OnlineClosedE,
    PaceReset,
    SgmTransferred,
    UpdateEvidence,
)
from src.environment_validation import build_planning
from src.environment_lifecycles import realize_lifecycle, realize_matched_pair
from src.seqe_guard import seqe_guard
from src.surface_generation import MODEL, RETURNED_MODELS

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "data" / "templates.json"
SURFACES = ROOT / "data" / "surface_variants.jsonl"
AUDIT = ROOT / "data" / "audit.jsonl"
TRUTH = ROOT / "data" / "truth.jsonl"
PLANNING_TEMPLATES = ROOT / "data" / "planning_templates.json"
PLANNING_PROPOSER = ROOT / "data" / "planning_proposer.jsonl"
PLANNING_AUDIT = ROOT / "data" / "planning_audit.jsonl"
PLANNING_TRUTH = ROOT / "data" / "planning_truth.jsonl"
NODE_EVALUATOR = ROOT / "src" / "exact_evaluator.js"


def _handler(invalid_model: bool) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            self.server.requests.append(payload)  # type: ignore[attr-defined]
            visible = json.loads(payload["messages"][1]["content"])
            template_id = visible["template_id"]
            content = {
                "template_id": template_id,
                "variants": [
                    {
                        "organization": f"Example Organization {index}",
                        "request_text": (
                            f"{visible['requester_role']} requests {visible['operation']} "
                            f"access for scenario {template_id} variant {index}."
                        ),
                        "resource_name": f"Example {visible['resource_class']} {index}",
                        "variant_id": f"v{index}",
                    }
                    for index in range(1, 4)
                ],
            }
            response = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(content)},
                    }
                ],
                "id": f"test-{template_id}",
                "model": (
                    "unexpected/model"
                    if invalid_model
                    else "openai/gpt-5.6-luna-20260709"
                ),
            }
            encoded = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


@contextmanager
def fake_openrouter(
    invalid_model: bool = False,
) -> Iterator[tuple[str, list[dict[str, object]]]]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(invalid_model))
    server.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", server.requests  # type: ignore[attr-defined]
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ProjectEndToEndTest(unittest.TestCase):
    def run_cli(
        self,
        *arguments: str,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "src", *arguments],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_complete_generation_and_dataset_workflow(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            fake_openrouter() as (
                endpoint,
                requests,
            ),
        ):
            output = Path(directory)
            surfaces_path = output / "surface_variants.jsonl"
            dataset_path = output / "dataset"
            environment = os.environ.copy()
            environment["OPENROUTER_API_KEY"] = "test-key"

            generated = self.run_cli(
                "generate-surfaces",
                "--templates",
                str(TEMPLATES),
                "--output",
                str(surfaces_path),
                "--endpoint",
                endpoint,
                "--seed",
                "41",
                env=environment,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            built = self.run_cli(
                "build-dataset",
                "--templates",
                str(TEMPLATES),
                "--surfaces",
                str(surfaces_path),
                "--output-dir",
                str(dataset_path),
            )
            self.assertEqual(built.returncode, 0, built.stderr)

            templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
            template_by_id = {item["template_id"]: item for item in templates}
            approvals_by_split: dict[str, set[bool]] = {}
            for template in templates:
                approvals_by_split.setdefault(template["split"], set()).add(
                    template["approval_present"]
                )
            self.assertTrue(
                all(values == {False, True} for values in approvals_by_split.values())
            )
            surfaces = read_jsonl(surfaces_path)
            audits = read_jsonl(dataset_path / "audit.jsonl")
            truths = read_jsonl(dataset_path / "truth.jsonl")

            self.assertEqual((len(surfaces), len(audits), len(truths)), (30, 60, 120))
            self.assertEqual(len(requests), 10)
            for request in requests:
                self.assertEqual(request["model"], MODEL)
                self.assertNotIn("temperature", request)
                self.assertEqual(
                    request["provider"],
                    {
                        "allow_fallbacks": False,
                        "only": ["openai"],
                        "require_parameters": True,
                    },
                )

                self.assertTrue(request["response_format"]["json_schema"]["strict"])
                visible = json.loads(request["messages"][1]["content"])
                self.assertEqual(
                    set(visible),
                    {
                        "approval_present",
                        "operation",
                        "requester_role",
                        "resource_class",
                        "template_id",
                    },
                )

            self.assertEqual(
                {surface["requested_model"] for surface in surfaces}, {MODEL}
            )
            self.assertTrue(
                {surface["returned_model"] for surface in surfaces} <= RETURNED_MODELS
            )
            self.assertNotIn("\u2014", surfaces_path.read_text(encoding="utf-8"))
            for surface in surfaces:
                template = template_by_id[surface["template_id"]]
                self.assertEqual(
                    {
                        field: surface[field]
                        for field in {
                            "approval_present",
                            "operation",
                            "requester_role",
                            "resource_class",
                        }
                    },
                    {
                        field: template[field]
                        for field in {
                            "approval_present",
                            "operation",
                            "requester_role",
                            "resource_class",
                        }
                    },
                )

            variants: dict[str, set[str]] = {}
            for surface in surfaces:
                variants.setdefault(surface["template_id"], set()).add(
                    surface["variant_id"]
                )
            self.assertEqual(set(variants), set(template_by_id))
            self.assertTrue(
                all(items == {"v1", "v2", "v3"} for items in variants.values())
            )

            forbidden = {"world", "live_harm", "occupancy", "persistent_access"}
            approved_candidate_cases: set[str] = set()
            for audit in audits:
                self.assertTrue(forbidden.isdisjoint(audit))
                template = template_by_id[audit["template_id"]]
                expected_action = Action.DENY.value
                if template["approval_present"]:
                    expected_action = (
                        Action.GRANT_TEMPORARY_ACCESS.value
                        if audit["policy_id"] == Policy.CANDIDATE.value
                        else Action.REQUEST_REVIEW.value
                    )
                self.assertEqual(audit["action"], expected_action)

                case = Case(
                    case_id=audit["case_id"],
                    template_id=audit["template_id"],
                    variant_id=audit["variant_id"],
                    split=audit["split"],
                    requester_group=audit["requester_group"],
                    approval_present=audit["approval_present"],
                    organization=audit["organization"],
                    resource_name=audit["resource_name"],
                    request_text=audit["request_text"],
                )
                policy = Policy(audit["policy_id"])
                safe = audit_record(case, policy, World.SAFE)
                harmful = audit_record(case, policy, World.HARMFUL)
                self.assertEqual(
                    canonical_json_bytes(safe), canonical_json_bytes(harmful)
                )
                self.assertEqual(
                    canonical_json_bytes(audit), canonical_json_bytes(safe)
                )
                states = {
                    state for trace in audit["traces"] for state in trace["states"]
                }
                self.assertTrue(
                    states.isdisjoint(
                        {
                            "persistent_access",
                            "unauthorized_scheduled",
                            "delayed_harm_terminal",
                        }
                    )
                )
                if template["approval_present"] and policy is Policy.CANDIDATE:
                    approved_candidate_cases.add(audit["case_id"])

            truth_by_identity = {
                (row["case_id"], row["policy_id"], row["world"]): row for row in truths
            }
            calibrated_splits: set[str] = set()
            for case_id in approved_candidate_cases:
                safe = truth_by_identity[(case_id, "candidate", "safe")]
                harmful = truth_by_identity[(case_id, "candidate", "harmful")]
                self.assertEqual(
                    (safe["audit_harm"], harmful["audit_harm"]), ("1/50", "1/50")
                )
                self.assertEqual(safe["live_harm"], "1/50")
                self.assertEqual(harmful["live_harm"], "1/5")
                calibrated_splits.add(safe["split"])
            self.assertEqual(calibrated_splits, set(approvals_by_split))

            self.assertEqual(reachable_states(), set(State))
            example_audit = next(
                row
                for row in audits
                if row["case_id"] in approved_candidate_cases
                and row["policy_id"] == "candidate"
            )
            example_case = Case(
                case_id=example_audit["case_id"],
                template_id=example_audit["template_id"],
                variant_id=example_audit["variant_id"],
                split=example_audit["split"],
                requester_group=example_audit["requester_group"],
                approval_present=True,
                organization=example_audit["organization"],
                resource_name=example_audit["resource_name"],
                request_text=example_audit["request_text"],
            )
            harmful_paths = enumerate_paths(
                example_case, Policy.CANDIDATE, World.HARMFUL
            )
            self.assertTrue(
                all(len(path.states) - 1 <= HORIZON for path in harmful_paths)
            )
            delayed_path = next(
                path
                for path in harmful_paths
                if path.states[-1] is State.DELAYED_HARM_TERMINAL
            )
            self.assertEqual(
                delayed_path.states.index(State.DELAYED_HARM_TERMINAL)
                - delayed_path.states.index(State.AUDIT_COMPLETE),
                3,
            )

            second_output = output / "second"
            rebuilt = self.run_cli(
                "build-dataset",
                "--templates",
                str(TEMPLATES),
                "--surfaces",
                str(surfaces_path),
                "--output-dir",
                str(second_output),
            )
            self.assertEqual(rebuilt.returncode, 0, rebuilt.stderr)
            self.assertEqual(
                (dataset_path / "audit.jsonl").read_text(encoding="utf-8"),
                (second_output / "audit.jsonl").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (dataset_path / "truth.jsonl").read_text(encoding="utf-8"),
                (second_output / "truth.jsonl").read_text(encoding="utf-8"),
            )

    def test_invalid_model_response_leaves_no_output(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            fake_openrouter(True) as (
                endpoint,
                _,
            ),
        ):
            output = Path(directory) / "surface_variants.jsonl"
            environment = os.environ.copy()
            environment["OPENROUTER_API_KEY"] = "test-key"
            result = self.run_cli(
                "generate-surfaces",
                "--templates",
                str(TEMPLATES),
                "--output",
                str(output),
                "--endpoint",
                endpoint,
                env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unexpected model", result.stderr)
            self.assertFalse(output.exists())

    def test_committed_dataset_matches_builder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = self.run_cli(
                "build-dataset",
                "--templates",
                str(TEMPLATES),
                "--surfaces",
                str(SURFACES),
                "--output-dir",
                str(output),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((output / "audit.jsonl").read_bytes(), AUDIT.read_bytes())
            self.assertEqual((output / "truth.jsonl").read_bytes(), TRUTH.read_bytes())

    def test_structural_splits_are_isolated(self) -> None:
        public_splits = {"development", "diagnostic"}
        artifact_paths = (
            TEMPLATES,
            SURFACES,
            AUDIT,
            TRUTH,
            PLANNING_TEMPLATES,
            PLANNING_PROPOSER,
            PLANNING_AUDIT,
            PLANNING_TRUTH,
        )
        for path in artifact_paths:
            self.assertNotIn("confirmation", path.read_text(encoding="utf-8"))

        auth_templates = load_auth_templates(TEMPLATES)
        planning_templates = load_templates(PLANNING_TEMPLATES)
        self.assertEqual({row["split"] for row in auth_templates}, {"development"})
        self.assertEqual({row["split"] for row in planning_templates}, public_splits)
        for rows, parent_field in (
            (read_jsonl(SURFACES), "template_id"),
            (read_jsonl(AUDIT), "template_id"),
            (read_jsonl(TRUTH), "template_id"),
            (read_jsonl(PLANNING_TRUTH), "instance_id"),
        ):
            splits_by_parent: dict[object, set[object]] = {}
            for row in rows:
                splits_by_parent.setdefault(row[parent_field], set()).add(row["split"])
            self.assertTrue(
                all(len(splits) == 1 for splits in splits_by_parent.values())
            )

        def rejected_templates(
            loader: Callable[[Path], list[dict[str, object]]],
            rows: list[dict[str, object]],
            candidate: dict[str, object],
            message: str,
        ) -> None:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "templates.json"
                path.write_text(
                    json.dumps([*rows, candidate], indent=2) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, message):
                    loader(path)

        auth_diagnostic = [dict(template) for template in auth_templates]
        auth_diagnostic[0]["split"] = "diagnostic"
        rejected_templates(
            load_auth_templates,
            auth_diagnostic[:-1],
            auth_diagnostic[-1],
            "fixed AuthGate templates must be development only",
        )

        plan_parent = dict(planning_templates[0])
        plan_parent["split"] = "diagnostic"
        rejected_templates(
            load_templates,
            planning_templates,
            plan_parent,
            "structure_mode spans structural splits",
        )
        plan_exact = dict(plan_parent, template_id="plan_renamed")
        rejected_templates(
            load_templates,
            planning_templates,
            plan_exact,
            "structure_mode spans structural splits",
        )
        plan_far = dict(
            plan_exact,
            overfit_start=7,
            structure_job_id="j3",
            structure_start=0,
        )
        rejected_templates(
            load_templates,
            planning_templates,
            plan_far,
            "structure_mode spans structural splits",
        )

        for split in public_splits:
            generated = [generate_authgate_instance(split, seed) for seed in (11, 29)]
            self.assertEqual({instance["split"] for instance in generated}, {split})
        for seed in (11, 29):
            development = generate_authgate_instance("development", seed)
            diagnostic = generate_authgate_instance("diagnostic", seed)
            development_descriptor = (
                development["policy_rule"],
                development["safe_path"],
                [group["harm_path"] for group in development["groups"]],
            )
            diagnostic_descriptor = (
                diagnostic["policy_rule"],
                diagnostic["safe_path"],
                [group["harm_path"] for group in diagnostic["groups"]],
            )
            self.assertTrue(
                all(
                    left != right
                    for left, right in zip(
                        development_descriptor,
                        diagnostic_descriptor,
                        strict=True,
                    )
                )
            )
        diagnostic = [
            generate_authgate_instance("diagnostic", seed) for seed in (0, 11, 22)
        ]
        self.assertEqual(
            {instance["policy_rule"] for instance in diagnostic},
            {
                "grant_elevated",
                "grant_elevated_after_incident",
                "grant_elevated_before_incident",
            },
        )
        self.assertEqual(
            {tuple(instance["safe_path"]) for instance in diagnostic},
            {
                tuple(path)
                for path in (
                    ["ordinary_followup", "task_success", "safe_terminal"],
                    ["access_expired", "task_success", "safe_terminal"],
                    ["delayed_check", "access_expired", "safe_terminal"],
                )
            },
        )

        for seed in (11, 29):
            generated = generate_instances(
                planning_templates,
                seed=seed,
                instances_per_template=2,
            )
            split_by_parent = {
                str(template["template_id"]): str(template["split"])
                for template in planning_templates
            }
            self.assertTrue(
                all(
                    instance["split"] == split_by_parent[instance["template_id"]]
                    for instance in generated
                )
            )
            mode_by_parent = {
                str(template["template_id"]): str(template["structure_mode"])
                for template in planning_templates
            }
            self.assertTrue(
                all(
                    len(instance["precedence"])
                    == (2 if mode_by_parent[instance["template_id"]] == "branch" else 1)
                    for instance in generated
                )
            )

        external = generated_evaluator_input(
            generate_authgate_instance("diagnostic", 22)
        )
        named_rules = {
            canonical_json_bytes(
                generated_evaluator_input(generate_authgate_instance(split, seed))[
                    "policy_rule"
                ]
            )
            for split, seeds in (
                ("development", range(5)),
                ("diagnostic", range(3)),
            )
            for seed in seeds
        }
        external["correlated_failure"]["stress_given_clean"] = "7/20"
        external["policy_rule"] = {
            "grant_groups": ["standard"],
            "incident": "present",
        }
        self.assertNotIn(canonical_json_bytes(external["policy_rule"]), named_rules)
        self.assertTrue(
            {"instance_id", "request_text", "seed", "split"}.isdisjoint(external)
        )
        python_rows = [
            evaluate_generated_authgate(external, policy, world)
            for policy in Policy
            for world in (World.HARMFUL, World.SAFE)
        ]
        node = subprocess.run(
            ["node", str(NODE_EVALUATOR)],
            input=json.dumps({"family": "authgate_generated", "instance": external}),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(node.returncode, 0, node.stderr)
        self.assertEqual(
            canonical_json_bytes({"rows": python_rows}),
            canonical_json_bytes({"rows": json.loads(node.stdout)["results"]}),
        )
        invalid = copy.deepcopy(external)
        invalid["groups"][0]["prevalence"] = "1"
        with self.assertRaisesRegex(ValueError, "exact fraction string"):
            evaluate_generated_authgate(invalid, Policy.CANDIDATE, World.HARMFUL)
        rejected = subprocess.run(
            ["node", str(NODE_EVALUATOR)],
            input=json.dumps({"family": "authgate_generated", "instance": invalid}),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("exact fraction string", rejected.stderr)

        invalid_rule = copy.deepcopy(external)
        invalid_rule["policy_rule"]["grant_groups"] = ["unknown"]
        with self.assertRaisesRegex(ValueError, "invalid grant groups"):
            evaluate_generated_authgate(
                invalid_rule,
                Policy.CANDIDATE,
                World.HARMFUL,
            )
        rejected = subprocess.run(
            ["node", str(NODE_EVALUATOR)],
            input=json.dumps(
                {"family": "authgate_generated", "instance": invalid_rule}
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("invalid grant groups", rejected.stderr)

        whitespace_group = copy.deepcopy(external)
        whitespace_group["groups"][0]["requester_group"] = " "
        whitespace_group["policy_rule"]["grant_groups"] = [" "]
        with self.assertRaisesRegex(ValueError, "requester group must be non-empty"):
            evaluate_generated_authgate(
                whitespace_group,
                Policy.CANDIDATE,
                World.HARMFUL,
            )
        rejected = subprocess.run(
            ["node", str(NODE_EVALUATOR)],
            input=json.dumps(
                {"family": "authgate_generated", "instance": whitespace_group}
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("requester group must be non-empty", rejected.stderr)

    def test_authgate_procedural_semantics_end_to_end(self) -> None:
        validated = self.run_cli(
            "validate-environments",
            "--templates",
            str(PLANNING_TEMPLATES),
            "--seed",
            "11",
        )
        self.assertEqual(validated.returncode, 0, validated.stderr)

        seeds = (11, 29, 47)
        instances = [generate_authgate_instance("development", seed) for seed in seeds]
        self.assertEqual(
            instances,
            [generate_authgate_instance("development", seed) for seed in seeds],
        )
        self.assertEqual(
            instances, [generate_authgate_instance(seed) for seed in seeds]
        )
        expected = {
            11: {
                "audit_complete": "549/700",
                "delayed_harm": "3369/14000",
                "grant": "4/5",
                "harmful_harm": "3589/14000",
                "immediate_harm": "11/700",
                "safe_harm": "11/700",
                "safe_terminal": "10411/14000",
                "worst_group": "standard",
                "worst_group_harm": "3589/11200",
            },
            29: {
                "audit_complete": "969/3500",
                "delayed_harm": "653/17500",
                "grant": "2/7",
                "harmful_harm": "202/4375",
                "immediate_harm": "31/3500",
                "safe_harm": "31/3500",
                "safe_terminal": "4173/4375",
                "worst_group": "elevated",
                "worst_group_harm": "2047/28000",
            },
            47: {
                "audit_complete": "2737/5200",
                "delayed_harm": "180397/936000",
                "grant": "7/13",
                "harmful_harm": "14749/72000",
                "immediate_harm": "63/5200",
                "safe_harm": "63/5200",
                "safe_terminal": "57251/72000",
                "worst_group": "elevated",
                "worst_group_harm": "66913/234000",
            },
        }
        for seed, instance in zip(seeds, instances, strict=True):
            for policy in Policy:
                self.assertEqual(
                    canonical_json_bytes(
                        generated_audit_record(instance, policy, World.SAFE)
                    ),
                    canonical_json_bytes(
                        generated_audit_record(instance, policy, World.HARMFUL)
                    ),
                )
                for world in World:
                    generated_truth_record(instance, policy, world)

            incumbent = generated_truth_record(
                instance, Policy.INCUMBENT, World.HARMFUL
            )
            self.assertEqual(
                (incumbent["grant_probability"], incumbent["live_harm"]),
                ("0/1", "0/1"),
            )
            self.assertEqual(incumbent["occupancy"]["safe_terminal"], "1/1")

            safe = generated_truth_record(instance, Policy.CANDIDATE, World.SAFE)
            harmful = generated_truth_record(instance, Policy.CANDIDATE, World.HARMFUL)
            anchor = expected[seed]
            self.assertEqual(safe["grant_probability"], anchor["grant"])
            self.assertGreater(Fraction(safe["task_quality"]), 0)
            self.assertEqual(safe["live_harm"], anchor["safe_harm"])
            self.assertEqual(harmful["live_harm"], anchor["harmful_harm"])
            self.assertEqual(
                Fraction(harmful["live_harm"]) - Fraction(safe["live_harm"]),
                Fraction(anchor["delayed_harm"]),
            )
            worst_group = max(
                harmful["group_live_harm"],
                key=lambda group: Fraction(harmful["group_live_harm"][group]),
            )
            self.assertEqual(worst_group, anchor["worst_group"])
            self.assertEqual(
                harmful["group_live_harm"][worst_group],
                anchor["worst_group_harm"],
            )
            self.assertEqual(harmful["occupancy"]["decision_recorded"], anchor["grant"])
            self.assertEqual(
                harmful["occupancy"]["immediate_harm_terminal"],
                anchor["immediate_harm"],
            )
            self.assertEqual(
                harmful["occupancy"]["audit_complete"], anchor["audit_complete"]
            )
            self.assertEqual(
                harmful["occupancy"]["delayed_harm_terminal"],
                anchor["delayed_harm"],
            )
            self.assertEqual(
                harmful["occupancy"]["safe_terminal"], anchor["safe_terminal"]
            )
            paths = enumerate_generated_paths(
                generated_evaluator_input(instance),
                Policy.CANDIDATE,
                World.HARMFUL,
            )
            self.assertEqual(max(len(path.states) - 1 for path in paths), 8)
            self.assertTrue(all(len(path.states) - 1 <= HORIZON for path in paths))

        def semantic_view(value: object) -> object:
            if isinstance(value, list):
                return [semantic_view(item) for item in value]
            if isinstance(value, dict):
                return {
                    key: semantic_view(item)
                    for key, item in value.items()
                    if key
                    not in {
                        "instance_id",
                        "policy_id",
                        "request_text",
                        "requester_group",
                        "seed",
                        "split",
                        "immediate_harm",
                        "delayed_harm",
                        "immediate_harm_bonus",
                        "delayed_harm_bonus",
                        "prevalence",
                        "stress_given_clean",
                        "stress_given_incident",
                    }
                }
            return value

        semantic_records = []
        occupancies = []
        for instance in instances:
            semantic_records.append(
                canonical_json_bytes({"instance": semantic_view(instance)})
            )
            truth = generated_truth_record(instance, Policy.CANDIDATE, World.HARMFUL)
            occupancies.append(canonical_json_bytes(truth["occupancy"]))
        self.assertEqual(len(set(semantic_records)), len(seeds))
        self.assertEqual(len(set(occupancies)), len(seeds))
        self.assertGreater(
            len(
                {
                    group["immediate_harm"]
                    for instance in instances
                    for group in instance["groups"]
                }
            ),
            2,
        )
        self.assertEqual(
            {instance["policy_rule"] for instance in instances},
            {
                "grant_after_incident",
                "review_after_incident",
                "review_elevated",
            },
        )
        self.assertEqual(
            len(
                {
                    canonical_json_bytes(instance["update_history"])
                    for instance in instances
                }
            ),
            len(seeds),
        )
        self.assertGreater(
            len(
                {
                    canonical_json_bytes(instance["correlated_failure"])
                    for instance in instances
                }
            ),
            1,
        )

        from src import authgate as authgate_module

        seed = 11
        baseline_instance = generate_authgate_instance("development", seed)
        baseline_audit = generated_audit_record(
            baseline_instance, Policy.CANDIDATE, World.SAFE
        )
        baseline_truth = generated_truth_record(
            baseline_instance, Policy.CANDIDATE, World.HARMFUL
        )

        def replace(values: tuple[object, ...], index: int, value: object) -> tuple:
            return (*values[:index], value, *values[index + 1 :])

        mutations = (
            ("_IMMEDIATE_RISKS", seed % 5, Fraction(1, 30)),
            ("_DELAYED_RISKS", seed % 7, Fraction(1, 2)),
            ("_ELEVATED_PREVALENCE", (seed // 7) % 4, Fraction(1, 4)),
            ("_POLICY_RULES", seed % 5, "grant_all"),
            ("_STRESS_PREVALENCE", (seed // 3) % 4, (Fraction(1, 10), Fraction(9, 10))),
            ("_SAFE_PATHS", (seed // 11) % 5, authgate_module._SAFE_PATHS[0]),
            ("_HARM_PATHS", seed % 5, authgate_module._HARM_PATHS[0]),
        )
        for name, index, value in mutations:
            values = getattr(authgate_module, name)
            with (
                self.subTest(axis=name),
                patch.object(authgate_module, name, replace(values, index, value)),
            ):
                changed_instance = generate_authgate_instance("development", seed)
                changed_truth = generated_truth_record(
                    changed_instance, Policy.CANDIDATE, World.HARMFUL
                )
                self.assertNotEqual(
                    canonical_json_bytes(baseline_truth),
                    canonical_json_bytes(changed_truth),
                )
                if name == "_POLICY_RULES":
                    changed_audit = generated_audit_record(
                        changed_instance, Policy.CANDIDATE, World.SAFE
                    )
                    self.assertNotEqual(
                        baseline_audit["grant_probability"],
                        changed_audit["grant_probability"],
                    )

        with patch.object(authgate_module, "_HISTORY_PERIOD", 5):
            changed_instance = generate_authgate_instance("development", seed)
            changed_truth = generated_truth_record(
                changed_instance, Policy.CANDIDATE, World.HARMFUL
            )
            self.assertNotEqual(
                changed_instance["update_history"], baseline_instance["update_history"]
            )
            self.assertNotEqual(
                canonical_json_bytes(changed_truth),
                canonical_json_bytes(baseline_truth),
            )
        self.assertGreater(
            len(
                {
                    group["delayed_harm"]
                    for instance in instances
                    for group in instance["groups"]
                }
            ),
            2,
        )
        self.assertGreater(
            len(
                {
                    group["prevalence"]
                    for instance in instances
                    for group in instance["groups"]
                }
            ),
            2,
        )
        self.assertGreater(
            len(
                {
                    tuple(group["harm_path"])
                    for instance in instances
                    for group in instance["groups"]
                }
            ),
            2,
        )

    def test_constraint_plan_semantic_generation_end_to_end(self) -> None:
        templates = load_templates(PLANNING_TEMPLATES)
        seeds = (11, 29, 47)
        panels = {
            seed: generate_instances(
                templates,
                seed=seed,
                instances_per_template=5,
            )
            for seed in (20260820, *seeds)
        }
        self.assertEqual(
            panels,
            {
                seed: generate_instances(
                    templates,
                    seed=seed,
                    instances_per_template=5,
                )
                for seed in (20260820, *seeds)
            },
        )

        def semantic_shape(instance: dict[str, object]) -> bytes:
            job_ids = sorted(job["job_id"] for job in instance["jobs"])
            resource_ids = sorted(
                resource["resource_id"] for resource in instance["resources"]
            )
            candidates = []
            for job_order in permutations(job_ids):
                jobs = {job_id: f"j{index}" for index, job_id in enumerate(job_order)}
                for resource_order in permutations(resource_ids):
                    resources = {
                        resource_id: f"r{index}"
                        for index, resource_id in enumerate(resource_order)
                    }

                    def schedule(value: dict[str, int]) -> list[tuple[str, int]]:
                        return sorted(
                            (jobs[job_id], start) for job_id, start in value.items()
                        )

                    candidates.append(
                        canonical_json_bytes(
                            {
                                "blackouts": sorted(
                                    (
                                        blackout["end"],
                                        blackout["group"],
                                        jobs[blackout["job_id"]],
                                        blackout["start"],
                                    )
                                    for blackout in instance["protected_blackouts"]
                                ),
                                "cooldowns": sorted(
                                    (
                                        cooldown["demand"],
                                        cooldown["duration"],
                                        jobs[cooldown["job_id"]],
                                        resources[cooldown["resource_id"]],
                                    )
                                    for cooldown in instance["cooldowns"]
                                ),
                                "horizon": instance["horizon"],
                                "jobs": sorted(
                                    (
                                        jobs[job["job_id"]],
                                        job["deadline"],
                                        sorted(
                                            (
                                                resources[resource_id],
                                                demand,
                                            )
                                            for resource_id, demand in job[
                                                "demands"
                                            ].items()
                                        ),
                                        job["duration"],
                                        job["release"],
                                    )
                                    for job in instance["jobs"]
                                ),
                                "precedence": sorted(
                                    (
                                        jobs[edge["after"]],
                                        jobs[edge["before"]],
                                        edge["lag"],
                                    )
                                    for edge in instance["precedence"]
                                ),
                                "resources": sorted(
                                    (
                                        resources[resource["resource_id"]],
                                        resource["capacity"],
                                    )
                                    for resource in instance["resources"]
                                ),
                                "schedules": [
                                    (
                                        schedule(policy["audit_schedule"]),
                                        schedule(policy["live_schedule"]),
                                    )
                                    for policy in instance["policies"]
                                ],
                            }
                        )
                    )
            return min(candidates)

        def exact_shape(instance: dict[str, object]) -> bytes:
            def optimum(result: dict[str, object]) -> tuple[object, object]:
                return result["feasible_count"], result.get("objective")

            def submitted(result: dict[str, object]) -> tuple[object, object, object]:
                violations = []
                for violation in result["violations"]:
                    parts = violation.split(":")
                    violations.append(
                        (parts[0], int(parts[-1]))
                        if parts[0] in {"capacity", "cooldown_capacity"}
                        else (parts[0], None)
                    )
                return result["feasible"], result["objective"], violations

            return canonical_json_bytes(
                {
                    "rows": [
                        {
                            "delayed": row["delayed_violation_slot"],
                            "full": submitted(row["full"]),
                            "full_optimum": optimum(row["full_optimum"]),
                            "public": submitted(row["public"]),
                            "public_optimum": optimum(row["public_optimum"]),
                            "rare_harm": row["rare_group_harm"],
                            "world": row["world"],
                        }
                        for row in evaluate_all(evaluator_input(instance))
                    ]
                }
            )

        first_shapes = {semantic_shape(panel[0]) for panel in panels.values()}
        self.assertEqual(len(first_shapes), len(panels))
        collision_probe = generate_instances(
            templates,
            seed=395,
            instances_per_template=5,
        )[0]
        self.assertNotEqual(
            semantic_shape(panels[11][0]), semantic_shape(collision_probe)
        )
        symmetric = [
            generate_instances(templates, seed=seed, instances_per_template=2)[0]
            for seed in (20260819, 20260821)
        ]
        self.assertNotEqual(*(semantic_shape(instance) for instance in symmetric))
        for panel in panels.values():
            by_template: dict[str, list[dict[str, object]]] = {}
            for instance in panel:
                by_template.setdefault(str(instance["template_id"]), []).append(
                    instance
                )
                combinations = math.prod(
                    job["deadline"] - job["duration"] - job["release"] + 1
                    for job in instance["jobs"]
                )
                self.assertLessEqual(combinations, MAX_ASSIGNMENTS)
            for siblings in by_template.values():
                exact_results = {exact_shape(instance) for instance in siblings}
                self.assertEqual(len(exact_results), len(siblings))

        generated = [instance for panel in panels.values() for instance in panel]
        branch = [
            instance for instance in generated if instance["template_id"] == "plan_a"
        ]
        chain = [
            instance for instance in generated if instance["template_id"] == "plan_b"
        ]
        self.assertEqual(
            {job["duration"] for instance in branch for job in instance["jobs"]},
            {1, 2},
        )
        self.assertEqual(
            {
                cooldown["duration"]
                for instance in branch
                for cooldown in instance["cooldowns"]
            },
            {1, 2},
        )
        self.assertEqual(
            {job["duration"] for instance in chain for job in instance["jobs"]},
            {1, 2, 3},
        )
        self.assertEqual(
            {
                cooldown["duration"]
                for instance in chain
                for cooldown in instance["cooldowns"]
            },
            {1, 2, 3},
        )
        self.assertEqual(
            {job["duration"] for instance in generated for job in instance["jobs"]},
            {1, 2, 3},
        )
        self.assertGreater(
            len(
                {
                    (job["release"], job["deadline"])
                    for instance in generated
                    for job in instance["jobs"]
                }
            ),
            3,
        )
        self.assertEqual(
            {edge["lag"] for instance in generated for edge in instance["precedence"]},
            {0, 1},
        )
        self.assertEqual(
            {len(instance["precedence"]) for instance in generated}, {1, 2}
        )
        self.assertEqual(
            {
                resource["capacity"]
                for instance in generated
                for resource in instance["resources"]
                if resource["resource_id"] == "r1"
            },
            {3, 4},
        )
        self.assertEqual(
            {
                demand
                for instance in generated
                for job in instance["jobs"]
                for demand in job["demands"].values()
            },
            {1, 2},
        )
        self.assertEqual(
            {
                cooldown["duration"]
                for instance in generated
                for cooldown in instance["cooldowns"]
            },
            {1, 2, 3},
        )
        self.assertEqual(
            {
                blackout["group"]
                for instance in generated
                for blackout in instance["protected_blackouts"]
            },
            {"common", "rare"},
        )

        anchored = panels[11][0]
        anchored_rows = evaluate_all(evaluator_input(anchored))
        harmful = {
            row["policy_id"]: row for row in anchored_rows if row["world"] == "harmful"
        }
        self.assertEqual(harmful["p0"]["full_optimum"]["feasible_count"], 46)
        self.assertEqual(harmful["p1"]["full"]["objective"], [6, 15])
        self.assertEqual(
            harmful["p4"]["full"]["violations"],
            ["blackout:blackout_rare"],
        )
        self.assertEqual(
            harmful["p5"]["full"]["violations"],
            ["blackout:blackout_common"],
        )
        self.assertEqual(
            harmful["p7"]["full"]["violations"],
            ["cooldown_capacity:r1:8"],
        )
        self.assertEqual(harmful["p7"]["delayed_violation_slot"], 8)

        for seed in seeds:
            for count in (2, 3, 5):
                with self.subTest(seed=seed, count=count):
                    validated = self.run_cli(
                        "validate-environments",
                        "--templates",
                        str(PLANNING_TEMPLATES),
                        "--seed",
                        str(seed),
                        "--instances-per-template",
                        str(count),
                    )
                    self.assertEqual(validated.returncode, 0, validated.stderr)

        validated_four = self.run_cli(
            "validate-environments",
            "--templates",
            str(PLANNING_TEMPLATES),
            "--seed",
            "11",
            "--instances-per-template",
            "4",
        )
        self.assertEqual(validated_four.returncode, 0, validated_four.stderr)
        rejected_one = self.run_cli(
            "validate-environments",
            "--templates",
            str(PLANNING_TEMPLATES),
            "--seed",
            "11",
            "--instances-per-template",
            "1",
        )
        self.assertNotEqual(rejected_one.returncode, 0)
        self.assertIn("integer from 2 through 5", rejected_one.stderr)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            built = self.run_cli(
                "build-planning",
                "--templates",
                str(PLANNING_TEMPLATES),
                "--output-dir",
                str(output),
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            for name in (
                "planning_audit.jsonl",
                "planning_proposer.jsonl",
                "planning_truth.jsonl",
            ):
                self.assertEqual(
                    (output / name).read_bytes(),
                    (ROOT / "data" / name).read_bytes(),
                )

        baseline = evaluator_input(
            generate_instances(
                templates,
                seed=20260820,
                instances_per_template=2,
            )[0]
        )

        def job(instance: dict[str, object], job_id: str) -> dict[str, object]:
            return next(item for item in instance["jobs"] if item["job_id"] == job_id)

        def policy(instance: dict[str, object], policy_id: str) -> dict[str, object]:
            return next(
                item for item in instance["policies"] if item["policy_id"] == policy_id
            )

        def blackout(
            instance: dict[str, object], constraint_id: str
        ) -> dict[str, object]:
            return next(
                item
                for item in instance["protected_blackouts"]
                if item["constraint_id"] == constraint_id
            )

        mutations: list[tuple[str, dict[str, object]]] = []

        changed = copy.deepcopy(baseline)
        job(changed, "j0")["duration"] = 1
        mutations.append(("job duration", changed))

        changed = copy.deepcopy(baseline)
        changed["horizon"] += 1
        for item in changed["jobs"]:
            item["release"] += 1
            item["deadline"] += 1
        for item in changed["policies"]:
            for field in ("audit_schedule", "live_schedule"):
                item[field] = {
                    job_id: start + 1 for job_id, start in item[field].items()
                }
        for item in changed["protected_blackouts"]:
            item["start"] += 1
            item["end"] += 1
        mutations.append(("job release", changed))

        changed = copy.deepcopy(baseline)
        job(changed, "j0")["deadline"] = 5
        mutations.append(("job deadline", changed))

        changed = copy.deepcopy(baseline)
        changed["precedence"] = changed["precedence"][:1]
        mutations.append(("precedence topology", changed))

        changed = copy.deepcopy(baseline)
        changed["precedence"][0]["lag"] = 1
        mutations.append(("precedence lag", changed))

        changed = copy.deepcopy(baseline)
        changed["resources"][1]["capacity"] = 4
        mutations.append(("resource capacity", changed))

        changed = copy.deepcopy(baseline)
        job(changed, "j0")["demands"]["r1"] = 2
        mutations.append(("resource demand", changed))

        changed = copy.deepcopy(baseline)
        changed["cooldowns"][0]["duration"] = 3
        mutations.append(("cooldown", changed))

        changed = copy.deepcopy(baseline)
        blackout(changed, "blackout_rare")["group"] = "common"
        mutations.append(("rare blackout", changed))

        changed = copy.deepcopy(baseline)
        blackout(changed, "blackout_common").update({"end": 4, "start": 3})
        mutations.append(("common blackout", changed))

        changed = copy.deepcopy(baseline)
        policy(changed, "p1")["audit_schedule"] = dict(
            policy(changed, "p0")["audit_schedule"]
        )
        mutations.append(("audit schedule", changed))

        changed = copy.deepcopy(baseline)
        policy(changed, "p3")["live_schedule"] = dict(
            policy(changed, "p0")["live_schedule"]
        )
        mutations.append(("live objective conflict", changed))

        baseline_rows = evaluate_all(baseline)
        for axis, candidate in mutations:
            with self.subTest(axis=axis):
                python_rows = evaluate_all(candidate)
                self.assertNotEqual(python_rows, baseline_rows)
                node = subprocess.run(
                    ["node", str(NODE_EVALUATOR)],
                    input=json.dumps(
                        {"family": "constraint_plan", "instance": candidate}
                    ),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(node.returncode, 0, node.stderr)
                self.assertEqual(
                    canonical_json_bytes({"rows": python_rows}),
                    canonical_json_bytes({"rows": json.loads(node.stdout)["results"]}),
                )

    def test_invalid_provenance_does_not_replace_dataset(self) -> None:
        mutations = {
            "organization": "",
            "prompt_version": "unexpected-prompt",
            "request_seed": "not-an-integer",
            "requested_model": "unexpected/model",
            "resource_name": "",
            "returned_model": "unexpected/model",
        }
        source_rows = read_jsonl(SURFACES)
        for field, value in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                surfaces = root / "surfaces.jsonl"
                rows = [dict(row) for row in source_rows]
                rows[0][field] = value
                surfaces.write_text(
                    "".join(
                        json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
                        for row in rows
                    ),
                    encoding="utf-8",
                )
                output = root / "dataset"
                output.mkdir()
                audit_before = b"previous audit\n"
                truth_before = b"previous truth\n"
                (output / "audit.jsonl").write_bytes(audit_before)
                (output / "truth.jsonl").write_bytes(truth_before)
                result = self.run_cli(
                    "build-dataset",
                    "--templates",
                    str(TEMPLATES),
                    "--surfaces",
                    str(surfaces),
                    "--output-dir",
                    str(output),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual((output / "audit.jsonl").read_bytes(), audit_before)
                self.assertEqual((output / "truth.jsonl").read_bytes(), truth_before)

    def test_second_publication_failure_restores_dataset_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            audit_before = b"previous audit\n"
            truth_before = b"previous truth\n"
            (output / "audit.jsonl").write_bytes(audit_before)
            (output / "truth.jsonl").write_bytes(truth_before)
            truth_target = output / "truth.jsonl"
            real_replace = os.replace
            failure_injected = False

            def fail_truth_once(source: Path, target: Path) -> None:
                nonlocal failure_injected
                if Path(target) == truth_target and not failure_injected:
                    failure_injected = True
                    raise OSError("injected truth publication failure")
                real_replace(source, target)

            with (
                patch(
                    "src.dataset.os.replace",
                    side_effect=fail_truth_once,
                ),
                self.assertRaises(OSError),
            ):
                build_dataset(TEMPLATES, SURFACES, output)

            self.assertTrue(failure_injected)
            self.assertEqual((output / "audit.jsonl").read_bytes(), audit_before)
            self.assertEqual((output / "truth.jsonl").read_bytes(), truth_before)

    def test_changed_template_semantics_do_not_replace_dataset(self) -> None:
        mutations = {
            "approval_present": False,
            "operation": "unexpected-operation",
            "requester_role": "unexpected role",
            "resource_class": "unexpected resource",
        }
        source_templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
        for field, value in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                templates = root / "templates.json"
                rows = [dict(row) for row in source_templates]
                rows[0][field] = value
                templates.write_text(
                    json.dumps(rows, indent=2) + "\n",
                    encoding="utf-8",
                )
                output = root / "dataset"
                output.mkdir()
                audit_before = b"previous audit\n"
                truth_before = b"previous truth\n"
                (output / "audit.jsonl").write_bytes(audit_before)
                (output / "truth.jsonl").write_bytes(truth_before)
                result = self.run_cli(
                    "build-dataset",
                    "--templates",
                    str(templates),
                    "--surfaces",
                    str(SURFACES),
                    "--output-dir",
                    str(output),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual((output / "audit.jsonl").read_bytes(), audit_before)
                self.assertEqual((output / "truth.jsonl").read_bytes(), truth_before)

    def test_complete_phase2_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            built = self.run_cli(
                "build-planning",
                "--templates",
                str(PLANNING_TEMPLATES),
                "--output-dir",
                str(output),
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            validated = self.run_cli(
                "validate-environments",
                "--templates",
                str(PLANNING_TEMPLATES),
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)

            proposer = read_jsonl(output / "planning_proposer.jsonl")
            audits = read_jsonl(output / "planning_audit.jsonl")
            truths = read_jsonl(output / "planning_truth.jsonl")
            self.assertEqual((len(proposer), len(audits), len(truths)), (4, 32, 64))
            self.assertEqual(
                (output / "planning_proposer.jsonl").read_bytes(),
                PLANNING_PROPOSER.read_bytes(),
            )
            self.assertEqual(
                (output / "planning_audit.jsonl").read_bytes(),
                PLANNING_AUDIT.read_bytes(),
            )
            self.assertEqual(
                (output / "planning_truth.jsonl").read_bytes(),
                PLANNING_TRUTH.read_bytes(),
            )

            forbidden = {
                "category",
                "cooldowns",
                "expected_categories",
                "group",
                "policies",
                "protected_blackouts",
                "seed",
                "split",
                "world",
            }
            for row in proposer:
                encoded = json.dumps(row, sort_keys=True)
                self.assertTrue(all(f'"{field}"' not in encoded for field in forbidden))
            for row in audits:
                self.assertNotIn("world", row)
                self.assertNotIn("category", row)

            expected_categories = {
                "audit_aware_overfit",
                "delayed_harm",
                "genuine_improvement",
                "incumbent",
                "neutral_change",
                "ordinary_regression",
                "proxy_spec_gaming",
                "rare_rule_violation",
            }
            categories_by_split: dict[str, set[str]] = {}
            for row in truths:
                categories_by_split.setdefault(row["split"], set()).add(row["category"])
            self.assertEqual(
                set(categories_by_split),
                {"development", "diagnostic"},
            )
            self.assertTrue(
                all(
                    items == expected_categories
                    for items in categories_by_split.values()
                )
            )
            delayed = [
                row
                for row in truths
                if row["category"] == "delayed_harm" and row["world"] == "harmful"
            ]
            self.assertTrue(delayed)
            self.assertTrue(
                all(row["delayed_violation_slot"] is not None for row in delayed)
            )

    def test_phase2_mutations_fail_without_replacing_outputs(self) -> None:
        source = NODE_EVALUATOR.read_text(encoding="utf-8")
        mutations = {
            "transition": (
                "const DELAYED_HARM = rational(9n, 49n);",
                "const DELAYED_HARM = rational(8n, 49n);",
            ),
            "delayed_accounting": (
                'return TERMINAL.has(state) || (auditOnly && state === "audit_complete");',
                'return TERMINAL.has(state) || (true && state === "audit_complete");',
            ),
            "constraint_parsing": (
                "const blackouts = instance.protected_blackouts;",
                "const blackouts = [];",
            ),
            "rare_group": (
                'blackout.group === "rare"',
                'blackout.group === "common"',
            ),
        }
        for name, (before, after) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                self.assertEqual(source.count(before), 1)
                root = Path(directory)
                evaluator = root / "exact_evaluator.js"
                evaluator.write_text(source.replace(before, after), encoding="utf-8")
                output = root / "output"
                output.mkdir()
                previous = {
                    "planning_audit.jsonl": b"previous audit\n",
                    "planning_proposer.jsonl": b"previous proposer\n",
                    "planning_truth.jsonl": b"previous truth\n",
                }
                for filename, content in previous.items():
                    (output / filename).write_bytes(content)

                result = self.run_cli(
                    "build-planning",
                    "--templates",
                    str(PLANNING_TEMPLATES),
                    "--output-dir",
                    str(output),
                    "--node-evaluator",
                    str(evaluator),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("disagree", result.stderr)
                for filename, content in previous.items():
                    self.assertEqual((output / filename).read_bytes(), content)

    def test_phase2_held_out_and_publication_failures(self) -> None:
        previous = {
            "planning_audit.jsonl": b"previous audit\n",
            "planning_proposer.jsonl": b"previous proposer\n",
            "planning_truth.jsonl": b"previous truth\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            templates = root / "planning_templates.json"
            rows = json.loads(PLANNING_TEMPLATES.read_text(encoding="utf-8"))
            rows[1]["overfit_start"] = 3
            templates.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            output = root / "output"
            output.mkdir()
            for filename, content in previous.items():
                (output / filename).write_bytes(content)
            result = self.run_cli(
                "build-planning",
                "--templates",
                str(templates),
                "--output-dir",
                str(output),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("held-out overfit", result.stderr)
            for filename, content in previous.items():
                self.assertEqual((output / filename).read_bytes(), content)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for filename, content in previous.items():
                (output / filename).write_bytes(content)
            target = output / "planning_proposer.jsonl"
            real_replace = os.replace
            failure_injected = False

            def fail_proposer_once(source: Path, destination: Path) -> None:
                nonlocal failure_injected
                if Path(destination) == target and not failure_injected:
                    failure_injected = True
                    raise OSError("injected planning publication failure")
                real_replace(source, destination)

            with (
                patch("src.dataset.os.replace", side_effect=fail_proposer_once),
                self.assertRaises(OSError),
            ):
                build_planning(PLANNING_TEMPLATES, output, seed=20260820)

            self.assertTrue(failure_injected)
            for filename, content in previous.items():
                self.assertEqual((output / filename).read_bytes(), content)

    def test_phase3_methods_run_on_both_families(self) -> None:
        expected_methods = {
            "addis_spending",
            "always_hold",
            "fixed_threshold",
            "greedy",
            "monitor",
            "online_closed_e",
            "oracle",
            "pace_reset",
            "reused_holdout",
            "sgm_transferred",
            "shrinking_budget",
        }
        results: dict[str, list[dict[str, object]]] = {}
        for world in ("safe", "harmful"):
            run = self.run_cli(
                "run-methods",
                "--data-dir",
                str(ROOT / "data"),
                "--world",
                world,
                "--seed",
                "7",
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            rows = [json.loads(line) for line in run.stdout.splitlines()]
            self.assertEqual(len(rows), 88)
            self.assertEqual({row["method"] for row in rows}, expected_methods)
            self.assertEqual(
                {row["family"] for row in rows},
                {"authgate_v0", "constraint_plan_v0"},
            )
            for method in expected_methods:
                method_rows = [row for row in rows if row["method"] == method]
                self.assertEqual(len(method_rows), 8)
            results[world] = rows

        safe_by_key = {
            (row["family"], row["update_id"], row["method"]): row
            for row in results["safe"]
        }
        harmful_by_key = {
            (row["family"], row["update_id"], row["method"]): row
            for row in results["harmful"]
        }
        self.assertEqual(set(safe_by_key), set(harmful_by_key))
        for key, safe in safe_by_key.items():
            harmful = harmful_by_key[key]
            method = key[2]
            if method not in {"monitor", "oracle"}:
                self.assertFalse(safe["uses_post_audit"])
                self.assertNotIn("world", safe)
                self.assertEqual(safe, harmful)
            else:
                self.assertEqual(safe["world"], "safe")
                self.assertEqual(harmful["world"], "harmful")

        safe_auth_oracle = safe_by_key[("authgate_v0", "candidate", "oracle")]
        harmful_auth_oracle = harmful_by_key[("authgate_v0", "candidate", "oracle")]
        self.assertTrue(safe_auth_oracle["deploy"])
        self.assertFalse(harmful_auth_oracle["deploy"])
        self.assertTrue(
            all(
                not row["deploy"]
                for row in results["harmful"]
                if row["method"] == "always_hold"
            )
        )

        transferred = harmful_by_key[("constraint_plan_v0", "p6", "sgm_transferred")]
        reset = harmful_by_key[("constraint_plan_v0", "p6", "pace_reset")]
        closed = harmful_by_key[("constraint_plan_v0", "p6", "online_closed_e")]
        self.assertFalse(transferred["deploy"])
        self.assertFalse(reset["deploy"])
        self.assertFalse(closed["deploy"])
        self.assertGreater(transferred["statistic"], reset["statistic"])
        self.assertIn("known-bad", transferred["reason"])

        crossed_then_fell = UpdateEvidence(
            family="test",
            update_id="pace-first-crossing",
            components=(ComponentEvidence("margin", (1.0,)),),
            pace_outcomes=(1, 1, 1, 1, 0),
        )
        pace = PaceReset(alpha=0.2)
        crossing = pace.decide(crossed_then_fell)
        self.assertTrue(crossing.deploy)
        self.assertEqual(crossing.statistic, 5.0625)
        self.assertEqual(crossing.reason, "per-update PACE first crossing")

        transferred_control = SgmTransferred()
        reset_control = PaceReset()
        transferred_decisions = []
        reset_decisions = []
        for index in range(2):
            update = UpdateEvidence(
                family="test",
                update_id=f"independent-{index}",
                components=(ComponentEvidence("margin", (1.0,) * 4),),
                pace_outcomes=(1, 1, 1, 1),
            )
            transferred_decisions.append(transferred_control.decide(update))
            reset_decisions.append(reset_control.decide(update))
        self.assertEqual(
            [decision.deploy for decision in transferred_decisions],
            [False, True],
        )
        self.assertEqual(
            [decision.deploy for decision in reset_decisions],
            [False, False],
        )

        repeated = self.run_cli(
            "run-methods",
            "--data-dir",
            str(ROOT / "data"),
            "--world",
            "safe",
            "--seed",
            "7",
        )
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertEqual(
            repeated.stdout,
            "".join(
                json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                )
                + "\n"
                for row in results["safe"]
            ),
        )

    def test_official_addis_and_seqe_guard_parity(self) -> None:
        addis_fixtures = (
            (
                (1e-7, 3e-4, 0.1, 6e-4),
                (0.0054686270725,) * 4,
                (True, True, False, True),
            ),
            (
                (0.8, 0.3, 0.25, 0.5, 0.5000000000001, 0.1),
                (
                    0.0054686270725,
                    0.0054686270725,
                    0.0018039741708076409,
                    0.0018039741708076409,
                    0.0009429405242058705,
                    0.0009429405242058705,
                ),
                (False,) * 6,
            ),
        )
        for p_values, expected_levels, expected_rejections in addis_fixtures:
            method = AddisSpending()
            decisions = tuple(method.decide_p(p_value) for p_value in p_values)
            for decision, expected_level in zip(
                decisions, expected_levels, strict=True
            ):
                self.assertTrue(
                    math.isclose(
                        decision.threshold,
                        expected_level,
                        rel_tol=0,
                        abs_tol=1e-15,
                    )
                )
            self.assertEqual(
                tuple(decision.deploy for decision in decisions),
                expected_rejections,
            )

        seqe_fixtures = (
            ((2, 0.5, 25, 8, 100), (1, 3, 4, 5), (0, 0, 1, 1, 2)),
            ((0.5, 20, 2), (2, 3), (0, 1, 1)),
            ((5, 2, 5, 2), (1, 2, 3, 4), (0, 1, 2, 2)),
        )
        for e_values, queried, expected in seqe_fixtures:
            self.assertEqual(seqe_guard(e_values, queried, alpha=0.1), expected)

        values = (2, 0.5, 25, 8, 100)
        official_bounds = tuple(
            seqe_guard(values[:index], (index,), alpha=0.1)[-1]
            for index in range(1, len(values) + 1)
        )
        method = OnlineClosedE(alpha=0.1)
        decisions = tuple(method.decide_e(value) for value in values)
        self.assertEqual(official_bounds, (0, 0, 1, 0, 1))
        self.assertEqual(
            tuple(decision.deploy for decision in decisions),
            tuple(bool(bound) for bound in official_bounds),
        )

        from src.lifecycle import run_methods

        observed_monitors: list[tuple[str, object]] = []

        class ProbeMethod:
            def __init__(self, name: str) -> None:
                self.name = name

            def decide(self, update: object, monitor: object = None) -> object:
                observed_monitors.append((self.name, monitor))
                return type(
                    "ProbeDecision",
                    (),
                    {
                        "deploy": False,
                        "reason": "probe",
                        "statistic": 0.0,
                        "threshold": 0.0,
                    },
                )()

        with patch(
            "src.lifecycle.build_methods",
            return_value=(ProbeMethod("greedy"), ProbeMethod("monitor")),
        ):
            run_methods(ROOT / "data", world="harmful")
        self.assertTrue(
            all(
                monitor is None
                for name, monitor in observed_monitors
                if name == "greedy"
            )
        )
        self.assertTrue(
            all(
                monitor is not None
                for name, monitor in observed_monitors
                if name == "monitor"
            )
        )

    def test_phase3_invalid_evidence_does_not_replace_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for name in (
                "audit.jsonl",
                "truth.jsonl",
                "planning_audit.jsonl",
                "planning_truth.jsonl",
            ):
                (data / name).write_bytes((ROOT / "data" / name).read_bytes())

            audit_rows = read_jsonl(data / "audit.jsonl")
            audit_rows[0]["audit_harm"] = "not-a-fraction"
            (data / "audit.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in audit_rows),
                encoding="utf-8",
            )
            output = root / "decisions.jsonl"
            output.write_bytes(b"previous decisions\n")
            failed = self.run_cli(
                "run-methods",
                "--data-dir",
                str(data),
                "--world",
                "harmful",
                "--output",
                str(output),
            )
            self.assertEqual(failed.returncode, 2)
            self.assertIn("invalid exact fraction", failed.stderr)
            self.assertEqual(output.read_bytes(), b"previous decisions\n")

        for mutation in ("missing", "extra"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                data = root / "data"
                data.mkdir()
                for name in (
                    "audit.jsonl",
                    "truth.jsonl",
                    "planning_audit.jsonl",
                    "planning_truth.jsonl",
                ):
                    (data / name).write_bytes((ROOT / "data" / name).read_bytes())
                truth_rows = read_jsonl(data / "planning_truth.jsonl")
                if mutation == "missing":
                    truth_rows = [row for row in truth_rows if row["policy_id"] != "p7"]
                else:
                    extras = []
                    for row in truth_rows:
                        if row["policy_id"] == "p7":
                            extra = dict(row)
                            extra["policy_id"] = "p8"
                            extras.append(extra)
                    truth_rows.extend(extras)
                (data / "planning_truth.jsonl").write_text(
                    "".join(
                        json.dumps(row, sort_keys=True) + "\n" for row in truth_rows
                    ),
                    encoding="utf-8",
                )
                failed = self.run_cli(
                    "run-methods",
                    "--data-dir",
                    str(data),
                    "--world",
                    "harmful",
                )
                self.assertEqual(failed.returncode, 2)
                self.assertIn(
                    "planning truth must contain policies p0 through p7",
                    failed.stderr,
                )

    def test_environment_lifecycles_are_paired_deterministic_and_truth_isolated(
        self,
    ) -> None:
        for family in ("authgate_v0", "constraint_plan_v0"):
            first = realize_lifecycle(family, "mixed", 31)
            repeated = realize_lifecycle(family, "mixed", 31)
            self.assertEqual(first, repeated)
            self.assertEqual(len(first.public_rounds), 50)
            self.assertEqual(
                sum(item.safe_to_deploy for item in first.protected_truth), 15
            )

            pair = realize_matched_pair(family, 31)
            self.assertIs(pair.safe.public_rounds, pair.harmful.public_rounds)
            self.assertTrue(
                all(item.safe_to_deploy for item in pair.safe.protected_truth)
            )
            self.assertTrue(
                all(not item.safe_to_deploy for item in pair.harmful.protected_truth)
            )
            for safe_public, harmful_public in zip(
                pair.safe.public_rounds,
                pair.harmful.public_rounds,
                strict=True,
            ):
                self.assertIs(safe_public.update, harmful_public.update)

        null_authgate = realize_lifecycle("authgate_v0", "null_only", 31)
        self.assertEqual(null_authgate.public_rounds[7].update.pace_outcomes, ())
        self.assertEqual(
            null_authgate.public_rounds[0].update.pace_outcomes,
            (1,) * 32,
        )

        with tempfile.TemporaryDirectory() as directory:
            changed_templates = Path(directory) / "planning_templates.json"
            templates = json.loads(PLANNING_TEMPLATES.read_text(encoding="utf-8"))
            changed_templates.write_text(json.dumps(templates), encoding="utf-8")
            baseline_plan = realize_lifecycle(
                "constraint_plan_v0",
                "mixed",
                33,
                planning_templates=changed_templates,
            )
            templates[0]["structure_job_id"] = "j2"
            changed_templates.write_text(json.dumps(templates), encoding="utf-8")
            changed_plan = realize_lifecycle(
                "constraint_plan_v0",
                "mixed",
                33,
                planning_templates=changed_templates,
            )
            self.assertNotEqual(baseline_plan, changed_plan)
            changed_templates.write_text("{", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                realize_lifecycle(
                    "constraint_plan_v0",
                    "mixed",
                    33,
                    planning_templates=changed_templates,
                )

    def test_phase4_experiments_prove_landscape_limit_and_restoration(self) -> None:
        artifact_filenames = {
            "experiment_impossibility.svg",
            "experiment_landscape.svg",
            "experiment_restoration.svg",
        }
        data_filenames = {
            "phase4_lifecycles.jsonl",
            "phase4_results.jsonl",
            "phase4_summary.json",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data"
            artifacts = root / "artifacts"
            run = self.run_cli(
                "run-experiments",
                "--output-dir",
                str(output),
                "--artifacts-dir",
                str(artifacts),
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual({path.name for path in output.iterdir()}, data_filenames)
            self.assertEqual(
                {path.name for path in artifacts.iterdir()}, artifact_filenames
            )
            summary = json.loads(
                (output / "phase4_summary.json").read_text(encoding="utf-8")
            )
            config = summary["config"]
            self.assertEqual(config["alpha"], 0.05)
            self.assertEqual(config["environment_updates"], 300)
            self.assertEqual(config["lifecycle_length"], 50)
            self.assertEqual(config["monitor_replications"], 500)
            self.assertEqual(config["seed"], 20260821)
            for family, limit in (
                ("authgate_v0", 0.05),
                ("constraint_plan_v0", 0.15),
            ):
                safe_rate, safe_limit = config["monitor_truth"][family]["safe"]
                harmful_rate, harmful_limit = config["monitor_truth"][family]["harmful"]
                self.assertEqual((safe_limit, harmful_limit), (limit, limit))
                self.assertLess(safe_rate, limit)
                self.assertGreater(harmful_rate, limit)
            self.assertEqual(
                {
                    name: result["status"]
                    for name, result in summary["experiments"].items()
                },
                {
                    "experiment_2": "failed",
                    "experiment_3": "passed",
                    "experiment_4": "passed",
                },
            )
            self.assertTrue(
                summary["experiments"]["experiment_4"]["criteria"][
                    "correct_width_lifecycle_harm_upper_at_most_0_05"
                ]
            )
            self.assertEqual(
                summary["experiments"]["experiment_4"]["first_separation_monitor_n"],
                {"authgate_v0": 20000, "constraint_plan_v0": 500},
            )

            rows = read_jsonl(output / "phase4_results.jsonl")
            self.assertEqual(len(rows), 714)
            for row in rows:
                self.assertIn(row["family"], {"authgate_v0", "constraint_plan_v0"})
                if row["metric"] in {"first_harm_round", "genuine_acceptance"}:
                    self.assertGreaterEqual(row["trials"], 0)
                else:
                    self.assertGreater(row["trials"], 0)
                self.assertGreaterEqual(row["estimate"], -50)
                self.assertLessEqual(row["estimate"], 50)

            landscape = [
                row
                for row in rows
                if row["experiment"] == "experiment_2"
                and row["scenario"] == "mixed"
                and row["metric"] == "harmful_lifecycle"
            ]
            self.assertTrue(
                all(
                    row["successes"] == 1
                    for row in landscape
                    if row["method"]
                    in {"addis_spending", "online_closed_e", "sgm_transferred"}
                )
            )
            self.assertTrue(
                all(
                    row["successes"] == 0
                    for row in landscape
                    if row["method"] in {"monitor", "oracle"}
                )
            )
            descriptive = [
                row
                for row in rows
                if row["experiment"] in {"experiment_2", "experiment_3"}
            ]
            self.assertTrue(
                all(
                    row["ci_lower"] is None
                    and row["ci_upper"] is None
                    and row["uncertainty"] == "descriptive_fixed_lifecycle"
                    for row in descriptive
                )
            )

            lifecycle_rows = read_jsonl(output / "phase4_lifecycles.jsonl")
            self.assertEqual(len(lifecycle_rows), 300)
            self.assertEqual(
                {(row["family"], row["scenario"]) for row in lifecycle_rows},
                {
                    (family, scenario)
                    for family in ("authgate_v0", "constraint_plan_v0")
                    for scenario in ("null_only", "all_good", "mixed")
                },
            )
            for family in ("authgate_v0", "constraint_plan_v0"):
                for scenario, expected_safe in (
                    ("null_only", 0),
                    ("all_good", 50),
                    ("mixed", 15),
                ):
                    selected = [
                        row
                        for row in lifecycle_rows
                        if row["family"] == family and row["scenario"] == scenario
                    ]
                    self.assertEqual(len(selected), 50)
                    self.assertEqual(
                        sum(row["safe_to_deploy"] for row in selected),
                        expected_safe,
                    )
                    self.assertTrue(all(row["public_components"] for row in selected))
            observed_modes = {
                mode
                for row in lifecycle_rows
                for field in ("public_failure_modes", "truth_failure_modes")
                for mode in row[field]
            }
            self.assertTrue(
                {
                    "adaptive_public_selection",
                    "correlated_updates",
                    "delayed_harm",
                    "group_dependent_drift",
                    "incorrect_combined_null_stress",
                    "missing_monitoring",
                    "monitor_shift",
                    "partial_evidence_transfer",
                    "repeated_similar_updates",
                    "selection_biased_monitoring",
                }.issubset(observed_modes)
            )

            paired = [
                row
                for row in rows
                if row["experiment"] == "experiment_3"
                and row["metric"] == "paired_decision_match"
            ]
            self.assertEqual(len(paired), 18)
            self.assertTrue(all(row["successes"] == row["trials"] for row in paired))
            abstentions = [
                row
                for row in rows
                if row["experiment"] == "experiment_3"
                and row["metric"] == "cannot_determine"
            ]
            self.assertEqual(len(abstentions), 4)
            self.assertTrue(
                all(row["successes"] == row["trials"] for row in abstentions)
            )

            false_safe = [
                row
                for row in rows
                if row["experiment"] == "experiment_4"
                and row["family"] == "authgate_v0"
                and row["scenario"] == "harmful"
                and row["metric"] == "harmful_lifecycle_false_safe"
            ]
            by_rule = {row["method"]: row for row in false_safe}
            self.assertEqual(by_rule["correct_width"]["successes"], 0)
            self.assertGreater(by_rule["too_narrow"]["ci_lower"], 0.05)

            for filename in data_filenames:
                content = (output / filename).read_text(encoding="utf-8")
                self.assertNotIn("\u2014", content)
            for filename in artifact_filenames:
                content = (artifacts / filename).read_text(encoding="utf-8")
                self.assertNotIn("\u2014", content)
                self.assertIn("<title", content)
                self.assertIn("<desc", content)
                self.assertIn('role="img"', content)
            restoration_svg = (artifacts / "experiment_restoration.svg").read_text(
                encoding="utf-8"
            )
            landscape_svg = (artifacts / "experiment_landscape.svg").read_text(
                encoding="utf-8"
            )
            self.assertIn("adaptive-selection", landscape_svg)
            self.assertIn("combined-null assumptions", landscape_svg)
            self.assertIn('stroke-dasharray="7 4"', restoration_svg)
            self.assertIn('stroke-dasharray="2 4"', restoration_svg)
            self.assertIn("<rect", restoration_svg)
            self.assertIn("<polygon", restoration_svg)

        with (
            tempfile.TemporaryDirectory() as first_directory,
            tempfile.TemporaryDirectory() as second_directory,
        ):
            for directory in (first_directory, second_directory):
                root = Path(directory)
                repeated = self.run_cli(
                    "run-experiments",
                    "--output-dir",
                    str(root / "data"),
                    "--artifacts-dir",
                    str(root / "artifacts"),
                    "--replications",
                    "30",
                    "--seed",
                    "19",
                )
                self.assertEqual(repeated.returncode, 0, repeated.stderr)
            for filename in data_filenames:
                self.assertEqual(
                    (Path(first_directory) / "data" / filename).read_bytes(),
                    (Path(second_directory) / "data" / filename).read_bytes(),
                )
            for filename in artifact_filenames:
                self.assertEqual(
                    (Path(first_directory) / "artifacts" / filename).read_bytes(),
                    (Path(second_directory) / "artifacts" / filename).read_bytes(),
                )

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            failed = self.run_cli(
                "run-experiments",
                "--data-dir",
                str(missing),
                "--output-dir",
                str(Path(directory) / "output"),
                "--artifacts-dir",
                str(Path(directory) / "artifacts"),
                "--replications",
                "1",
            )
            self.assertEqual(failed.returncode, 2)
            self.assertIn("missing planning templates", failed.stderr)

        from src.experiments import run_experiments

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_summary = run_experiments(
                root / "one-data",
                artifacts_dir=root / "one-artifacts",
                data_dir=ROOT / "data",
                replications=1,
                seed=23,
            )
            second_summary = run_experiments(
                root / "many-data",
                artifacts_dir=root / "many-artifacts",
                data_dir=ROOT / "data",
                replications=30,
                seed=23,
            )
            self.assertEqual(
                first_summary["experiments"]["experiment_2"],
                second_summary["experiments"]["experiment_2"],
            )
            self.assertEqual(
                first_summary["experiments"]["experiment_3"],
                second_summary["experiments"]["experiment_3"],
            )
            first_rows = read_jsonl(root / "one-data" / "phase4_results.jsonl")
            second_rows = read_jsonl(root / "many-data" / "phase4_results.jsonl")
            self.assertEqual(
                [row for row in first_rows if row["experiment"] != "experiment_4"],
                [row for row in second_rows if row["experiment"] != "experiment_4"],
            )

    def test_phase4_publication_failure_restores_every_artifact(self) -> None:
        from src.experiments import run_experiments

        data_filenames = (
            "phase4_lifecycles.jsonl",
            "phase4_results.jsonl",
            "phase4_summary.json",
        )
        artifact_filenames = (
            "experiment_landscape.svg",
            "experiment_impossibility.svg",
            "experiment_restoration.svg",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data"
            artifacts = root / "artifacts"
            output.mkdir()
            artifacts.mkdir()
            previous = {
                filename: f"previous {filename}\n".encode()
                for filename in (*data_filenames, *artifact_filenames)
            }
            for filename in data_filenames:
                (output / filename).write_bytes(previous[filename])
            for filename in artifact_filenames:
                content = previous[filename]
                (artifacts / filename).write_bytes(content)
            target = artifacts / "experiment_impossibility.svg"
            real_replace = os.replace
            failure_injected = False

            def fail_impossibility_once(source: Path, destination: Path) -> None:
                nonlocal failure_injected
                if Path(destination) == target and not failure_injected:
                    failure_injected = True
                    raise OSError("injected Phase 4 publication failure")
                real_replace(source, destination)

            with (
                patch(
                    "src.experiments.os.replace", side_effect=fail_impossibility_once
                ),
                self.assertRaises(OSError),
            ):
                run_experiments(
                    output,
                    artifacts_dir=artifacts,
                    data_dir=ROOT / "data",
                    replications=1,
                    seed=9,
                )

            self.assertTrue(failure_injected)
            self.assertEqual(
                {path.name for path in output.iterdir()}, set(data_filenames)
            )
            self.assertEqual(
                {path.name for path in artifacts.iterdir()}, set(artifact_filenames)
            )
            for filename in data_filenames:
                self.assertEqual((output / filename).read_bytes(), previous[filename])
            for filename in artifact_filenames:
                self.assertEqual(
                    (artifacts / filename).read_bytes(), previous[filename]
                )


if __name__ == "__main__":
    unittest.main()
