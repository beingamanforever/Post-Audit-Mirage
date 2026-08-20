from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

from post_audit_mirage.authgate import (
    HORIZON,
    Action,
    Case,
    Policy,
    State,
    World,
    audit_record,
    enumerate_paths,
    reachable_states,
)
from post_audit_mirage.dataset import build_dataset, canonical_json_bytes
from post_audit_mirage.surface_generation import MODEL, RETURNED_MODELS

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "data" / "authgate_v0" / "templates.json"
SURFACES = ROOT / "data" / "authgate_v0" / "surface_variants.jsonl"
AUDIT = ROOT / "data" / "authgate_v0" / "audit.jsonl"
TRUTH = ROOT / "data" / "authgate_v0" / "truth.jsonl"


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
            [sys.executable, "-m", "post_audit_mirage", *arguments],
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

            self.assertEqual((len(surfaces), len(audits), len(truths)), (36, 72, 144))
            self.assertEqual(len(requests), 12)
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
                    "post_audit_mirage.dataset.os.replace",
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


if __name__ == "__main__":
    unittest.main()
