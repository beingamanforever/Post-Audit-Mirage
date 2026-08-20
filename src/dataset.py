from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from .authgate import (
    Case,
    Policy,
    World,
    audit_record,
    truth_record,
)
from .surface_generation import MODEL, PROMPT_VERSION, RETURNED_MODELS

TEMPLATE_FIELDS = {
    "approval_present",
    "operation",
    "requester_group",
    "requester_role",
    "resource_class",
    "split",
    "template_id",
}
SURFACE_FIELDS = {
    "approval_present",
    "operation",
    "organization",
    "prompt_version",
    "request_seed",
    "request_text",
    "requester_role",
    "requested_model",
    "resource_class",
    "resource_name",
    "returned_model",
    "split",
    "template_id",
    "variant_id",
}


def canonical_json_line(row: dict[str, object]) -> str:
    return (
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


def canonical_json_bytes(row: dict[str, object]) -> bytes:
    return canonical_json_line(row).encode("utf-8")


def load_templates(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as source:
        templates = json.load(source)
    if not isinstance(templates, list) or len(templates) != 12:
        raise ValueError("templates.json must contain twelve templates")

    template_ids: set[str] = set()
    for template in templates:
        if not isinstance(template, dict) or set(template) != TEMPLATE_FIELDS:
            raise ValueError("template has unexpected fields")
        template_id = template["template_id"]
        if not isinstance(template_id, str) or template_id in template_ids:
            raise ValueError("template IDs must be unique strings")
        if not isinstance(template["approval_present"], bool):
            raise ValueError("approval_present must be boolean")
        for field in TEMPLATE_FIELDS - {"approval_present"}:
            if not isinstance(template[field], str) or not template[field]:
                raise ValueError(f"template {field} must be a non-empty string")
        template_ids.add(template_id)

    split_counts = Counter(str(template["split"]) for template in templates)
    if split_counts != {"development": 8, "diagnostic": 2, "confirmation": 2}:
        raise ValueError(
            "template splits must be 8 development, 2 diagnostic, and 2 confirmation"
        )
    return sorted(templates, key=lambda item: str(item["template_id"]))


def load_surfaces(
    path: Path,
    templates: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid surface JSON on line {line_number}"
                ) from error
            if not isinstance(row, dict) or set(row) != SURFACE_FIELDS:
                raise ValueError(f"surface row {line_number} has unexpected fields")
            rows.append(row)
    if len(rows) != 36:
        raise ValueError("surface_variants.jsonl must contain 36 rows")

    templates_by_id = {str(item["template_id"]): item for item in templates}
    variants_by_template: defaultdict[str, set[str]] = defaultdict(set)
    text_by_template: defaultdict[str, set[str]] = defaultdict(set)
    seeds_by_template: defaultdict[str, set[int]] = defaultdict(set)
    for row in rows:
        template_id = row["template_id"]
        if not isinstance(template_id, str):
            raise ValueError("surface template IDs must be strings")
        if template_id not in templates_by_id:
            raise ValueError("surface row references an unknown template")
        template = templates_by_id[template_id]
        for field in {
            "approval_present",
            "operation",
            "requester_role",
            "resource_class",
        }:
            if (
                type(row[field]) is not type(template[field])
                or row[field] != template[field]
            ):
                raise ValueError(f"surface row changed its template {field}")
        if row["split"] != template["split"]:
            raise ValueError("surface row changed its template split")
        for field in {
            "organization",
            "request_text",
            "resource_name",
            "split",
            "variant_id",
        }:
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"surface {field} must be a non-empty string")
        if row["prompt_version"] != PROMPT_VERSION:
            raise ValueError("surface row has an unexpected prompt version")
        if row["requested_model"] != MODEL:
            raise ValueError("surface row has an unexpected requested model")
        if (
            not isinstance(row["returned_model"], str)
            or row["returned_model"] not in RETURNED_MODELS
        ):
            raise ValueError("surface row has an unexpected returned model")
        request_seed = row["request_seed"]
        if type(request_seed) is not int:
            raise ValueError("surface request seed must be an integer")
        variant_id = row["variant_id"]
        request_text = row["request_text"]
        variants_by_template[str(template_id)].add(variant_id)
        text_by_template[str(template_id)].add(request_text)
        seeds_by_template[str(template_id)].add(request_seed)

    for template_id in templates_by_id:
        if variants_by_template[template_id] != {"v1", "v2", "v3"}:
            raise ValueError(f"template {template_id} must contain v1, v2, and v3")
        if len(text_by_template[template_id]) != 3:
            raise ValueError(f"template {template_id} must contain unique request text")
        if len(seeds_by_template[template_id]) != 1:
            raise ValueError("each template must use one request seed")

    first_seed = next(iter(seeds_by_template[str(templates[0]["template_id"])]))
    for index, template_id in enumerate(templates_by_id):
        if seeds_by_template[template_id] != {first_seed + index}:
            raise ValueError("surface request seeds do not match template order")
    return sorted(
        rows, key=lambda item: (str(item["template_id"]), str(item["variant_id"]))
    )


def _case(
    template: dict[str, object],
    surface: dict[str, object],
) -> Case:
    return Case(
        case_id=f"{template['template_id']}-{surface['variant_id']}",
        template_id=str(template["template_id"]),
        variant_id=str(surface["variant_id"]),
        split=str(template["split"]),
        requester_group=str(template["requester_group"]),
        approval_present=bool(template["approval_present"]),
        organization=str(surface["organization"]),
        resource_name=str(surface["resource_name"]),
        request_text=str(surface["request_text"]),
    )


def _stage_rows(directory: Path, rows: list[dict[str, object]]) -> Path:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=directory,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for row in rows:
                temporary.write(canonical_json_line(row))
        return temporary_path
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def publish_rows(
    output_dir: Path,
    rows_by_name: dict[str, list[dict[str, object]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    backups: dict[Path, Path] = {}
    published: set[Path] = set()
    try:
        for name, rows in rows_by_name.items():
            staged[name] = _stage_rows(output_dir, rows)
        for name in rows_by_name:
            target = output_dir / name
            if target.exists():
                with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as file:
                    backup = Path(file.name)
                try:
                    os.replace(target, backup)
                except Exception:
                    backup.unlink(missing_ok=True)
                    raise
                backups[target] = backup
        for name in rows_by_name:
            target = output_dir / name
            os.replace(staged[name], target)
            published.add(target)
    except Exception:
        for target in published:
            if target not in backups:
                target.unlink(missing_ok=True)
        for target, backup in backups.items():
            os.replace(backup, target)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


def _publish_dataset(
    output_dir: Path,
    audit_rows: list[dict[str, object]],
    truth_rows: list[dict[str, object]],
) -> None:
    publish_rows(
        output_dir,
        {"audit.jsonl": audit_rows, "truth.jsonl": truth_rows},
    )


def build_dataset(
    templates_path: Path,
    surfaces_path: Path,
    output_dir: Path,
) -> tuple[int, int]:
    templates = load_templates(templates_path)
    surfaces = load_surfaces(surfaces_path, templates)
    templates_by_id = {str(item["template_id"]): item for item in templates}
    audit_rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []

    for surface in surfaces:
        template = templates_by_id[str(surface["template_id"])]
        case = _case(template, surface)
        for policy in Policy:
            safe_audit = audit_record(case, policy, World.SAFE)
            harmful_audit = audit_record(case, policy, World.HARMFUL)
            if canonical_json_bytes(safe_audit) != canonical_json_bytes(harmful_audit):
                raise ValueError(
                    f"paired audit differs for {case.case_id} and {policy.value}"
                )
            audit_rows.append(safe_audit)
            for world in World:
                truth_rows.append(truth_record(case, policy, world))

    if len(audit_rows) != 72 or len(truth_rows) != 144:
        raise ValueError("AuthGate-v0 produced an unexpected number of rows")
    _publish_dataset(output_dir, audit_rows, truth_rows)
    return len(audit_rows), len(truth_rows)
