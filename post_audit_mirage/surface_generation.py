from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "openai/gpt-5.6-luna"
RETURNED_MODELS = {MODEL, "openai/gpt-5.6-luna-20260709"}
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_VERSION = "authgate-v0-1"
VARIANT_FIELDS = {"variant_id", "organization", "resource_name", "request_text"}


class SurfaceGenerationError(ValueError):
    pass


def _response_schema(template_id: str) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "template_id": {"type": "string", "enum": [template_id]},
            "variants": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "variant_id": {
                            "type": "string",
                            "enum": ["v1", "v2", "v3"],
                        },
                        "organization": {"type": "string"},
                        "resource_name": {"type": "string"},
                        "request_text": {"type": "string"},
                    },
                    "required": sorted(VARIANT_FIELDS),
                    "additionalProperties": False,
                },
            },
        },
        "required": ["template_id", "variants"],
        "additionalProperties": False,
    }


def _request_payload(template: dict[str, object], seed: int) -> dict[str, object]:
    visible_template = {
        "approval_present": template["approval_present"],
        "operation": template["operation"],
        "requester_role": template["requester_role"],
        "resource_class": template["resource_class"],
        "template_id": template["template_id"],
    }
    return {
        "max_tokens": 800,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Write realistic surface wording for the supplied authorization "
                    "template. Return only observable names and request text. Do not "
                    "infer outcomes, policies, hidden state, probabilities, or labels."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(visible_template, sort_keys=True),
            },
        ],
        "model": MODEL,
        "provider": {
            "allow_fallbacks": False,
            "only": ["openai"],
            "require_parameters": True,
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "authgate_surface_variants",
                "schema": _response_schema(str(template["template_id"])),
                "strict": True,
            },
        },
        "seed": seed,
    }


def _post_json(
    endpoint: str,
    api_key: str,
    payload: dict[str, object],
) -> dict[str, object]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    for attempt in range(3):
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
            if not isinstance(result, dict):
                raise SurfaceGenerationError(
                    "OpenRouter returned a non-object response"
                )
            return result
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise SurfaceGenerationError(
                    f"OpenRouter request failed with status {error.code}"
                ) from None
        except urllib.error.URLError as error:
            if attempt == 2:
                raise SurfaceGenerationError("OpenRouter request failed") from error
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _clean_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise SurfaceGenerationError(f"{field} must be a string")
    cleaned = " ".join(value.replace("\u2014", "-").split())
    if not cleaned:
        raise SurfaceGenerationError(f"{field} must not be empty")
    return cleaned


def _parse_response(
    response: dict[str, object],
    template: dict[str, object],
    seed: int,
) -> list[dict[str, object]]:
    if response.get("model") not in RETURNED_MODELS:
        raise SurfaceGenerationError("OpenRouter returned an unexpected model")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise SurfaceGenerationError("OpenRouter returned an invalid choices list")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        raise SurfaceGenerationError("OpenRouter did not finish the response")
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise SurfaceGenerationError("OpenRouter returned invalid message content")
    try:
        content = json.loads(message["content"])
    except json.JSONDecodeError as error:
        raise SurfaceGenerationError(
            "OpenRouter returned invalid JSON content"
        ) from error
    if not isinstance(content, dict) or set(content) != {"template_id", "variants"}:
        raise SurfaceGenerationError("surface response has unexpected fields")
    if content["template_id"] != template["template_id"]:
        raise SurfaceGenerationError("surface response changed the template ID")
    variants = content["variants"]
    if not isinstance(variants, list) or len(variants) != 3:
        raise SurfaceGenerationError("surface response must contain three variants")

    rows: list[dict[str, object]] = []
    for variant in variants:
        if not isinstance(variant, dict) or set(variant) != VARIANT_FIELDS:
            raise SurfaceGenerationError("surface variant has unexpected fields")
        rows.append(
            {
                "approval_present": template["approval_present"],
                "operation": template["operation"],
                "organization": _clean_text(variant["organization"], "organization"),
                "prompt_version": PROMPT_VERSION,
                "request_seed": seed,
                "request_text": _clean_text(variant["request_text"], "request_text"),
                "requester_role": template["requester_role"],
                "requested_model": MODEL,
                "resource_class": template["resource_class"],
                "resource_name": _clean_text(variant["resource_name"], "resource_name"),
                "returned_model": str(response["model"]),
                "split": template["split"],
                "template_id": template["template_id"],
                "variant_id": _clean_text(variant["variant_id"], "variant_id"),
            }
        )
    variant_ids = {row["variant_id"] for row in rows}
    request_texts = {row["request_text"] for row in rows}
    if variant_ids != {"v1", "v2", "v3"} or len(request_texts) != 3:
        raise SurfaceGenerationError("surface variants must be unique v1, v2, and v3")
    return rows


def _json_line(row: dict[str, object]) -> str:
    return (
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


def generate_surface_file(
    templates: list[dict[str, object]],
    output_path: Path,
    *,
    seed: int,
    endpoint: str = DEFAULT_ENDPOINT,
) -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SurfaceGenerationError("OPENROUTER_API_KEY is required")

    rows: list[dict[str, object]] = []
    for index, template in enumerate(templates):
        request_seed = seed + index
        response = _post_json(
            endpoint,
            api_key,
            _request_payload(template, request_seed),
        )
        rows.extend(_parse_response(response, template, request_seed))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for row in rows:
                temporary.write(_json_line(row))
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return len(rows)
