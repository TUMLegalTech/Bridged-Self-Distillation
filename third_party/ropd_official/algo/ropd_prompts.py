from __future__ import annotations

import json
import re
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

PROMPT_TEMPLATE_VERSION = "phase2.v3"

_SUPPORTED_MESSAGE_ROLES = {"user", "assistant", "system", "developer"}
_PLACEHOLDER_PATTERN = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@cache
def _load_template(template_name: str) -> str:
    template_path = _repo_root() / "prompts" / template_name
    return template_path.read_text(encoding="utf-8")


def _normalize_raw_prompt(raw_prompt: Any) -> str | tuple[dict[str, Any], ...]:
    if isinstance(raw_prompt, str):
        return raw_prompt

    if hasattr(raw_prompt, "tolist") and not isinstance(raw_prompt, bytes | bytearray):
        raw_prompt = raw_prompt.tolist()

    if isinstance(raw_prompt, tuple):
        raw_prompt = list(raw_prompt)

    if isinstance(raw_prompt, list):
        normalized_messages: list[dict[str, Any]] = []
        for message in raw_prompt:
            if not isinstance(message, Mapping):
                raise TypeError(f"raw_prompt messages must be mappings, got {type(message)!r}")
            normalized_messages.append(dict(message))
        return tuple(normalized_messages)

    raise TypeError(f"Unsupported raw_prompt type: {type(raw_prompt)!r}")


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                if "text" in item:
                    parts.append(str(item["text"]).strip())
                    continue
                if item.get("type") == "input_text" and "text" in item:
                    parts.append(str(item["text"]).strip())
                    continue
            parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
        return "\n".join(part for part in parts if part)

    if isinstance(content, Mapping):
        if "text" in content:
            return str(content["text"]).strip()
        return json.dumps(dict(content), ensure_ascii=False, sort_keys=True)

    return str(content).strip()


def extract_question_text(raw_prompt: Any) -> str:
    normalized = _normalize_raw_prompt(raw_prompt)
    if isinstance(normalized, str):
        return normalized.strip()

    if len(normalized) == 1:
        only_message = normalized[0]
        return _stringify_content(only_message.get("content"))

    rendered_messages: list[str] = []
    for message in normalized:
        role = str(message.get("role", "user")).strip() or "user"
        content = _stringify_content(message.get("content"))
        if not content:
            continue
        rendered_messages.append(f"{role.upper()}: {content}")

    return "\n\n".join(rendered_messages).strip()


def build_teacher_input_messages(raw_prompt: Any) -> list[dict[str, Any]]:
    normalized = _normalize_raw_prompt(raw_prompt)
    if isinstance(normalized, str):
        return [{"role": "user", "content": normalized}]

    messages: list[dict[str, Any]] = []
    for message in normalized:
        role = str(message.get("role", "user")).strip() or "user"
        if role not in _SUPPORTED_MESSAGE_ROLES:
            role = "user"

        content = message.get("content")
        if isinstance(content, str | list):
            normalized_content = content
        else:
            normalized_content = _stringify_content(content)

        messages.append({"role": role, "content": normalized_content})

    return messages


def _dump_prompt_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _looks_like_skywork_model(model: str | None) -> bool:
    if model is None:
        return False
    return "skywork" in model.strip().lower()


def _resolve_verifier_template_name(model: str | None) -> str:
    # Skywork-specific template was retired; both branches now share verifier.txt.
    del model
    return "verifier.txt"


def _count_rubrics(rubrics: Any) -> int | None:
    if isinstance(rubrics, list | tuple):
        return len(rubrics)
    return None


def _render_ordered_rubrics(rubrics: Any) -> str:
    if not isinstance(rubrics, list | tuple):
        return _dump_prompt_json(rubrics)

    rendered_items: list[str] = []
    for index, rubric in enumerate(rubrics, start=1):
        if isinstance(rubric, Mapping):
            criterion_id = str(rubric.get("criterion_id", f"c{index}")).strip() or f"c{index}"
            points = rubric.get("points", "")
            category = str(rubric.get("category", "")).strip()
            criterion = str(rubric.get("criterion", "")).strip()
            rendered_items.append(
                f"{index}. [{criterion_id}] points={points} | category={category} | criterion={criterion}"
            )
            continue
        rendered_items.append(f"{index}. {_dump_prompt_json(rubric)}")
    return "\n".join(rendered_items)


def _render_judgement_slot_mapping(rubrics: Any) -> str:
    if not isinstance(rubrics, list | tuple):
        return "judgement[0] -> unknown"

    rendered_items: list[str] = []
    for index, rubric in enumerate(rubrics):
        if isinstance(rubric, Mapping):
            criterion_id = str(rubric.get("criterion_id", f"c{index + 1}")).strip() or f"c{index + 1}"
        else:
            criterion_id = f"c{index + 1}"
        rendered_items.append(f"judgement[{index}] -> {criterion_id}")
    return "\n".join(rendered_items)


def _render_template(template: str, *, replacements: Mapping[str, str]) -> str:
    unknown_placeholders = {
        match.group(1) for match in _PLACEHOLDER_PATTERN.finditer(template) if match.group(1) not in replacements
    }
    if unknown_placeholders:
        unknown_list = ", ".join(sorted(unknown_placeholders))
        raise ValueError(f"Unsupported template placeholder(s): {unknown_list}")

    return _PLACEHOLDER_PATTERN.sub(lambda match: replacements[match.group(1)], template)


def build_rubricator_prompt(raw_prompt: Any, *, teacher_response: str, student_response: str) -> str:
    template = _load_template("rubricator.txt")
    return _render_template(
        template,
        replacements={
            "question": extract_question_text(raw_prompt),
            "teacher_response": teacher_response,
            "student_response": student_response,
        },
    )


def build_verifier_prompt(raw_prompt: Any, *, response: str, rubrics: Any, model: str | None = None) -> str:
    template = _load_template(_resolve_verifier_template_name(model))
    rubric_count = _count_rubrics(rubrics)
    return _render_template(
        template,
        replacements={
            "question": extract_question_text(raw_prompt),
            "resp": response,
            "rubrics": _dump_prompt_json(rubrics),
            "ordered_rubrics": _render_ordered_rubrics(rubrics),
            "judgement_slot_mapping": _render_judgement_slot_mapping(rubrics),
            "rubric_count": "unknown" if rubric_count is None else str(rubric_count),
        },
    )


__all__ = [
    "PROMPT_TEMPLATE_VERSION",
    "build_rubricator_prompt",
    "build_teacher_input_messages",
    "build_verifier_prompt",
    "extract_question_text",
]
