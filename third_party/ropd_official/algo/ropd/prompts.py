from __future__ import annotations

import json
from collections.abc import Sequence as SequenceABC
from functools import cache
from pathlib import Path
from typing import Any, Sequence

from algo.ropd_prompts import _render_template, extract_question_text


@cache
def _load_template(template_name: str) -> str:
    template_path = Path(__file__).resolve().parents[2] / "prompts" / template_name
    return template_path.read_text(encoding="utf-8")


def _normalize_answers(answers: str | Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(answers, str):
        return (answers,)
    if not isinstance(answers, SequenceABC):
        raise TypeError(f"{field_name} must be a string or sequence of strings")

    normalized_answers = tuple(str(answer) for answer in answers)
    if not normalized_answers:
        raise ValueError(f"{field_name} must not be empty")
    return normalized_answers


def _render_answer_block(
    label: str,
    answers: str | Sequence[str],
    *,
    field_name: str,
    start_index: int = 0,
    force_labels: bool = False,
) -> str:
    normalized_answers = _normalize_answers(answers, field_name=field_name)
    if len(normalized_answers) == 1 and not force_labels:
        return normalized_answers[0]
    return "\n\n".join(
        f"{label} {index}:\n{answer}"
        for index, answer in enumerate(normalized_answers, start=start_index)
    )


def _resolve_ropd_verifier_template_name(model: str | None) -> str:
    # Skywork-specific template was retired.
    del model
    return "verifier.txt"


def _resolve_ropd_rubricator_template_name(model: str | None) -> str:
    # Skywork-specific template was retired.
    del model
    return "rubricator.txt"


def build_ropd_rubricator_prompt(
    raw_prompt: Any,
    *,
    teacher_answer: str | Sequence[str],
    student_answers: Sequence[str],
    extra_rubric_instructions: str,
    model: str | None = None,
) -> str:
    template = _load_template(_resolve_ropd_rubricator_template_name(model))
    return _render_template(
        template,
        replacements={
            "question": extract_question_text(raw_prompt),
            "teacher_response": _render_answer_block("Reference", teacher_answer, field_name="teacher_answer"),
            "student_response": _render_answer_block("Student", student_answers, field_name="student_answers"),
            "extra_rubric_instructions": extra_rubric_instructions,
        },
    )


def build_ropd_verifier_prompt(
    raw_prompt: Any,
    *,
    rubrics: Any,
    answers: Sequence[str],
    extra_scoring_instructions: str,
    model: str | None = None,
) -> str:
    template = _load_template(_resolve_ropd_verifier_template_name(model))
    return _render_template(
        template,
        replacements={
            "question": extract_question_text(raw_prompt),
            "rubrics": json.dumps(rubrics, ensure_ascii=False, indent=2),
            "answers": _render_answer_block("Answer", answers, field_name="answers", start_index=1, force_labels=True),
            "extra_scoring_instructions": extra_scoring_instructions,
        },
    )
