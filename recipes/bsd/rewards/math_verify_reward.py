"""Shared math verifier reward for Bridged Self-Distillation training and evaluation."""

from __future__ import annotations

import os
import re
from typing import Any


def _format_bonus() -> float:
    """Score for a boxed-but-wrong answer. The default 0 keeps the verifier
    reward binary, so group advantages are driven by correctness alone; a
    positive BSD_FORMAT_BONUS adds format shaping."""

    return float(os.environ.get("BSD_FORMAT_BONUS", "0"))


def extract_boxed_answer(text: str) -> str | None:
    """Extract the final ``\\boxed{...}`` answer from a model response."""

    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    brace_idx = text.find("{", idx)
    if brace_idx < 0:
        return None
    depth = 0
    for pos in range(brace_idx, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_idx + 1 : pos].strip()
    return None


def _normalize_string(value: str) -> str:
    return re.sub(r"\s+", "", value.replace("$", "")).lower()


def verify_answer(predicted: str | None, ground_truth: str) -> bool:
    """Verify a boxed prediction against the ground truth answer."""

    if predicted is None:
        return False
    try:
        from math_verify import parse, verify

        pred_text = predicted if "$" in predicted else f"${predicted}$"
        gt_text = ground_truth if "$" in ground_truth else f"${ground_truth}$"
        return bool(verify(parse(gt_text, fallback_mode="no_fallback"), parse(pred_text, fallback_mode="no_fallback")))
    except Exception:
        return _normalize_string(predicted) == _normalize_string(str(ground_truth))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reward manager entry point used by verl."""

    del data_source, extra_info
    boxed = extract_boxed_answer(solution_str)
    correct = verify_answer(boxed, str(ground_truth))
    formatted = boxed is not None
    return {
        "score": 1.0 if correct else (_format_bonus() if formatted else 0.0),
        "accuracy": 1.0 if correct else 0.0,
        "format": 1.0 if formatted else 0.0,
        "boxed_answer": boxed or "",
    }
