"""Shared utility functions for Bridged Self-Distillation scripts."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

# Storage root; the default mirrors the recipe launchers: <repository root>/workspace.
DATA_ROOT = Path(os.environ.get("BSD_DATA_ROOT") or str(Path(__file__).resolve().parents[2] / "workspace"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_generation_manifest(
    path: str | Path,
    *,
    status: str,
    model: str,
    parameters: dict[str, Any],
) -> None:
    """Write reproducible runtime metadata for an offline generation job."""

    from tools.bsd.runtime import installed_versions

    packages = {name: "" for name in ("torch", "transformers", "vllm", "math-verify")}
    write_json(
        path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "model": model,
            "python": str(Path(sys.executable).resolve()),
            "versions": installed_versions(packages),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "parameters": parameters,
        },
    )


def _pick_consistent(label: str, candidates: list[tuple[str, Any]]) -> Any:
    """Return the first present candidate, refusing silent disagreement.

    The candidates name the same fact stored in more than one place; when two
    of them are present with different values, no precedence order is correct,
    so fail instead of letting each tool quietly pick a different one.
    """

    present = [(name, value) for name, value in candidates if value is not None]
    if not present:
        return None
    if len({str(value) for _, value in present}) > 1:
        detail = ", ".join(f"{name}={value!r}" for name, value in present)
        raise ValueError(f"Conflicting {label} values on one row: {detail}")
    return present[0][1]


def row_uid(row: dict[str, Any], *, allow_missing: bool = False) -> str | None:
    """Stable row identifier: the uid tier first, then the id tier.

    uid and id are different tiers, not synonyms, so they are never
    cross-checked against each other; within a tier the top-level and
    extra_info copies must agree.
    """

    extra_info = row.get("extra_info") or {}
    identifier = _pick_consistent(
        "uid", [("uid", row.get("uid")), ("extra_info.uid", extra_info.get("uid"))]
    )
    if identifier is None:
        identifier = _pick_consistent(
            "id", [("id", row.get("id")), ("extra_info.id", extra_info.get("id"))]
        )
    if identifier is None:
        if allow_missing:
            return None
        raise ValueError(f"Row has no uid/id: keys={sorted(row)}")
    return str(identifier)


def row_problem(row: dict[str, Any]) -> str:
    """Problem text, wherever the dataset stores it, with mirror-copy checking."""

    extra_info = row.get("extra_info") or {}
    problem = _pick_consistent(
        "problem",
        [("problem", row.get("problem")), ("extra_info.problem", extra_info.get("problem"))],
    )
    if problem is None or not str(problem).strip():
        raise ValueError(f"Row has no problem text: keys={sorted(row)}")
    return str(problem)


def row_gold_answer(row: dict[str, Any]) -> str:
    """Final gold answer for verification.

    gold_answer and reward_model.ground_truth record the same fact and must
    agree when both are present; the looser answer fields are only a fallback
    when both are absent, because in some datasets they hold a full solution
    rather than the boxed value.
    """

    reward_model = row.get("reward_model") or {}
    extra_info = row.get("extra_info") or {}
    gold = _pick_consistent(
        "gold answer",
        [
            ("gold_answer", row.get("gold_answer")),
            ("reward_model.ground_truth", reward_model.get("ground_truth")),
        ],
    )
    if gold is None:
        gold = row.get("answer") if row.get("answer") is not None else extra_info.get("answer")
    if gold is None:
        raise ValueError(f"Row has no gold answer: keys={sorted(row)}")
    return str(gold)


def normalize_problem(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def stable_problem_id(problem: str) -> str:
    return sha256(normalize_problem(problem).encode("utf-8")).hexdigest()


def compact_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text).lower())


def ngram_set(text: str, n: int = 5) -> set[str]:
    tokens = re.findall(r"\w+", str(text).lower())
    if len(tokens) < n:
        return set(tokens)
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
