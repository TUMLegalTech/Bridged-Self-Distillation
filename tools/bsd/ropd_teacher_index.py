"""Shared helpers for building and validating official ROPD teacher indexes."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from verl.trainer.bsd.prompt_assembly import render_student_user_prompt
from verl.trainer.bsd.ropd_vendor import activate_ropd_vendor


def load_ropd_rows(
    parquet_path: str | Path,
    max_rows: int | None = None,
    *,
    emit_think: bool = False,
) -> list[tuple[str, Any]]:
    """Read stable IDs and raw prompts from a verl training parquet.

    ``emit_think`` picks the student template variant and must match what the consuming
    training run resolves, because the offline teacher index is keyed on a hash of this
    prompt. The runtime dataset renders through
    ``resolve_emit_think(data.apply_chat_template_kwargs.enable_thinking, data.emit_think)``
    and the two variants differ in their trailing instruction, so a mismatch here makes
    every lookup miss rather than degrade quietly.
    """

    frame = pd.read_parquet(parquet_path, columns=["extra_info"])
    if max_rows is not None:
        frame = frame.iloc[:max_rows]

    rows: list[tuple[str, Any]] = []
    for row_index, row in frame.iterrows():
        extra_info = row["extra_info"]
        if hasattr(extra_info, "as_py"):
            extra_info = extra_info.as_py()
        if not isinstance(extra_info, Mapping):
            raise ValueError(f"extra_info at parquet row {row_index} must be a mapping.")
        # This explicit extra_info-only chain is load-bearing: together with the
        # rendered prompt it keys the offline teacher index, so adopting the
        # shared common.row_uid precedence would silently miss index lookups.
        uid = extra_info.get("uid") or extra_info.get("id")
        if uid is None or not str(uid).strip():
            raise ValueError(f"extra_info at parquet row {row_index} must contain a non-empty uid or id.")
        problem = extra_info.get("problem")
        if problem is None or not str(problem).strip():
            raise ValueError(f"extra_info at parquet row {row_index} must contain a non-empty problem.")
        user_prompt = render_student_user_prompt(str(problem), emit_think=emit_think)
        rows.append((str(uid), [{"role": "user", "content": user_prompt}]))

    uids = [uid for uid, _ in rows]
    if len(set(uids)) != len(uids):
        raise ValueError("ROPD teacher-index input requires one unique uid per parquet row.")
    return rows


def ropd_resolution_config(
    *,
    provider_spec: str | Path,
    entrypoint: str,
    teacher_answer_count: int,
) -> dict[str, Any]:
    """Build the minimal official client config used by project tooling."""

    return {
        "teacher_answer_count": teacher_answer_count,
        "provider_resolution": {
            "spec_path": str(Path(provider_spec).resolve()),
            "entrypoint": entrypoint,
            "overrides": {"global": {}},
        },
    }


def teacher_fingerprint(judge_config: Any) -> dict[str, Any]:
    """Build the exact fingerprint expected by the official offline-index loader."""

    activate_ropd_vendor()
    from algo.ropd_clients import PROMPT_TEMPLATE_VERSION
    from algo.ropd_teacher_index import build_teacher_fingerprint_payload

    role = judge_config.teacher
    return build_teacher_fingerprint_payload(
        provider="openai_compatible",
        model=role.model,
        base_url=role.base_url,
        reasoning_effort=role.reasoning_effort,
        max_output_tokens=role.max_output_tokens,
        temperature=role.temperature,
        top_p=role.top_p,
        timeout_seconds=role.timeout_seconds,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )


def set_teacher_index_environment(index_path: str | Path) -> None:
    """Set the variable consumed by the pinned provider specification."""

    os.environ["ROPD_TEACHER_INDEX_PATH"] = str(Path(index_path).resolve())


def read_completed_index_keys(index_path: Path) -> set[tuple[str, str]]:
    """Read keys from a partial or complete teacher index."""

    if not index_path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["uid"]), str(row["raw_prompt_hash"]))
        if key in keys:
            raise ValueError(f"Duplicate ROPD teacher-index key at {index_path}:{line_number}: {key!r}")
        keys.add(key)
    return keys
