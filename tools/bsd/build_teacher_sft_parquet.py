#!/usr/bin/env python3
"""Build the SFT parquet (a ``messages`` column) from teacher-response JSONL rows.

Each kept row becomes the student prompt followed by the teacher response as the
assistant turn. A response read from closed_teacher_output, the field in which
generate_closed_teacher_data.py stores the response of a closed teacher, is
first turned into the trace format of the paper (the derivation in a reasoning
block, then the boxed final answer); --no-cot_structure keeps it as returned,
and --cot_structure applies the restructuring to every response field.
--preset closed is a named flag bundle for rows written by
generate_closed_teacher_data.py: the closed_teacher_output field with its
verified flag, and closed-teacher provenance recorded in both extra_info and
the manifest. Like BSD, it keeps the rows whose final answer is wrong;
--require_verified true drops them.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bsd.common import read_jsonl, stable_problem_id, write_json  # noqa: E402
from verl.trainer.bsd.prompt_assembly import (  # noqa: E402
    STUDENT_USER_TEMPLATE,
    load_prompt_template,
    render_student_user_prompt,
)

# generate_closed_teacher_data.py stores the response of a closed teacher under this key.
CLOSED_RESPONSE_FIELD = "closed_teacher_output"

DEFAULT_RESPONSE_FIELDS = (
    CLOSED_RESPONSE_FIELD,
    "teacher_output",
    "response",
    "solution",
    "assistant",
)

# Every field the closed preset fixes is set explicitly, never inherited from a
# default. The preset does not filter on the verified flag: BSD and SFT use the
# same traces, and a trace with a wrong final answer is one of them. The
# enable_thinking column is written for completeness; the SFT launcher takes the
# thinking mode from SFT_THINKING and does not read it.
CLOSED_PRESET = {
    "response_field": CLOSED_RESPONSE_FIELD,
    "verified_field": "closed_teacher_verified",
    "enable_thinking": False,
    "data_source": "bsd/closed_sft",
}


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _write_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(output_path)
    except Exception:
        from datasets import Dataset

        Dataset.from_list(rows).to_parquet(str(output_path))


def _pick_response(row: dict[str, Any], response_field: str | None) -> tuple[str | None, str | None]:
    fields = (response_field,) if response_field else DEFAULT_RESPONSE_FIELDS
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value), field
    return None, None


def _restructure_response(response: str, gold_answer: str) -> str:
    # Imported on demand: only the restructuring needs the verifier.
    from tools.bsd.closed_teacher_trace import cot_structure

    return cot_structure(response, gold_answer)[0]


def build_rows(
    source_rows: list[dict[str, Any]],
    *,
    problem_field: str = "problem",
    response_field: str | None = None,
    answer_field: str = "gold_answer",
    verified_field: str = "verified",
    require_verified: bool = False,
    enable_thinking: bool = False,
    data_source: str = "bsd/offpolicy_sft",
    max_rows: int | None = None,
    closed_provenance: bool = False,
    require_row_fields: bool = False,
    cot_structure: bool | None = None,
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    rows: list[dict[str, Any]] = []
    response_field_counts: Counter[str] = Counter()
    dropped_count = 0
    for source_row in source_rows:
        if require_verified and not bool(source_row.get(verified_field, False)):
            dropped_count += 1
            continue
        problem = source_row.get(problem_field)
        response, used_field = _pick_response(source_row, response_field)
        if response is None or used_field is None:
            dropped_count += 1
            continue
        if require_row_fields:
            # The closed flow fails on a selected row missing its identity or
            # gold answer instead of hashing an id or writing ground_truth=None.
            missing = [
                name
                for name, value in (
                    ("id", source_row.get("id")),
                    (problem_field, problem),
                    (answer_field, source_row.get(answer_field)),
                )
                if value is None
            ]
            if missing:
                raise SystemExit(
                    f"Selected row is missing required field(s) {missing}: keys={sorted(source_row)}"
                )
        if problem is None or not str(problem).strip():
            dropped_count += 1
            continue
        problem_text = str(problem)
        row_id = str(source_row.get("id") or stable_problem_id(problem_text))
        response_field_counts[used_field] += 1
        extra_info: dict[str, Any] = {
            "id": row_id,
            "problem": problem_text,
            "response_field": used_field,
            "verified": bool(source_row.get(verified_field, False)),
        }
        if closed_provenance:
            metadata = source_row.get("metadata")
            generation = metadata.get("closed_teacher_generation", {}) if isinstance(metadata, dict) else {}
            extra_info["closed_teacher_verified"] = bool(source_row.get(verified_field, False))
            extra_info["closed_teacher_model"] = generation.get("model")
        # The target becomes the trace format of the paper for a closed teacher: the
        # derivation in a reasoning block and the boxed final answer after it. Undecided
        # (None), this holds for a response read from closed_teacher_output and for no other.
        restructure = used_field == CLOSED_RESPONSE_FIELD if cot_structure is None else cot_structure
        if restructure:
            gold_answer = source_row.get(answer_field)
            if gold_answer is None:
                raise SystemExit(
                    f"The trace restructuring needs {answer_field!r} in every row; missing for id={row_id}. "
                    "Pass --no-cot_structure to keep the responses as returned."
                )
            response = _restructure_response(response, str(gold_answer))
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": render_student_user_prompt(problem_text)},
                    {"role": "assistant", "content": response},
                ],
                "enable_thinking": bool(enable_thinking),
                "data_source": data_source,
                "reward_model": {"style": "rule", "ground_truth": source_row.get(answer_field)},
                "extra_info": extra_info,
            }
        )
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows, response_field_counts, dropped_count


def main() -> None:
    # allow_abbrev=False keeps the preset's sys.argv conflict scan complete: an
    # abbreviated flag like --response_f would otherwise parse yet evade it.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_parquet", required=True)
    parser.add_argument("--manifest_json", default=None)
    parser.add_argument("--problem_field", default="problem")
    parser.add_argument("--response_field", default=None)
    parser.add_argument("--answer_field", default="gold_answer")
    parser.add_argument("--verified_field", default="verified")
    parser.add_argument("--require_verified", type=_parse_bool, default=False)
    parser.add_argument("--allow_unverified", action="store_true")
    parser.add_argument("--enable_thinking", type=_parse_bool, default=False)
    parser.add_argument("--data_source", default="bsd/offpolicy_sft")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--preset", choices=("closed",), default=None,
                        help="Named field bundle; 'closed' is the closed-teacher SFT flow.")
    parser.add_argument("--closed_teacher_model", default=os.environ.get("CLOSED_TEACHER_MODEL"),
                        help="Recorded in the manifest when --preset closed is used.")
    parser.add_argument("--cot_structure", action=argparse.BooleanOptionalAction, default=None,
                        help="For a closed teacher that hides its reasoning: build the SFT target from the stored "
                             "response by enclosing its derivation in a reasoning block (<think> and </think>) and "
                             "placing its boxed final answer after the block. On by default for a response read "
                             "from closed_teacher_output and off for every other field; --no-cot_structure keeps "
                             "every response as returned, and --cot_structure restructures every response. The "
                             "verified flag and --require_verified still refer to the response as stored. Not for "
                             "a response that already has a reasoning block.")
    args = parser.parse_args()

    if args.preset == "closed":
        # The preset owns these flags outright; any explicit occurrence on the
        # command line is an error, even one that restates a default, so a
        # caller's stated intent is never silently overridden.
        owned_flags = {"--response_field", "--verified_field", "--enable_thinking", "--data_source"}
        explicit = sorted({arg.split("=", 1)[0] for arg in sys.argv[1:]} & owned_flags)
        if explicit:
            parser.error(f"--preset closed owns these flags; do not pass them: {', '.join(explicit)}")
        for name, value in CLOSED_PRESET.items():
            setattr(args, name, value)

    require_verified = False if args.allow_unverified else bool(args.require_verified)
    source_rows = read_jsonl(args.input_jsonl)
    rows, response_field_counts, dropped_count = build_rows(
        source_rows,
        problem_field=args.problem_field,
        response_field=args.response_field,
        answer_field=args.answer_field,
        verified_field=args.verified_field,
        require_verified=require_verified,
        enable_thinking=bool(args.enable_thinking),
        data_source=args.data_source,
        max_rows=args.max_rows,
        closed_provenance=args.preset == "closed",
        require_row_fields=args.preset == "closed",
        cot_structure=args.cot_structure,
    )
    if not rows:
        raise SystemExit("No teacher-response rows found; cannot build SFT parquet.")

    if args.cot_structure is None:
        cot_structured_count = response_field_counts.get(CLOSED_RESPONSE_FIELD, 0)
    else:
        cot_structured_count = len(rows) if args.cot_structure else 0

    output_path = Path(args.output_parquet)
    _write_parquet(rows, output_path)
    manifest_path = Path(args.manifest_json) if args.manifest_json else output_path.with_name("manifest.json")
    manifest = {
        "input_jsonl": args.input_jsonl,
        "output_parquet": str(output_path),
        "source_count": len(source_rows),
        "written_count": len(rows),
        "dropped_count": dropped_count,
        "problem_field": args.problem_field,
        "response_field": args.response_field or list(DEFAULT_RESPONSE_FIELDS),
        "response_field_counts": dict(response_field_counts),
        "answer_field": args.answer_field,
        "verified_field": args.verified_field,
        "require_verified": require_verified,
        "enable_thinking": bool(args.enable_thinking),
        "data_source": args.data_source,
        "cot_structure": cot_structured_count > 0,
        "cot_structured_count": cot_structured_count,
        "prompt_template_sha256": sha256(load_prompt_template(STUDENT_USER_TEMPLATE).encode("utf-8")).hexdigest(),
        "messages_key": "messages",
        "loss_mask_source": "verl MultiTurnSFTDataset masks non-assistant turns and assistant generation prompt.",
    }
    if args.preset == "closed":
        verified_source_count = sum(
            1
            for row in source_rows
            if bool(row.get(args.verified_field, False)) and row.get(args.response_field)
        )
        manifest["preset"] = "closed"
        manifest["closed_teacher_model"] = args.closed_teacher_model
        manifest["verified_source_count"] = verified_source_count
        manifest["verified_rate"] = verified_source_count / len(source_rows) if source_rows else 0.0
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
