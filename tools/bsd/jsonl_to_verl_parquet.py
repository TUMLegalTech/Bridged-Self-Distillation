#!/usr/bin/env python3
"""Convert Bridged Self-Distillation JSONL rows into verl-compatible parquet rows.

The trace of a row is read from --pi_trace_field and written to extra_info under the
same key, or under --output_trace_field. A response read from closed_teacher_output,
the field in which generate_closed_teacher_data.py stores the response of a closed
teacher, is first turned into the trace format of the paper (the derivation in a
reasoning block, then the boxed final answer); --no-cot_structure keeps it as returned.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.bsd.common import read_jsonl, row_uid
from verl.trainer.bsd.prompt_assembly import DEFAULT_PI_TRACE_FIELD, render_student_user_prompt


# generate_closed_teacher_data.py stores the response of a closed teacher under this key.
CLOSED_TEACHER_RESPONSE_FIELD = "closed_teacher_output"


def _verified_field(trace_field: str) -> str:
    if trace_field == CLOSED_TEACHER_RESPONSE_FIELD:
        # generate_closed_teacher_data.py writes the correctness flag of this field under this key.
        return "closed_teacher_verified"
    return trace_field.replace("_trace", "_verified") if trace_field.endswith("_trace") else f"{trace_field}_verified"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_parquet", required=True)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--pi_trace_field", default=DEFAULT_PI_TRACE_FIELD)
    parser.add_argument(
        "--carry_key",
        action="append",
        default=[],
        help="Extra top-level source key copied into extra_info when present (repeatable).",
    )
    parser.add_argument(
        "--output_trace_field",
        default=None,
        help=(
            "Key of extra_info under which the trace is written; default: the value of "
            "--pi_trace_field. With --pi_trace_field closed_teacher_output it stores the trace "
            "of a closed teacher under a <teacher>_pi_trace key such as luna_pi_trace."
        ),
    )
    parser.add_argument(
        "--cot_structure",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "For a closed teacher that hides its reasoning: build the trace from the stored "
            "response by enclosing its derivation in a reasoning block (<think> and </think>) "
            "and placing its boxed final answer after the block, and set the verified flag from "
            "that answer. On by default when --pi_trace_field is closed_teacher_output and off "
            "for every other field; --no-cot_structure keeps the response as returned. Not for "
            "a trace that already has a reasoning block."
        ),
    )
    args = parser.parse_args()
    if args.cot_structure is None:
        # Undecided: the paper's trace format is the default for the response of a closed teacher.
        args.cot_structure = args.pi_trace_field == CLOSED_TEACHER_RESPONSE_FIELD
    if args.cot_structure:
        # Imported on demand: only this option needs the restructuring and the verifier.
        from recipes.bsd.rewards.math_verify_reward import verify_answer
        from tools.bsd.closed_teacher_trace import cot_structure

    rows = []
    pi_verified_field = _verified_field(args.pi_trace_field)
    output_trace_field = args.output_trace_field or args.pi_trace_field
    output_verified_field = _verified_field(output_trace_field)
    for row in read_jsonl(args.input_jsonl):
        problem = row["problem"]
        pi_trace = row.get(args.pi_trace_field)
        pi_verified = row.get(pi_verified_field, False)
        if args.cot_structure and pi_trace:
            # The answer placed after the block is the box that the verifier accepts when
            # there is one, so the flag is derived again from it to stay true of the trace.
            gold_answer = str(row["gold_answer"])
            pi_trace, boxed_answer = cot_structure(str(pi_trace), gold_answer)
            pi_verified = bool(boxed_answer) and bool(verify_answer(boxed_answer, gold_answer))
        rows.append(
            {
                args.prompt_key: [
                    {
                        "role": "user",
                        "content": render_student_user_prompt(problem),
                    }
                ],
                "data_source": "bsd/math",
                "reward_model": {"style": "rule", "ground_truth": row["gold_answer"]},
                "extra_info": {
                    # Rows that carry only a uid (the trace generator accepts either key) use it as id.
                    "id": row["id"] if row.get("id") is not None else row_uid(row),
                    "problem": problem,
                    # Carry through whatever teacher keys the source actually has,
                    # rather than materialising empty slots for teachers it never used.
                    # The selected field is written after, so it always wins.
                    **{
                        k: row[k]
                        for k in row
                        if k.endswith(("_pi_trace", "_pi_verified"))
                        and k not in (args.pi_trace_field, pi_verified_field)
                    },
                    output_trace_field: pi_trace,
                    output_verified_field: pi_verified,
                    "pi_trace_field": output_trace_field,
                    **{k: row[k] for k in args.carry_key if k in row},
                },
            }
        )
    output = Path(args.output_parquet)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(output)
    except Exception:
        from datasets import Dataset

        Dataset.from_list(rows).to_parquet(str(output))


if __name__ == "__main__":
    main()
