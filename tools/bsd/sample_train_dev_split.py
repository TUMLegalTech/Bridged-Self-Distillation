#!/usr/bin/env python3
"""Sample a training split and a disjoint development split from rows with teacher traces.

The input is the JSONL output of a trace generator (generate_local_traces.py or
generate_closed_teacher_data.py): one row per problem, with the trace in
--pi_trace_field and the verifier's verdict on its final answer in the matching flag.
A row is eligible when it has a trace that was not cut at the token limit and, unless
--keep_unverified_traces is given, a verified final answer. The eligible rows are
shuffled with --seed; the first --train_size (default 2,400) go to train.jsonl and the
next --dev_size (default 128) to dev.jsonl. split_manifest.json records the seed and
the counts.

Without --keep_unverified_traces every trace of the training split has a correct final
answer, as in the paper's training sets with the Qwen3-32B traces and in its
Nemotron-3-30B-A3B sets for Qwen3-4B and Qwen3-4B-Instruct. With it the traces with a
wrong final answer are kept as well, as in the paper's training sets of the closed API
teacher; the share of correct traces is then the one that the teacher reaches on the
sampled problems. No option fixes this share, so the 90.0% of the paper's
Nemotron-3-30B-A3B set for Qwen3-1.7B and Qwen3-0.6B is not reproduced exactly.

The split is a function of the seed and of the order of the input rows, so pass several
input files in a fixed order.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from tools.bsd.common import read_jsonl, row_uid, write_json, write_jsonl


def _verified_field(trace_field: str) -> str:
    if trace_field == "closed_teacher_output":
        # generate_closed_teacher_data.py writes the correctness flag of this field under this key.
        return "closed_teacher_verified"
    return trace_field.replace("_trace", "_verified") if trace_field.endswith("_trace") else f"{trace_field}_verified"


def _trace_truncated(row: dict, trace_field: str) -> bool:
    """Whether the stored trace was cut at the token limit, as its generator recorded it."""

    if trace_field == "closed_teacher_output":
        generation = (row.get("metadata") or {}).get("closed_teacher_generation") or {}
        return bool(generation.get("trace_truncated", False))
    return bool(row.get("truncated", False))


def eligible(row: dict, *, pi_trace_field: str, keep_unverified_traces: bool) -> bool:
    if not row.get(pi_trace_field) or _trace_truncated(row, pi_trace_field):
        return False
    return keep_unverified_traces or bool(row.get(_verified_field(pi_trace_field), False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_jsonl",
        nargs="+",
        required=True,
        help="One or more JSONL files with traces, for example the worker_<i>/raw.jsonl files of one teacher.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_size", type=int, default=2400, help="Number of training problems.")
    parser.add_argument("--dev_size", type=int, default=128, help="Number of development problems.")
    parser.add_argument(
        "--pi_trace_field",
        required=True,
        help="Key that holds the trace of the external teacher, for example nemotron_pi_trace.",
    )
    parser.add_argument(
        "--keep_unverified_traces",
        action="store_true",
        help="Also sample rows whose trace has a final answer that the verifier rejected.",
    )
    args = parser.parse_args()
    if args.train_size < 0 or args.dev_size < 0:
        parser.error("--train_size and --dev_size must not be negative")

    rows = [row for path in args.input_jsonl for row in read_jsonl(path)]
    pool = [
        row
        for row in rows
        if eligible(row, pi_trace_field=args.pi_trace_field, keep_unverified_traces=args.keep_unverified_traces)
    ]
    if len({row_uid(row) for row in pool}) != len(pool):
        # A problem that occurs twice could land in both splits.
        raise SystemExit("The eligible rows contain duplicate problem ids; pass every trace file only once.")
    if len(pool) < args.train_size + args.dev_size:
        raise SystemExit(
            f"Only {len(pool)} of {len(rows)} rows are eligible, fewer than --train_size {args.train_size} "
            f"plus --dev_size {args.dev_size}. Generate traces for more problems or lower the two sizes."
        )

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    train = pool[: args.train_size]
    dev = pool[args.train_size : args.train_size + args.dev_size]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "dev.jsonl", dev)
    write_json(
        output_dir / "split_manifest.json",
        {
            "seed": args.seed,
            "input_jsonl": [str(path) for path in args.input_jsonl],
            "pi_trace_field": args.pi_trace_field,
            "keep_unverified_traces": args.keep_unverified_traces,
            "counts": {
                "input": len(rows),
                "eligible": len(pool),
                "train": len(train),
                "dev": len(dev),
            },
        },
    )


if __name__ == "__main__":
    main()
