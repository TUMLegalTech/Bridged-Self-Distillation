#!/usr/bin/env python3
"""Normalize the OpenThoughts math subset into the Bridged Self-Distillation JSONL schema.

The default source is the mathematical-reasoning subset of OpenThoughts in the 30k release
siyanzhao/Openthoughts_math_30k_opsd. Duplicate problems are dropped, and problems that
match an evaluation problem (--eval_jsonl) are written to contamination_quarantine.jsonl
instead of all_clean.jsonl.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from tools.bsd.common import (
    DATA_ROOT,
    compact_text,
    jaccard,
    ngram_set,
    normalize_problem,
    stable_problem_id,
    write_json,
    write_jsonl,
)

PROBLEM_KEYS = ("problem", "question", "prompt", "Question")
ANSWER_KEYS = ("gold_answer", "answer", "Answer", "final_answer", "target")
SOLUTION_KEYS = ("source_solution", "solution", "COT_Reason", "reasoning", "reference_solution")


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def normalize_row(row: dict[str, Any], dataset_revision: str | None, source_split: str) -> dict[str, Any]:
    problem = _first(row, PROBLEM_KEYS)
    if problem is None:
        raise ValueError(f"Cannot find problem field in row keys={sorted(row)}")
    answer = _first(row, ANSWER_KEYS)
    if answer is None:
        raise ValueError(f"Cannot find answer field for problem={str(problem)[:80]!r}")
    source_solution = _first(row, SOLUTION_KEYS) or ""
    return {
        "id": stable_problem_id(str(problem)),
        "problem": str(problem),
        "gold_answer": str(answer),
        "source_solution": str(source_solution),
        "source_split": source_split,
        "qwen32b_pi_trace": None,
        "qwen32b_pi_verified": False,
        "qwen32b_pi_attempts": 0,
        "closed_teacher_output": None,
        "closed_teacher_verified": False,
        "metadata": {
            "dataset_revision": dataset_revision,
            "original_fields": row,
        },
    }


def load_source(args: argparse.Namespace):
    if args.input_jsonl:
        from tools.bsd.common import read_jsonl

        return read_jsonl(args.input_jsonl)
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split, revision=args.revision)
    return list(dataset)


def is_contaminated(problem: str, eval_problems: list[str], ngram_threshold: float) -> bool:
    norm = normalize_problem(problem)
    compact = compact_text(problem)
    grams = ngram_set(problem)
    for eval_problem in eval_problems:
        if norm == normalize_problem(eval_problem):
            return True
        if compact and compact == compact_text(eval_problem):
            return True
        if jaccard(grams, ngram_set(eval_problem)) >= ngram_threshold:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="siyanzhao/Openthoughts_math_30k_opsd")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--eval_jsonl", action="append", default=[])
    parser.add_argument("--output_dir", default=str(DATA_ROOT / "datasets" / "opsd_clean"))
    parser.add_argument("--ngram_threshold", type=float, default=0.8)
    args = parser.parse_args()

    rows = load_source(args)
    eval_problems = []
    if args.eval_jsonl:
        from tools.bsd.common import read_jsonl

        for path in args.eval_jsonl:
            for row in read_jsonl(path):
                if row.get("problem"):
                    eval_problems.append(str(row["problem"]))

    seen: set[str] = set()
    clean = []
    quarantine = []
    stats = Counter()
    for source in rows:
        try:
            normalized = normalize_row(dict(source), args.revision, args.split)
        except ValueError as exc:
            stats["invalid"] += 1
            message = str(exc)
            if "answer field" in message:
                stats["invalid_missing_answer"] += 1
            elif "problem field" in message:
                stats["invalid_missing_problem"] += 1
            else:
                stats["invalid_other"] += 1
            continue
        if normalized["id"] in seen:
            stats["duplicate"] += 1
            continue
        seen.add(normalized["id"])
        if eval_problems and is_contaminated(normalized["problem"], eval_problems, args.ngram_threshold):
            quarantine.append(normalized)
            stats["contamination_quarantine"] += 1
            continue
        clean.append(normalized)
        stats["clean"] += 1

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "all_clean.jsonl", clean)
    write_jsonl(output_dir / "contamination_quarantine.jsonl", quarantine)
    write_json(
        output_dir / "manifest.json",
        {
            "dataset": args.dataset,
            "revision": args.revision,
            "split": args.split,
            "input_jsonl": args.input_jsonl,
            "eval_jsonl": args.eval_jsonl,
            "eval_problem_count": len(eval_problems),
            "stats": dict(stats),
            "output_files": {
                "all_clean": str(output_dir / "all_clean.jsonl"),
                "contamination_quarantine": str(output_dir / "contamination_quarantine.jsonl"),
            },
        },
    )


if __name__ == "__main__":
    main()
