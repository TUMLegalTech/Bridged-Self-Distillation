#!/usr/bin/env python3
"""Aggregate Bridged Self-Distillation result JSON files into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "run",
    "checkpoint",
    "dataset",
    "avg",
    "majority",
    "pass",
    "greedy_accuracy",
    "format_rate",
    "think_open_rate",
    "think_close_rate",
    "think_wellformed_rate",
    "think_open_first_token_rate",
    "think_duplicate_tag_rate",
    "truncation_rate",
    "mean_response_length",
    "group_variance",
    "all_correct_rate",
    "all_wrong_rate",
    "prompt_tokens",
    "completion_tokens",
    "total_samples",
    "privacy_leakage_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    rows = []
    for path in args.input:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        row = {field: payload.get(field) for field in FIELDS}
        row["run"] = row["run"] or payload.get("method") or Path(path).stem
        rows.append(row)

    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["|" + "|".join(FIELDS) + "|", "|" + "|".join(["---"] * len(FIELDS)) + "|"]
    for row in rows:
        lines.append("|" + "|".join("" if row.get(field) is None else str(row[field]) for field in FIELDS) + "|")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
