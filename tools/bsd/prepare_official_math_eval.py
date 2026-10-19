#!/usr/bin/env python3
"""Prepare official math evaluation sets for the Bridged Self-Distillation evaluator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from tools.bsd.common import DATA_ROOT, stable_problem_id, write_json, write_jsonl
from verl.trainer.bsd.prompt_assembly import render_opsd_user_prompt, render_student_user_prompt


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    split: str
    trust_remote_code: bool
    question_id_keys: tuple[str, ...]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "aime24": DatasetSpec(
        name="aime24",
        source="HuggingFaceH4/aime_2024",
        split="train",
        trust_remote_code=False,
        question_id_keys=("id",),
    ),
    "aime25": DatasetSpec(
        name="aime25",
        source="yentinglin/aime_2025",
        split="train",
        trust_remote_code=True,
        question_id_keys=("problem_idx", "id"),
    ),
    "hmmt25": DatasetSpec(
        name="hmmt25",
        source="MathArena/hmmt_feb_2025",
        split="train",
        trust_remote_code=True,
        question_id_keys=("problem_idx",),
    ),
}


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def build_prompt(problem: str, prompt_style: str) -> str:
    if prompt_style == "opsd":
        return render_opsd_user_prompt(problem)
    if prompt_style == "bsd":
        return render_student_user_prompt(problem)
    if prompt_style == "emit_think":
        # The student prompt of Qwen3-4B-Instruct, with the instruction to reason inside the think tags.
        return render_student_user_prompt(problem, emit_think=True)
    raise ValueError(f"Unknown prompt_style={prompt_style!r}")


def normalize_example(
    *,
    dataset_name: str,
    spec: DatasetSpec,
    example: dict[str, Any],
    index: int,
    prompt_style: str,
) -> dict[str, Any]:
    problem = example.get("problem")
    if problem is None or not str(problem).strip():
        raise ValueError(f"{dataset_name} row {index} is missing a problem field")
    answer = example.get("answer")
    if answer is None or not str(answer).strip():
        raise ValueError(f"{dataset_name} row {index} is missing an answer field")
    question_id = _first_present(example, spec.question_id_keys)
    sample_id = f"{dataset_name}:{question_id if question_id is not None else index}"
    problem_text = str(problem)
    answer_text = str(answer)
    return {
        "id": sample_id,
        "problem": problem_text,
        "gold_answer": answer_text,
        "source_solution": "" if example.get("solution") is None else str(example.get("solution")),
        "prompt": [{"role": "user", "content": build_prompt(problem_text, prompt_style)}],
        "data_source": f"official_math/{dataset_name}",
        "reward_model": {"style": "rule", "ground_truth": answer_text},
        "extra_info": {
            "id": sample_id,
            "stable_problem_id": stable_problem_id(problem_text),
            "problem": problem_text,
            "dataset": dataset_name,
            "source_dataset": spec.source,
            "source_split": spec.split,
            "source_question_id": None if question_id is None else str(question_id),
            "prompt_style": prompt_style,
            "original_fields": example,
        },
    }


def write_parquet(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path)
    except Exception:
        from datasets import Dataset

        Dataset.from_list(rows).to_parquet(str(path))


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_rows(spec: DatasetSpec, revision: str | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": spec.split, "trust_remote_code": spec.trust_remote_code}
    if revision:
        kwargs["revision"] = revision
    return [dict(row) for row in load_dataset(spec.source, **kwargs)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(DATA_ROOT / "datasets" / "eval" / "official_math"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_SPECS))
    parser.add_argument("--prompt_style", choices=["opsd", "bsd", "emit_think"], default="opsd")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest: dict[str, Any] = {
        "prompt_style": args.prompt_style,
        "revision": args.revision,
        "datasets": {},
    }
    for dataset_name in args.datasets:
        if dataset_name not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset {dataset_name!r}; choose from {sorted(DATASET_SPECS)}")
        spec = DATASET_SPECS[dataset_name]
        rows = [
            normalize_example(
                dataset_name=dataset_name,
                spec=spec,
                example=example,
                index=index,
                prompt_style=args.prompt_style,
            )
            for index, example in enumerate(load_dataset_rows(spec, args.revision))
        ]
        jsonl_path = output_dir / f"{dataset_name}.jsonl"
        parquet_path = output_dir / f"{dataset_name}.parquet"
        write_jsonl(jsonl_path, rows)
        write_parquet(parquet_path, rows)
        manifest["datasets"][dataset_name] = {
            "source": spec.source,
            "split": spec.split,
            "trust_remote_code": spec.trust_remote_code,
            "rows": len(rows),
            "jsonl": str(jsonl_path),
            "parquet": str(parquet_path),
            "jsonl_sha256": file_sha256(jsonl_path),
            "parquet_sha256": file_sha256(parquet_path),
        }
    write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote official math eval data to {output_dir}")


if __name__ == "__main__":
    main()
