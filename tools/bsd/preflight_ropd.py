#!/usr/bin/env python3
"""Validate ROPD credentials, provider config, and full teacher-index coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.bsd.ropd_teacher_index import (
    load_ropd_rows,
    ropd_resolution_config,
    set_teacher_index_environment,
)
from verl.trainer.bsd.prompt_assembly import optional_bool, resolve_emit_think
from verl.trainer.bsd.ropd_vendor import activate_ropd_vendor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_parquet", required=True)
    parser.add_argument("--teacher_index", required=True)
    parser.add_argument("--provider_spec", required=True)
    parser.add_argument("--teacher_answer_count", type=int, default=4, choices=range(1, 5))
    # The student prompt variant, forwarded from the recipe rather than assumed, because the
    # index key is a hash of that prompt. Empty --emit_think means "follow the chat-template
    # thinking mode", the same tri-state the launcher and the dataset use.
    parser.add_argument("--enable_thinking", default="true")
    parser.add_argument("--emit_think", default="")
    args = parser.parse_args()

    emit_think = resolve_emit_think(optional_bool(args.enable_thinking), optional_bool(args.emit_think))

    set_teacher_index_environment(args.teacher_index)
    activate_ropd_vendor()
    from algo.ropd.client import build_ropd_clients, build_ropd_judge_config

    client_config = ropd_resolution_config(
        provider_spec=args.provider_spec,
        entrypoint="train",
        teacher_answer_count=args.teacher_answer_count,
    )
    judge_config = build_ropd_judge_config(client_config)
    teacher_client, _, _ = build_ropd_clients(judge_config)
    rows = load_ropd_rows(args.train_parquet, emit_think=emit_think)
    for uid, raw_prompt in rows:
        answers = teacher_client.generate_many(
            raw_prompt,
            uid=uid,
            count=args.teacher_answer_count,
        )
        if len(answers) != args.teacher_answer_count:
            raise RuntimeError(f"ROPD teacher index returned {len(answers)} answers for uid={uid!r}.")

    print(
        json.dumps(
            {
                "status": "ok",
                "train_parquet": str(Path(args.train_parquet).resolve()),
                "teacher_index": str(Path(args.teacher_index).resolve()),
                "row_count": len(rows),
                "emit_think": emit_think,
                "teacher_answer_count": args.teacher_answer_count,
                "rubricator_model": judge_config.rubricator.model,
                "verifier_model": judge_config.verifier.model,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
