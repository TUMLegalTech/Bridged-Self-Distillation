#!/usr/bin/env python3
"""Fail fast when student and teacher tokenizers cannot share exact token IDs."""

from __future__ import annotations

import argparse

from verl.trainer.bsd.scorer import assert_tokenizer_compatible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student_model", required=True)
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    student = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=args.trust_remote_code)
    teacher = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=args.trust_remote_code)
    assert_tokenizer_compatible(student, teacher)


if __name__ == "__main__":
    main()
