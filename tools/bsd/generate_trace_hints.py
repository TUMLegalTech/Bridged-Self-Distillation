#!/usr/bin/env python3
"""Hint prompt, answer-leak check, and a local hint generator for teacher traces.

HINT_INSTRUCTION and answer_leaks are shared with generate_pi_forms.py, which generates the hints of BSD-hint
through an API. The command-line entry point is the local alternative: it reads a JSON list of rows (fields uid,
problem, trace, ground_truth), asks a local teacher model (non-thinking mode) for a short hint that names the key
idea without the answer, samples several candidates, and keeps the first one that does not contain the
ground-truth string.
Output: JSONL of {uid, hint, answer_leak, n_candidates}. Run with the generation interpreter.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HINT_INSTRUCTION = """You are helping a student who is stuck on a math problem. Below are the problem and a worked reasoning trace written by a stronger solver.

Problem:
{problem}

Worked reasoning:
{trace}

Write a hint of at most three sentences (under 80 words) that points the student to the key idea or the first decisive step. Rules: do not state the final answer; do not give numerical intermediate results; do not write out the solution; name the method or the observation that unlocks the problem. Output only the hint."""


def _norm(text: str) -> str:
    return re.sub(r"[\s$]", "", text)


def answer_leaks(hint: str, ground_truth: str) -> bool:
    gt = _norm(ground_truth)
    if not gt:
        return False
    if re.fullmatch(r"-?\d+", gt):  # short integers: whole-number match only ("16." at a sentence end still counts)
        return re.search(rf"(?<![\d.]){re.escape(gt)}(?!\d)(?!\.\d)", hint) is not None
    return gt in _norm(hint)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-32B", help="hub id or local directory of the teacher")
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--max-model-len", type=int, default=20480)
    p.add_argument("--seed", type=int, default=43)
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = json.load(open(args.rows))
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": HINT_INSTRUCTION.format(problem=r["problem"], trace=r["trace"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for r in rows
    ]
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.88,
        seed=args.seed,
    )
    sp = SamplingParams(n=args.candidates, temperature=0.7, top_p=0.95, max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate(prompts, sp)
    n_leak = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            cands = [c.text.strip() for c in o.outputs if c.finish_reason == "stop" and c.text.strip()]
            cands = cands or [c.text.strip() for c in o.outputs]
            clean = [c for c in cands if not answer_leaks(c, r["ground_truth"])]
            hint = clean[0] if clean else min(cands, key=len)
            n_leak += not clean
            f.write(json.dumps({"uid": r["uid"], "hint": hint, "answer_leak": not clean, "n_candidates": len(cands)}) + "\n")
    print(f"hints: {len(rows)} rows -> {args.out}; {n_leak} rows with every candidate leaking the answer")


if __name__ == "__main__":
    main()
