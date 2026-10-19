#!/usr/bin/env python3
"""Generate the traces of a local external teacher with vLLM.

This is the local-generation entrypoint for the open teachers; the closed-API
teacher goes through generate_closed_teacher_data.py. Each prompt is sampled
--num_attempts times (default 1: one trace per problem) and the shortest attempt
whose final answer the verifier accepts is stored; with --keep_unverified_trace
the first attempt is stored when none verifies.

Sharding and resume: rows are dealt round-robin over --num_workers; each worker
appends durably to its own raw.jsonl and mirrors verified rows into
verified.jsonl, so an interrupted job resumes by rerunning the same command.
A worker skips every row that any worker of the output directory has already
written, so a resume with another number of workers does not generate a row
twice.

The output key names the teacher (--trace_field, e.g. nemotron_pi_trace). With
one fixed key for every teacher a file could not say which model wrote its
traces, so the key is mandatory and the companion keys
(<teacher>_pi_verified, <teacher>_pi_attempts, ...) are derived from it.

The teacher prompt never contains gold information. --prompt_template selects
which project prompt renders the user turn, and --thinking_mode is explicit:
"on" requires the chat template to honor enable_thinking=True and fails when it
cannot (the Qwen3-32B and Nemotron-3-30B-A3B templates both support the option),
while "template_default" declares that a control-less template's own behavior
is intended. There is no silent fallback between the two.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recipes.bsd.rewards.math_verify_reward import extract_boxed_answer, verify_answer
from tools.bsd.common import (
    read_jsonl,
    row_gold_answer,
    row_problem,
    row_uid,
    write_generation_manifest,
)
from tools.bsd.runtime import validate_runtime
from verl.trainer.bsd.prompt_assembly import (
    render_chat_messages,
    render_opsd_user_prompt,
    render_pi_generation_user_prompt,
)

PROMPT_TEMPLATES = {
    "pi_generation": render_pi_generation_user_prompt,
    "opsd": render_opsd_user_prompt,
}

# Named sampling presets. "A" is the Qwen3-32B setting of the paper (temperature 0.6,
# top-p 0.95) and "REASONING" the Nemotron-3-30B-A3B one (temperature 1.0, top-p 1.0).
SAMPLING_CONFIGS = {
    "A": {"temperature": 0.6, "top_p": 0.95},
    "REASONING": {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
    },
}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one durable row so interrupted jobs can resume safely."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_row_uids(path: Path) -> set[str]:
    """Uids of the rows in the raw.jsonl of another worker.

    That worker may be appending while this one starts, so a last line that is not complete
    yet is skipped: its row belongs to the worker that is writing it.
    """

    uids: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            uids.add(row_uid(json.loads(line)))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return uids


# Fields that decide what a trace IS. Resume compares these; anything else (row counts,
# concurrency, timestamps) may legitimately differ between passes over one directory.
RESUME_IDENTITY_FIELDS = ("model", "trace_field", "prompt_template", "thinking_mode", "sampling_config")


def assert_resume_compatible(manifest_path: Path, raw_path: Path, parameters: dict[str, Any]) -> None:
    """Refuse to append to a directory that a different generation configuration produced.

    Resume skips rows by uid alone, and the manifest is rewritten with the current
    settings before any row is read. Together that means pointing a second teacher at
    the first teacher's output directory reports every row as already done and relabels
    the old traces with the new model -- a silent swap that produces a file whose
    manifest is a lie about most of its contents. Nothing downstream can detect it,
    because the traces themselves carry no provenance.
    """

    if not raw_path.exists() or not manifest_path.exists():
        return
    try:
        previous = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    old_parameters = previous.get("parameters") or {}
    mismatched = {
        field: {"existing": old_parameters.get(field), "requested": parameters.get(field)}
        for field in RESUME_IDENTITY_FIELDS
        if field in old_parameters and old_parameters[field] != parameters.get(field)
    }
    if mismatched:
        raise SystemExit(
            f"{raw_path} was produced by a different configuration: "
            f"{json.dumps(mismatched, sort_keys=True)}. "
            "Point --output_dir at a fresh directory instead of resuming into this one."
        )


def build_prompt(tokenizer: Any, prompt_template: str, thinking_mode: str, problem: str) -> str:
    """Build the teacher prompt without gold information.

    thinking_mode "on" goes through render_chat_messages, which raises when the
    tokenizer cannot honor enable_thinking=True; "template_default" applies the
    chat template without the option, declaring the template's own behavior as
    the intended one.
    """

    messages = [{"role": "user", "content": PROMPT_TEMPLATES[prompt_template](problem)}]
    if thinking_mode == "on":
        return render_chat_messages(tokenizer, messages, enable_thinking=True)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def evaluate_attempts(candidates: list[tuple[str, list[int], str]], answer: str,
                      max_new_tokens: int) -> list[dict[str, Any]]:
    """Verify each (text, token_ids, finish_reason) candidate against the answer."""

    attempts = []
    for idx, (text, token_ids, finish_reason) in enumerate(candidates):
        boxed = extract_boxed_answer(text)
        attempts.append(
            {
                "attempt_index": idx,
                "text": text,
                "boxed_answer": boxed,
                "verified": bool(verify_answer(boxed, answer)),
                "finish_reason": finish_reason,
                "token_count": len(token_ids),
                "truncated": finish_reason in {"length", "max_tokens"}
                or len(token_ids) >= max_new_tokens,
            }
        )
    return attempts


def assemble_row(
    row: dict[str, Any],
    attempts: list[dict[str, Any]],
    *,
    trace_field: str,
    keep_unverified_trace: bool,
    mirror_extra_info: bool,
    generation_record: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Attach the selected trace and its provenance to a copy of the source row.

    The stored attempt is the shortest verified one; with keep_unverified_trace
    the first attempt is stored even when nothing verified (wrong traces are
    then kept and flagged, so the correctness rate stays measurable). Returns the
    new row and whether it belongs in verified.jsonl (verified, non-empty, and
    not truncated).
    """

    verified = [attempt for attempt in attempts if attempt["verified"]]
    selected = min(verified, key=lambda item: item["token_count"]) if verified else None
    stored = selected if selected is not None else (attempts[0] if keep_unverified_trace else None)
    prefix = trace_field.removesuffix("_pi_trace")

    new_row = dict(row)
    new_row[trace_field] = None if stored is None else stored["text"]
    new_row[f"{prefix}_pi_verified"] = selected is not None
    new_row[f"{prefix}_pi_boxed_answer"] = (stored or {}).get("boxed_answer") or ""
    new_row[f"{prefix}_pi_attempts"] = len(attempts)
    new_row["format_pass"] = bool(stored and stored["boxed_answer"] is not None)
    new_row["truncated"] = bool(stored and stored["truncated"])
    new_row["finish_reason"] = "" if stored is None else str(stored["finish_reason"])
    new_row[f"pi_trace_tokens_{prefix}"] = 0 if stored is None else stored["token_count"]
    new_row.setdefault("metadata", {})
    new_row["metadata"][f"{prefix}_pi_generation"] = {**generation_record, "attempts": attempts}
    if mirror_extra_info:
        extra_info = dict(new_row.get("extra_info") or {})
        extra_info.update(
            {
                trace_field: new_row[trace_field],
                f"{prefix}_pi_verified": new_row[f"{prefix}_pi_verified"],
                f"{prefix}_pi_boxed_answer": new_row[f"{prefix}_pi_boxed_answer"],
                f"pi_trace_tokens_{prefix}": new_row[f"pi_trace_tokens_{prefix}"],
            }
        )
        new_row["extra_info"] = extra_info
    is_verified = selected is not None and bool(new_row[trace_field]) and not new_row["truncated"]
    return new_row, is_verified


def resolve_sampling(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, Any]:
    """One source of sampling truth: a named preset or the explicit flags."""

    explicit = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}
    if args.config_name is not None:
        if any(value is not None for value in explicit.values()):
            parser.error("--config_name and explicit --temperature/--top_p/--top_k are exclusive")
        return dict(SAMPLING_CONFIGS[args.config_name])
    defaults = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    return {key: defaults[key] if value is None else value for key, value in explicit.items()}


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    """Convert argparse values to JSON-safe manifest fields."""

    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace_field", required=True, help="e.g. nemotron_pi_trace")
    parser.add_argument("--prompt_template", choices=sorted(PROMPT_TEMPLATES), required=True)
    parser.add_argument("--thinking_mode", choices=("on", "template_default"), required=True,
                        help="'on' fails unless enable_thinking=True is honored; "
                             "'template_default' uses the chat template as-is.")
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--config_name", choices=sorted(SAMPLING_CONFIGS), default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_attempts", type=int, default=1,
                        help="Samples per prompt; the shortest verified one is kept.")
    parser.add_argument("--keep_unverified_trace", action="store_true",
                        help="Store the first attempt in the trace field even when unverified.")
    parser.add_argument("--mirror_extra_info", action="store_true",
                        help="Also copy the trace fields into extra_info for verl parquet flows.")
    parser.add_argument("--target_successes", type=int, default=None,
                        help="Stop this worker once verified.jsonl holds this many rows.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--max_num_seqs", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_input_rows", type=int, default=None)
    parser.add_argument("--enforce_eager", action="store_true")
    args = parser.parse_args()
    if not args.trace_field.endswith("_pi_trace"):
        # The companion keys are derived by stripping this suffix, so a name
        # without it would silently produce a schema no reader looks for.
        parser.error(f"--trace_field must end with _pi_trace; got {args.trace_field!r}")
    sampling_config = resolve_sampling(args, parser)

    validate_runtime("generation", expected_prefix=os.environ.get("BSD_GENERATION_ENV_PREFIX"))
    if not 0 <= args.worker_id < args.num_workers:
        raise SystemExit("worker_id must be in [0, num_workers)")

    worker_dir = args.output_dir / f"worker_{args.worker_id}"
    raw_path = worker_dir / "raw.jsonl"
    verified_path = worker_dir / "verified.jsonl"
    manifest_path = worker_dir / "generation_manifest.json"
    parameters = {**serializable_args(args), "sampling_config": sampling_config}
    # Before the manifest is overwritten, while it still describes what is on disk.
    assert_resume_compatible(manifest_path, raw_path, parameters)
    write_generation_manifest(manifest_path, status="started", model=args.model, parameters=parameters)

    source_rows = read_jsonl(args.input_jsonl)
    if args.max_input_rows is not None:
        source_rows = source_rows[: args.max_input_rows]
    source_rows = [row for index, row in enumerate(source_rows) if index % args.num_workers == args.worker_id]
    raw_rows = read_jsonl(raw_path) if raw_path.exists() else []
    verified_seen = {row_uid(row) for row in read_jsonl(verified_path)} if verified_path.exists() else set()
    prefix = args.trace_field.removesuffix("_pi_trace")
    for row in raw_rows:
        if (
            row.get(f"{prefix}_pi_verified")
            and row.get(args.trace_field)
            and not row.get("truncated")
            and row_uid(row) not in verified_seen
        ):
            append_jsonl(verified_path, row)
            verified_seen.add(row_uid(row))
    # Rows are dealt by index modulo --num_workers, so after a resume with another worker
    # count a row may belong to a worker that did not write it. Deduplicating against the
    # raw files of all workers keeps such a row from being generated a second time.
    seen = {row_uid(row) for row in raw_rows}
    for other_raw_path in sorted(args.output_dir.glob("worker_*/raw.jsonl")):
        if other_raw_path != raw_path:
            seen.update(read_row_uids(other_raw_path))
    remaining = [row for row in source_rows if row_uid(row) not in seen]
    successes = len(verified_seen)

    if remaining and (args.target_successes is None or successes < args.target_successes):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            tensor_parallel_size=1,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
        )
        sampling = SamplingParams(
            **sampling_config,
            n=args.num_attempts,
            max_tokens=args.max_new_tokens,
            seed=args.seed,
        )

        for start in range(0, len(remaining), args.batch_size):
            if args.target_successes is not None and successes >= args.target_successes:
                break
            batch = remaining[start : start + args.batch_size]
            prompts = [
                build_prompt(tokenizer, args.prompt_template, args.thinking_mode, row_problem(row))
                for row in batch
            ]
            outputs = llm.generate(prompts, sampling_params=sampling)
            for row, output in zip(batch, outputs, strict=True):
                answer = row_gold_answer(row)
                attempts = evaluate_attempts(
                    [
                        (candidate.text or "", list(candidate.token_ids or []), str(candidate.finish_reason))
                        for candidate in output.outputs
                    ],
                    answer,
                    args.max_new_tokens,
                )
                new_row, is_verified = assemble_row(
                    row,
                    attempts,
                    trace_field=args.trace_field,
                    keep_unverified_trace=args.keep_unverified_trace,
                    mirror_extra_info=args.mirror_extra_info,
                    generation_record={
                        # Provenance follows --model, so a shard always records
                        # the teacher that generated it.
                        "model": Path(args.model).name,
                        "model_path": args.model,
                        "trace_label": prefix,
                        "prompt_template": args.prompt_template,
                        "thinking_mode": args.thinking_mode,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "worker_id": args.worker_id,
                        "num_workers": args.num_workers,
                        "params": parameters,
                    },
                )
                append_jsonl(raw_path, new_row)
                if is_verified:
                    append_jsonl(verified_path, new_row)
                    successes += 1
            print(
                f"worker={args.worker_id} processed={min(start + len(batch), len(remaining))}"
                f"/{len(remaining)} successes={successes}",
                flush=True,
            )

    write_generation_manifest(manifest_path, status="complete", model=args.model, parameters=parameters)
    print(f"worker={args.worker_id} complete raw={raw_path} verified={verified_path}", flush=True)


if __name__ == "__main__":
    main()
