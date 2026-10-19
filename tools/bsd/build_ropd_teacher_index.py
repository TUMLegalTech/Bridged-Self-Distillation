#!/usr/bin/env python3
"""Build the offline multi-answer teacher index consumed by official ROPD."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tools.bsd.ropd_teacher_index import (
    load_ropd_rows,
    read_completed_index_keys,
    ropd_resolution_config,
    teacher_fingerprint,
)
from verl.trainer.bsd.judge_usage import UsageRecorder, attach_usage_recorder
from verl.trainer.bsd.prompt_assembly import optional_bool, resolve_emit_think
from verl.trainer.bsd.ropd_vendor import activate_ropd_vendor

# api.openai.com intermittently returns Cloudflare 520/522/524 with
# retry_after=60; the pinned client only retries 408/429/500/502/503/504, so
# the builder retries those bursts itself without touching pinned behavior.
_EXTRA_RETRY_STATUS_CODES = frozenset({520, 521, 522, 524})
_RETRY_ATTEMPTS = 6
_RETRY_INITIAL_BACKOFF_SECONDS = 61.0
_RETRY_MAX_BACKOFF_SECONDS = 480.0


def _generate_with_retry(teacher_client: Any, raw_prompt: Any, uid: str) -> str:
    backoff = _RETRY_INITIAL_BACKOFF_SECONDS
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return str(teacher_client.generate(raw_prompt, uid=uid)).strip()
        except Exception as exc:  # noqa: BLE001 - pinned client error types load lazily
            status = getattr(exc, "status_code", None)
            retriable = bool(getattr(exc, "retriable", False)) or status in _EXTRA_RETRY_STATUS_CODES
            if not retriable or attempt == _RETRY_ATTEMPTS - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, _RETRY_MAX_BACKOFF_SECONDS)
    raise RuntimeError("unreachable")


def _generate_answers(teacher_client: Any, raw_prompt: Any, uid: str, count: int) -> list[str]:
    answers = [_generate_with_retry(teacher_client, raw_prompt, uid) for _ in range(count)]
    if any(not answer for answer in answers):
        raise RuntimeError(f"ROPD teacher returned an empty answer for uid={uid!r}.")
    return answers


def merge_usage_sidecar(
    usage_path: Path, *, model: Any, reasoning_effort: Any, this_run: dict[str, Any]
) -> dict[str, Any]:
    """Append one run's spend to the sidecar, keeping what earlier runs already paid.

    A --resume pass that overwrote this file would erase the first pass's bill, and a
    no-op resume over a finished index would replace a real total with an empty one.
    """

    record: dict[str, Any] = {"model": model, "reasoning_effort": reasoning_effort, "runs": []}
    if usage_path.exists():
        try:
            previous = json.loads(usage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        record["runs"] = list(previous.get("runs", []))
    record["runs"].append(this_run)
    cumulative: dict[str, float] = {}
    for run in record["runs"]:
        for field, value in run.get("usage", {}).get("total", {}).items():
            cumulative[field] = cumulative.get(field, 0) + value
    record["cumulative_usage"] = cumulative
    record["rows_generated_total"] = sum(run.get("rows_generated", 0) for run in record["runs"])
    usage_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def assert_index_matches_teacher(
    index_path: Path, *, fingerprint: dict[str, Any], teacher_answer_count: int
) -> None:
    """Refuse to resume an index that another teacher configuration produced.

    Resume skips a row by uid and prompt hash alone, while the training run rejects an index
    that holds a row with another teacher fingerprint or with fewer answers than it needs.
    Without this check the build would pay for the remaining rows and report success for an
    index that cannot be used.
    """

    from algo.ropd_teacher_index import hash_teacher_fingerprint

    expected_digest = hash_teacher_fingerprint(fingerprint)
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            hash_teacher_fingerprint(row.get("teacher_fingerprint") or {}) != expected_digest
            or len(row.get("teacher_answers") or []) < teacher_answer_count
        ):
            raise SystemExit(
                f"{index_path}:{line_number} was built with another teacher configuration or with "
                f"fewer than {teacher_answer_count} teacher answers; write the index to a new "
                "--output_jsonl (ROPD_TEACHER_INDEX_PATH) instead of resuming this one."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--provider_spec", required=True)
    parser.add_argument("--teacher_answer_count", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--max_concurrency", type=int, default=4)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    # Must match the run that will consume this index: the key is a hash of the student
    # prompt, and the two template variants differ. Empty --emit_think follows the
    # chat-template thinking mode.
    parser.add_argument("--enable_thinking", default="true")
    parser.add_argument("--emit_think", default="")
    args = parser.parse_args()

    emit_think = resolve_emit_think(optional_bool(args.enable_thinking), optional_bool(args.emit_think))

    if args.max_concurrency < 1:
        raise SystemExit("--max_concurrency must be positive.")
    output_path = Path(args.output_jsonl).expanduser().resolve()
    if output_path.exists() and not args.resume:
        raise SystemExit(f"Output already exists; pass --resume to continue: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    activate_ropd_vendor()
    from algo.ropd.client import build_ropd_clients, build_ropd_judge_config
    from algo.ropd_teacher_index import (
        OFFLINE_TEACHER_MULTI_ANSWER_SCHEMA_VERSION,
        hash_canonical_raw_prompt,
        hash_teacher_answers,
    )

    client_config = ropd_resolution_config(
        provider_spec=args.provider_spec,
        entrypoint="build_teacher_index",
        teacher_answer_count=args.teacher_answer_count,
    )
    judge_config = build_ropd_judge_config(client_config)
    teacher_client, _, _ = build_ropd_clients(judge_config)
    # This entrypoint puts the teacher on a live provider (build_teacher_index clears
    # index_path), so it is the one role that bills here. Reasoning tokens dominate that
    # bill and nothing else records them.
    usage_recorder = UsageRecorder()
    attach_usage_recorder(teacher_client, usage_recorder, "teacher")
    fingerprint = teacher_fingerprint(judge_config)
    rows = load_ropd_rows(args.input_parquet, max_rows=args.max_rows, emit_think=emit_think)
    if args.resume and output_path.exists():
        assert_index_matches_teacher(
            output_path, fingerprint=fingerprint, teacher_answer_count=args.teacher_answer_count
        )
    completed = read_completed_index_keys(output_path) if args.resume else set()
    pending = [
        (uid, raw_prompt, hash_canonical_raw_prompt(raw_prompt))
        for uid, raw_prompt in rows
        if (uid, hash_canonical_raw_prompt(raw_prompt)) not in completed
    ]

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            futures = {
                executor.submit(
                    _generate_answers,
                    teacher_client,
                    raw_prompt,
                    uid,
                    args.teacher_answer_count,
                ): (uid, raw_prompt_hash)
                for uid, raw_prompt, raw_prompt_hash in pending
            }
            failed_uids: list[str] = []
            for completed_count, future in enumerate(as_completed(futures), start=1):
                uid, raw_prompt_hash = futures[future]
                # One exhausted row must not abort the run and discard the
                # already-billed results of the remaining futures; failed rows
                # are swept up by a later --resume pass.
                try:
                    answers = future.result()
                except Exception as exc:  # noqa: BLE001
                    failed_uids.append(uid)
                    print(f"FAILED uid={uid}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                    continue
                record = {
                    "schema_version": OFFLINE_TEACHER_MULTI_ANSWER_SCHEMA_VERSION,
                    "uid": uid,
                    "raw_prompt_hash": raw_prompt_hash,
                    "teacher_answers": answers,
                    "teacher_answer_hashes": list(hash_teacher_answers(answers)),
                    "teacher_fingerprint": fingerprint,
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                print(f"Generated {completed_count}/{len(pending)} pending teacher-index rows.", flush=True)

    # A --resume pass must ADD to what earlier passes already paid for. Overwriting would
    # erase the first run's spend, and a no-op resume over a finished index would replace
    # a real total with an empty one.
    usage_path = output_path.with_name(output_path.stem + "_usage.json")
    merge_usage_sidecar(
        usage_path,
        model=fingerprint.get("model"),
        reasoning_effort=fingerprint.get("reasoning_effort"),
        this_run={
            "rows_generated": len(pending) - len(failed_uids),
            "rows_failed": len(failed_uids),
            "usage": usage_recorder.snapshot(),
        },
    )
    if failed_uids:
        raise SystemExit(
            f"ROPD teacher index incomplete: {len(failed_uids)} rows failed "
            f"(e.g. {failed_uids[:3]}); re-run with --resume to sweep them."
        )
    print(f"ROPD teacher index ready: {output_path} ({len(rows)} expected rows)")
    print(f"Recorded API usage: {usage_path}")


if __name__ == "__main__":
    main()
