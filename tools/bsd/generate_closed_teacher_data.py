#!/usr/bin/env python3
"""Generate closed-teacher traces through an OpenAI-compatible API.

One generation per row from the same plain prompt as the open teachers; the
response is stored as the endpoint returns it, and the converters
jsonl_to_verl_parquet.py and build_teacher_sft_parquet.py build the trace from
it by default (the derivation in a reasoning block, the boxed final answer
after it; closed_teacher_trace.py). Wrong traces are kept (the
correctness rate is recorded), and the run is resumable via per-row appends
keyed on the stable row uid; an output file that was written with another
model or other request settings is refused, not resumed. The defaults are
reasoning effort medium, at most 8,192 completion tokens, and the provider's
default sampling. The endpoint, the key and the model id come from
CLOSED_TEACHER_BASE_URL, CLOSED_TEACHER_API_KEY and CLOSED_TEACHER_MODEL.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recipes.bsd.rewards.math_verify_reward import extract_boxed_answer, verify_answer
from tools.bsd.common import (
    read_jsonl,
    row_gold_answer,
    row_problem,
    stable_problem_id,
    write_generation_manifest,
)
from verl.trainer.bsd.judge_usage import UsageRecorder, usage_of
from verl.trainer.bsd.prompt_assembly import (
    CLOSED_TEACHER_USER_TEMPLATE,
    prompt_template_sha256,
    render_closed_teacher_user_prompt,
)

API_RETRY_STATUS_HINTS = ("rate limit", "timeout", "overloaded", "temporarily", "connection")
# Request settings that decide what a stored response is. Resume compares them, and the
# model, with the manifest of the earlier pass.
RESUME_REQUEST_FIELDS = ("max_completion_tokens", "temperature", "top_p", "reasoning_effort")


def row_uid(row: dict[str, Any]) -> str:
    """Resume-dedup key for closed-teacher outputs.

    This chain (the uid fields, then a problem hash -- never the id tier) is
    the contract of existing output files: rows without a uid are deduplicated
    by problem hash, so adopting common.row_uid's id tier would make a resumed
    run pay for the same rows again. Keep it local and stable.
    """

    extra_info = row.get("extra_info") or {}
    uid = extra_info.get("uid") or row.get("uid")
    if uid:
        return str(uid)
    return stable_problem_id(row_problem(row))


def read_completed_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                done.add(row_uid(json.loads(line)))
    return done


def call_teacher(client: Any, request_kwargs: dict[str, Any], *, api_retries: int, backoff: float) -> Any:
    last_error: Exception | None = None
    for attempt in range(api_retries + 1):
        try:
            return client.chat.completions.create(**request_kwargs)
        except Exception as exc:  # noqa: BLE001 - SDK raises many transport error types
            last_error = exc
            status = getattr(exc, "status_code", None)
            retriable = status in (408, 409, 429, 500, 502, 503, 504) or (
                status is None and any(hint in str(exc).lower() for hint in API_RETRY_STATUS_HINTS)
            )
            if not retriable or attempt == api_retries:
                raise
            time.sleep(min(backoff * (2**attempt), 60.0))
    raise last_error  # unreachable; keeps type checkers honest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--max_rows", type=int, default=None)
    # 8192 is the trace budget of every external teacher; note that reasoning
    # tokens share this budget on reasoning models.
    parser.add_argument("--max_completion_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=None, help="Omitted from the request when unset.")
    parser.add_argument("--top_p", type=float, default=None, help="Omitted from the request when unset.")
    parser.add_argument(
        "--reasoning_effort",
        default="medium",
        help=(
            "Passed to the API verbatim (e.g. none/low/medium/high) and recorded in every row. "
            "An empty value omits the parameter from the request."
        ),
    )
    parser.add_argument("--api_retries", type=int, default=4)
    parser.add_argument("--api_retry_backoff", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.0, help="Fixed delay between rows.")
    args = parser.parse_args()
    # An empty value means the endpoint takes no reasoning-effort parameter.
    args.reasoning_effort = args.reasoning_effort or None

    base_url = os.environ.get("CLOSED_TEACHER_BASE_URL")
    api_key = os.environ.get("CLOSED_TEACHER_API_KEY")
    model = os.environ.get("CLOSED_TEACHER_MODEL")
    if not base_url or not api_key or not model:
        raise SystemExit("Set CLOSED_TEACHER_BASE_URL, CLOSED_TEACHER_API_KEY, and CLOSED_TEACHER_MODEL.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the openai package to generate closed-teacher data.") from exc

    client = OpenAI(base_url=base_url, api_key=api_key)
    rows = read_jsonl(args.input_jsonl)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_name(output_path.stem + "_generation_manifest.json")
    manifest_parameters = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "max_rows": args.max_rows,
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "reasoning_effort": args.reasoning_effort,
        "api_retries": args.api_retries,
        "protocol": "single_generation_keep_wrong_traces",
        "prompt_template": CLOSED_TEACHER_USER_TEMPLATE,
        "prompt_template_sha256": prompt_template_sha256(CLOSED_TEACHER_USER_TEMPLATE),
    }
    # Resume appends to an existing jsonl, skipping rows by uid only. If the template
    # changed since the first pass, the two halves of that one file would answer two
    # different prompts with nothing on disk to tell them apart.
    if output_path.exists() and manifest_path.exists():
        try:
            previous_manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            previous_manifest = {}
        previous = previous_manifest.get("parameters") or {}
        recorded = previous.get("prompt_template_sha256")
        if recorded and recorded != manifest_parameters["prompt_template_sha256"]:
            raise SystemExit(
                f"{output_path} was generated with prompt template {recorded}, but "
                f"{CLOSED_TEACHER_USER_TEMPLATE} now hashes to "
                f"{manifest_parameters['prompt_template_sha256']}. Resuming would mix two "
                "prompts in one file; write to a new --output_jsonl instead."
            )
        # The same holds for the model and the request settings: the manifest is rewritten
        # below with the current ones, and the file would mix two generation settings.
        changed = {
            field: {"existing": previous[field], "requested": manifest_parameters[field]}
            for field in RESUME_REQUEST_FIELDS
            if field in previous and previous[field] != manifest_parameters[field]
        }
        if previous_manifest.get("model") and previous_manifest["model"] != model:
            changed["model"] = {"existing": previous_manifest["model"], "requested": model}
        if changed:
            raise SystemExit(
                f"{output_path} was generated with other settings: {json.dumps(changed, sort_keys=True)}. "
                "Resuming would mix them in one file; write to a new --output_jsonl instead."
            )
    write_generation_manifest(manifest_path, status="started", model=model, parameters=manifest_parameters)

    done = read_completed_uids(output_path)
    pending = [row for row in rows if row_uid(row) not in done]
    print(f"[closed-teacher] input={len(rows)} completed={len(done)} pending={len(pending)}")

    correct = 0
    written = 0
    usage_recorder = UsageRecorder()

    def finalize_manifest(status: str) -> None:
        manifest_parameters["rows_written_this_run"] = written
        manifest_parameters["correct_this_run"] = correct
        manifest_parameters["usage_totals_this_run"] = usage_recorder.snapshot()
        write_generation_manifest(manifest_path, status=status, model=model, parameters=manifest_parameters)

    with output_path.open("a", encoding="utf-8") as handle:
        for row in pending:
            problem = row_problem(row)
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": render_closed_teacher_user_prompt(problem)}],
                "max_completion_tokens": args.max_completion_tokens,
            }
            if args.temperature is not None:
                request_kwargs["temperature"] = args.temperature
            if args.top_p is not None:
                request_kwargs["top_p"] = args.top_p
            if args.reasoning_effort is not None:
                request_kwargs["reasoning_effort"] = args.reasoning_effort
            response = call_teacher(
                client, request_kwargs, api_retries=args.api_retries, backoff=args.api_retry_backoff
            )
            # Bank the spend first. Everything below can raise -- an empty choices list, a
            # verifier that chokes on a malformed answer -- and the call is already paid
            # for by then, so accounting after the fact loses it and a --resume pays twice.
            row_usage = usage_of(response)
            usage_recorder.record("closed_teacher", response)
            choice = response.choices[0]
            text = choice.message.content or ""
            boxed = extract_boxed_answer(text)
            verified = bool(verify_answer(boxed, row_gold_answer(row)))
            new_row = dict(row)
            new_row["closed_teacher_output"] = text
            new_row["closed_teacher_verified"] = verified
            new_row.setdefault("metadata", {})
            new_row["metadata"]["closed_teacher_generation"] = {
                "provider": "openai_compatible",
                "base_url": base_url,
                "model": model,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "request_id": getattr(response, "id", None),
                "reasoning_effort": args.reasoning_effort,
                "generation_config": {
                    "max_completion_tokens": args.max_completion_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
                "boxed_answer": boxed,
                "trace_correct": verified,
                "trace_format_pass": boxed is not None,
                "trace_finish_reason": choice.finish_reason,
                "trace_truncated": choice.finish_reason == "length",
                **row_usage,
            }
            handle.write(json.dumps(new_row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written += 1
            correct += int(verified)
            if written % 25 == 0 or written == len(pending):
                print(f"[closed-teacher] {written}/{len(pending)} correct_rate={correct / written:.3f}")
                # Refresh the running totals. The authoritative per-call record is the
                # fsync'd jsonl row, so this only bounds how stale the manifest summary
                # can be if the process dies mid-run.
                finalize_manifest("running")
            if args.sleep > 0:
                time.sleep(args.sleep)

    finalize_manifest("complete")


if __name__ == "__main__":
    main()
