#!/usr/bin/env python3
"""Wait until the cross-vocabulary teacher RPC endpoint is ready."""

from __future__ import annotations

import argparse
import os
import time

from verl.trainer.bsd.cross_vocab_opd import CrossVocabTeacherClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--wait_seconds", type=float, default=900.0)
    parser.add_argument("--request_timeout_seconds", type=float, default=5.0)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument(
        "--expected_model",
        default=None,
        help="Fail when the server that answers reports another model path than this one.",
    )
    args = parser.parse_args()

    deadline = time.monotonic() + args.wait_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if args.pid is not None:
            try:
                os.kill(args.pid, 0)
            except ProcessLookupError as exc:
                raise SystemExit(f"Teacher process {args.pid} exited before becoming ready") from exc
        try:
            health = CrossVocabTeacherClient(
                args.endpoint,
                timeout_seconds=args.request_timeout_seconds,
            ).health()
            # A healthy answer alone does not show that it comes from the server this run
            # started: another server may already hold the endpoint.
            if args.expected_model is not None and health.get("model_path") != args.expected_model:
                raise SystemExit(
                    f"The teacher server at {args.endpoint} serves {health.get('model_path')!r}, "
                    f"expected {args.expected_model!r}; another server holds the endpoint."
                )
            print(
                "Cross-vocabulary teacher ready: "
                f"model={health.get('model_path')} max_model_len={health.get('max_model_len')}"
            )
            return
        except (RuntimeError, TimeoutError, ValueError) as exc:
            last_error = exc
            time.sleep(2.0)
    raise SystemExit(f"Teacher did not become ready within {args.wait_seconds:g}s: {last_error}")


if __name__ == "__main__":
    main()
