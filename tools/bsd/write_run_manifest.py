#!/usr/bin/env python3
"""Write a lightweight run manifest for a Bridged Self-Distillation command."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from verl.trainer.bsd.manifest import collect_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_json", default=None)
    parser.add_argument("--config_kv", action="append", default=[], help="Config key-value entry as key=value.")
    parser.add_argument("--dataset_manifest", default=None)
    parser.add_argument("--model", action="append", default=[], help="Model entry as name=path.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = {}
    if args.config_json:
        config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    for entry in args.config_kv:
        if "=" not in entry:
            raise SystemExit(f"--config_kv must be key=value, got {entry!r}")
        key, value = entry.split("=", 1)
        config[key] = value
    model_paths = {}
    for entry in args.model:
        if "=" not in entry:
            raise SystemExit(f"--model must be name=path, got {entry!r}")
        key, value = entry.split("=", 1)
        model_paths[key] = value
    collect_run_manifest(
        config=config,
        dataset_manifest_path=args.dataset_manifest or None,
        model_paths=model_paths,
        output_path=args.output,
        seed=args.seed,
        # Anchor git provenance to the repo even when invoked from elsewhere.
        cwd=os.environ.get("BSD_REPO_ROOT", "."),
    )


if __name__ == "__main__":
    main()
