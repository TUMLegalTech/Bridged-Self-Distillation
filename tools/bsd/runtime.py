#!/usr/bin/env python3
"""Validate the isolated Bridged Self-Distillation runtimes."""

from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping

RUNTIME_SPECS: dict[str, dict[str, str]] = {
    "training": {
        "torch": "2.8.0",
        "transformers": "4.57.3",
        "vllm": "0.11.0",
        "flash-attn": "2.8.3",
    },
    "generation": {
        "torch": "2.9.1",
        "transformers": "4.57.3",
        "vllm": "0.16.0",
    },
}


def default_prefix(role: str) -> Path:
    """Return the canonical environment prefix for a runtime role."""

    env_root = Path(
        os.environ.get("BSD_ENV_ROOT") or str(Path(__file__).resolve().parents[2] / "workspace" / "conda_envs")
    )
    env_name = "BSD" if role == "training" else "BSD-generation"
    return env_root / env_name


def installed_versions(packages: Mapping[str, str]) -> dict[str, str | None]:
    """Read installed versions without importing heavyweight runtime packages."""

    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def validate_runtime(
    role: str,
    *,
    expected_prefix: str | Path | None = None,
    executable: str | Path | None = None,
    versions: Mapping[str, str | None] | None = None,
) -> dict[str, str]:
    """Fail when the active interpreter does not match the requested runtime."""

    if role not in RUNTIME_SPECS:
        raise ValueError(f"Unknown Bridged Self-Distillation runtime role: {role}")

    prefix = Path(expected_prefix) if expected_prefix is not None else default_prefix(role)
    active_executable = Path(executable) if executable is not None else Path(sys.executable)
    resolved_prefix = prefix.expanduser().resolve()
    resolved_executable = active_executable.expanduser().resolve()
    if not resolved_executable.is_relative_to(resolved_prefix):
        raise RuntimeError(
            "Bridged Self-Distillation "
            f"{role} runtime requires Python under {resolved_prefix}, got {resolved_executable}."
        )

    expected = RUNTIME_SPECS[role]
    actual = dict(versions) if versions is not None else installed_versions(expected)
    mismatches = [
        f"{package}={actual.get(package) or 'not-installed'} (expected {expected_version})"
        for package, expected_version in expected.items()
        if actual.get(package) != expected_version
    ]
    if mismatches:
        raise RuntimeError(
            f"Bridged Self-Distillation {role} runtime version mismatch: {', '.join(mismatches)}"
        )

    return {package: str(actual[package]) for package in expected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=sorted(RUNTIME_SPECS), required=True)
    parser.add_argument("--expected-prefix", default=None)
    args = parser.parse_args()

    try:
        versions = validate_runtime(args.role, expected_prefix=args.expected_prefix)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "Bridged Self-Distillation "
        f"{args.role} runtime OK: python={Path(sys.executable).resolve()} versions={versions}"
    )


if __name__ == "__main__":
    main()
