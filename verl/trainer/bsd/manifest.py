"""Run manifest collection for reproducible Bridged Self-Distillation runs."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

from verl.trainer.bsd.prompt_assembly import (
    DEFAULT_PI_TRACE_FIELD,
    PI_SOURCE_GROUND_TRUTH,
    PI_SOURCE_TRACE_FIELD,
    answer_teacher_user_template,
    build_answer_teacher_prompt,
    build_pi_teacher_prompt,
    build_student_prompt,
    load_prompt_template,
    normalize_pi_source,
    optional_bool,
    pi_teacher_user_template,
    resolve_emit_think,
    student_user_template,
)

_PROMPT_HASH_PROBLEM = "What is 1 + 1?"
_PROMPT_HASH_PI_TRACE = "Since 1 + 1 = 2, the answer is \\boxed{2}."
_PROMPT_HASH_GROUND_TRUTH = "2"


def sha256_file(path: str | Path) -> str:
    """Hash a file in chunks."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], cwd: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _json_hash(payload: Any) -> str:
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _config_bool(config: dict[str, Any], key: str) -> bool:
    value = config[key]
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Manifest config {key} must be boolean, got {value!r}")


def _prompt_serialization_metadata(tokenizer: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Hash the prompts a run will actually serialize, so a template switch cannot go unnoticed.

    Off-policy SFT has no teacher, so the teacher half is reported only when the run declares
    a teacher thinking mode. Requiring it would leave SFT with no prompt check at all, and an
    SFT run could then silently train on a different student template than the online runs
    it is compared against.
    """

    student_thinking = _config_bool(config, "student_thinking")
    has_teacher = "teacher_thinking" in config
    teacher_thinking = _config_bool(config, "teacher_thinking") if has_teacher else None
    pi_source = normalize_pi_source(config.get("pi_source", PI_SOURCE_TRACE_FIELD))
    emit_think_override = optional_bool(config.get("emit_think"))
    student_emit_think = resolve_emit_think(student_thinking, emit_think_override)
    teacher_emit_think = resolve_emit_think(teacher_thinking, emit_think_override) if has_teacher else None
    student_template = student_user_template(student_emit_think)
    student_prompt = build_student_prompt(
        tokenizer,
        _PROMPT_HASH_PROBLEM,
        enable_thinking=student_thinking,
        emit_think=emit_think_override,
    )
    privileged_context_field = None
    teacher_template = None
    teacher_prompt = None
    if has_teacher:
        if pi_source == PI_SOURCE_GROUND_TRUTH:
            privileged_context_field = "reward_model.ground_truth"
            teacher_template = answer_teacher_user_template(teacher_emit_think)
            teacher_prompt = build_answer_teacher_prompt(
                tokenizer,
                _PROMPT_HASH_PROBLEM,
                _PROMPT_HASH_GROUND_TRUTH,
                enable_thinking=teacher_thinking,
                emit_think=emit_think_override,
            )
        else:
            privileged_context_field = f"extra_info.{config.get('pi_trace_field', DEFAULT_PI_TRACE_FIELD)}"
            template_override = str(config.get("pi_teacher_template") or "") or None
            teacher_template = pi_teacher_user_template(teacher_emit_think, template_override)
            teacher_prompt = build_pi_teacher_prompt(
                tokenizer,
                _PROMPT_HASH_PROBLEM,
                _PROMPT_HASH_PI_TRACE,
                enable_thinking=teacher_thinking,
                emit_think=emit_think_override,
                template=template_override,
            )

    def prompt_metadata(prompt: str) -> dict[str, Any]:
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        return {
            "text_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            "token_ids_sha256": _json_hash(token_ids),
            "token_count": len(token_ids),
        }

    def template_metadata(template_name: str) -> dict[str, Any]:
        return {
            "name": template_name,
            "source_sha256": sha256(load_prompt_template(template_name).encode("utf-8")).hexdigest(),
        }

    payload = {
        "probe_version": 3,
        "student_thinking": student_thinking,
        "teacher_thinking": teacher_thinking,
        "student_emit_think": student_emit_think,
        "teacher_emit_think": teacher_emit_think,
        "emit_think_override": emit_think_override,
        "pi_source": pi_source if has_teacher else None,
        "privileged_context_field": privileged_context_field,
        "student_template": template_metadata(student_template),
        "student_prompt": prompt_metadata(student_prompt),
    }
    if has_teacher:
        payload["teacher_template"] = template_metadata(teacher_template)
        payload["teacher_prompt"] = prompt_metadata(teacher_prompt)
    return payload


def _torch_cuda_metadata() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"error": repr(exc)}

    payload: dict[str, Any] = {
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        payload["devices"] = [
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "capability": torch.cuda.get_device_capability(idx),
            }
            for idx in range(torch.cuda.device_count())
        ]
    return payload


def _model_metadata(model_path: str, *, prompt_config: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"path": model_path}
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        payload["revision"] = getattr(config, "_commit_hash", None)
        payload["model_type"] = getattr(config, "model_type", None)
    except Exception as exc:
        payload["config_error"] = repr(exc)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        special_tokens = getattr(tokenizer, "special_tokens_map", {})
        chat_template = getattr(tokenizer, "chat_template", None)
        all_special_tokens = list(getattr(tokenizer, "all_special_tokens", []) or [])
        special_token_ids = {
            token: int(tokenizer.convert_tokens_to_ids(token))
            for token in all_special_tokens
            if tokenizer.convert_tokens_to_ids(token) is not None
        }
        tokenizer_payload = {
            "name_or_path": getattr(tokenizer, "name_or_path", None),
            "revision": getattr(tokenizer, "_commit_hash", None),
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "special_tokens_map": special_tokens,
            "special_token_ids": special_token_ids,
            "chat_template_sha256": sha256(chat_template.encode("utf-8")).hexdigest() if chat_template else None,
            "chat_template_revision": getattr(tokenizer, "_commit_hash", None),
            "compatibility_hash": _json_hash(
                {
                    "vocab_size": getattr(tokenizer, "vocab_size", None),
                    "bos_token_id": getattr(tokenizer, "bos_token_id", None),
                    "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                    "pad_token_id": getattr(tokenizer, "pad_token_id", None),
                    "special_tokens_map": special_tokens,
                    "chat_template": chat_template,
                }
            ),
        }
        if prompt_config is not None:
            tokenizer_payload["prompt_serialization"] = _prompt_serialization_metadata(tokenizer, prompt_config)
        payload["tokenizer"] = tokenizer_payload
    except Exception as exc:
        if prompt_config is not None:
            raise
        payload["tokenizer_error"] = repr(exc)
    return payload


# Settings that legitimately differ between launches of one experiment: where it
# resumes from, which physical cards it landed on, where it logs. Everything else is
# part of what the run IS. GPU *counts* are deliberately absent -- changing them
# changes per-device batching under use_dynamic_bsz, so that is a different run.
RELAUNCH_VARIABLE_KEYS = frozenset({"resume_mode", "cuda_visible_devices", "trainer_logger"})

OVERWRITE_ENV = "BSD_ALLOW_MANIFEST_OVERWRITE"


def _values_match(a: Any, b: Any) -> bool:
    """Compare two config values across the string/number split.

    Values arriving from ``--config_kv`` are strings while a manifest written from
    Python holds real types, so ``"1e-6"`` and ``1e-06`` are the same setting. An
    absent key and an empty one both mean unset; treating them as different would
    fill the report with noise and push operators toward the override.
    """

    if (a is None or a == "") and (b is None or b == ""):
        return True
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def _guard_existing_manifest(output_path: Path, config: dict[str, Any]) -> None:
    """Refuse to overwrite a manifest that belongs to a differently configured run.

    A manifest is the only record of what a finished run actually did. Without this
    guard, launching with an EXPERIMENT_NAME that already exists would silently replace
    it with the new launch's settings and destroy the provenance of the earlier run.
    """

    try:
        previous = json.loads(output_path.read_text(encoding="utf-8")).get("config") or {}
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"{output_path} exists but could not be read ({exc}); refusing to overwrite it. "
            f"Inspect it, then set {OVERWRITE_ENV}=1 to proceed."
        ) from exc

    differing = sorted(
        key
        for key in set(previous) | set(config)
        if key not in RELAUNCH_VARIABLE_KEYS and not _values_match(previous.get(key), config.get(key))
    )
    if not differing:
        return

    if os.environ.get(OVERWRITE_ENV) == "1":
        # Keep the superseded record rather than destroying it; the whole point of the
        # guard is that this file is irreplaceable.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        superseded = output_path.with_name(f"{output_path.stem}.superseded-{stamp}.json")
        output_path.rename(superseded)
        return

    lines = "\n".join(
        f"  {key}: existing={previous.get(key)!r} -> new={config.get(key)!r}" for key in differing
    )
    raise SystemExit(
        f"{output_path} already records a run configured differently on {len(differing)} "
        f"setting(s):\n{lines}\n"
        "Reusing one experiment name for two configurations destroys the older run's "
        "provenance. Pick a different EXPERIMENT_NAME, or set "
        f"{OVERWRITE_ENV}=1 to supersede the existing manifest (it is renamed, not deleted)."
    )


def collect_run_manifest(
    *,
    config: dict[str, Any],
    output_path: str | Path | None = None,
    dataset_manifest_path: str | Path | None = None,
    model_revisions: dict[str, str | None] | None = None,
    model_paths: dict[str, str] | None = None,
    seed: int | None = None,
    cwd: str | Path = ".",
) -> dict[str, Any]:
    """Collect run provenance into a JSON-serializable manifest."""

    cwd = Path(cwd)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _run(["git", "rev-parse", "HEAD"], cwd=cwd),
            "dirty_state": _run(["git", "status", "--short"], cwd=cwd),
        },
        "packages": {
            "verl": _package_version("verl"),
            "transformers": _package_version("transformers"),
            "vllm": _package_version("vllm"),
            "torch": _package_version("torch"),
            "peft": _package_version("peft"),
        },
        "system": {
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
            "nvidia_smi_gpu_query": _run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "cuda": _torch_cuda_metadata(),
        },
        "model_revisions": model_revisions or {},
        "models": {
            name: _model_metadata(
                path,
                prompt_config=config if name == "student" and "student_thinking" in config else None,
            )
            for name, path in (model_paths or {}).items()
        },
        "seed": seed,
        "config": config,
    }
    if dataset_manifest_path:
        dataset_manifest_path = Path(dataset_manifest_path)
        manifest["dataset_manifest"] = {
            "path": str(dataset_manifest_path),
            "sha256": sha256_file(dataset_manifest_path) if dataset_manifest_path.exists() else None,
        }
    if output_path:
        output_path = Path(output_path)
        if output_path.exists():
            _guard_existing_manifest(output_path, config)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
