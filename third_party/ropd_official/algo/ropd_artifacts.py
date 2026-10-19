from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from algo.ropd_prompts import PROMPT_TEMPLATE_VERSION


@dataclass(frozen=True, slots=True)
class BlackOPDExportConfig:
    enabled: bool = False
    output_dir: str = "outputs/ropd"
    retention_days: int = 14
    text_artifact_mode: str = "diagnostic_only"


class BlackOPDArtifactExporter:
    MARKER_FILENAME = ".ropd_run"
    _TEXT_ARTIFACT_MODES = frozenset({"diagnostic_only", "all_pairs"})

    def __init__(self, config: BlackOPDExportConfig) -> None:
        self.config = config
        if self.config.text_artifact_mode not in self._TEXT_ARTIFACT_MODES:
            raise ValueError(
                "text_artifact_mode must be one of "
                f"{sorted(self._TEXT_ARTIFACT_MODES)!r}, got {self.config.text_artifact_mode!r}."
            )
        self.run_dir = Path(config.output_dir)
        self.artifacts_dir = self.run_dir / "artifacts"
        self.rubrics_dir = self.artifacts_dir / "rubrics"
        self.trace_path = self.artifacts_dir / "pair_traces.jsonl"
        self.validation_dir = self.run_dir / "validation"
        self.rollout_dir = self.run_dir / "rollout"
        self.marker_path = self.run_dir / self.MARKER_FILENAME
        self._lock = Lock()
        self._retention_seconds = max(int(config.retention_days), 1) * 24 * 60 * 60
        self._cleanup_ran = False

    def record_pair(
        self,
        *,
        uid: str,
        pair_index: int,
        raw_prompt: Any,
        rubric: Any,
        teacher_answer: str | None,
        student_answer: str,
        rubric_model: str,
        teacher_score: float,
        student_score: float,
        reward_gap: float,
        student_win: bool,
        fallback_used: bool,
        judge_error: bool,
        error_stage: str,
        error_type: str,
        error_details: Mapping[str, Any] | None = None,
        warning_stage: str = "",
        warning_type: str = "",
        warning_details: Mapping[str, Any] | None = None,
        teacher_verifier_judgement: list[bool] | None = None,
        student_verifier_judgement: list[bool] | None = None,
    ) -> None:
        if not self.config.enabled:
            return

        rubric_hash = getattr(rubric, "rubric_hash", None)
        trace_entry = {
            "uid": uid,
            "pair_index": pair_index,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "raw_prompt_hash": self._hash_json(raw_prompt),
            "teacher_answer_hash": self._hash_optional_text(teacher_answer),
            "student_answer_hash": self._hash_text(student_answer),
            "rubric_hash": rubric_hash,
            "rubric_model": rubric_model,
            "teacher_score": teacher_score,
            "student_score": student_score,
            "reward_gap": reward_gap,
            "student_win": student_win,
            "fallback_used": fallback_used,
            "judge_error": judge_error,
        }

        should_write_error_context = fallback_used or judge_error or bool(error_stage) or bool(error_type)
        should_write_warning_context = bool(warning_stage) or bool(warning_type) or bool(warning_details)
        should_write_diagnostic_context = should_write_error_context or should_write_warning_context
        should_write_text_artifacts = (
            self.config.text_artifact_mode == "all_pairs" or should_write_diagnostic_context
        )
        if should_write_error_context:
            trace_entry["error_stage"] = error_stage
            trace_entry["error_type"] = error_type
            if error_details:
                trace_entry["error_details"] = dict(error_details)
        if should_write_warning_context:
            trace_entry["warning_stage"] = warning_stage
            trace_entry["warning_type"] = warning_type
            if warning_details:
                trace_entry["warning_details"] = dict(warning_details)
        if should_write_text_artifacts:
            trace_entry["raw_prompt"] = raw_prompt
            trace_entry["teacher_answer"] = teacher_answer
            trace_entry["student_answer"] = student_answer
            if teacher_verifier_judgement is not None:
                trace_entry["teacher_verifier_judgement"] = list(teacher_verifier_judgement)
            if student_verifier_judgement is not None:
                trace_entry["student_verifier_judgement"] = list(student_verifier_judgement)

        rubric_payload = None
        if should_write_text_artifacts and hasattr(rubric, "model_dump") and rubric_hash is not None:
            rubric_payload = rubric.model_dump(mode="json")

        with self._lock:
            self._ensure_run_layout()
            self._cleanup_expired_runs_once()

            if rubric_payload is not None:
                rubric_path = self.rubrics_dir / f"{rubric_hash}.json"
                if not rubric_path.exists():
                    rubric_path.write_text(
                        json.dumps(rubric_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

            with self.trace_path.open("a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")

    def _ensure_run_layout(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.rubrics_dir.mkdir(parents=True, exist_ok=True)
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        if not self.marker_path.exists():
            self.marker_path.write_text("", encoding="utf-8")

    def _cleanup_expired_runs_once(self) -> None:
        if self._cleanup_ran:
            return
        self._cleanup_ran = True

        parent_dir = self.run_dir.parent
        if not parent_dir.exists():
            return

        cutoff_timestamp = time.time() - self._retention_seconds
        for candidate in parent_dir.iterdir():
            if candidate == self.run_dir or not candidate.is_dir():
                continue
            marker_path = candidate / self.MARKER_FILENAME
            if not marker_path.exists():
                continue
            try:
                marker_mtime = marker_path.stat().st_mtime
            except OSError:
                continue
            if marker_mtime >= cutoff_timestamp:
                continue
            shutil.rmtree(candidate, ignore_errors=True)

    @staticmethod
    def _hash_json(value: Any) -> str:
        canonical_json = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def _hash_optional_text(cls, text: str | None) -> str | None:
        if text is None:
            return None
        return cls._hash_text(text)


__all__ = ["BlackOPDArtifactExporter", "BlackOPDExportConfig"]
