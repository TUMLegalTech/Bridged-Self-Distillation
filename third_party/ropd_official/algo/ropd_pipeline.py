from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol

from algo.ropd_artifacts import BlackOPDArtifactExporter
from algo.ropd_clients import BlackOPDClientError, BlackOPDVerifierScore

RawPrompt = str | tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class BlackOPDRollout:
    batch_index: int
    response_text: str
    response_length: int


@dataclass(frozen=True, slots=True)
class BlackOPDGroup:
    uid: str
    raw_prompt: RawPrompt
    rollouts: tuple[BlackOPDRollout, ...]


@dataclass(frozen=True, slots=True)
class BlackOPDPairResult:
    batch_index: int
    pair_index: int
    teacher_score: float
    student_score: float
    reward: float
    student_win: bool = False
    fallback_used: bool = False
    judge_error: bool = False
    pair_concurrency_limited: bool = False
    verifier_subject_concurrency_limited: bool = False
    error_stage: str = ""
    error_type: str = ""
    error_details: dict[str, Any] = field(default_factory=dict)
    warning_stage: str = ""
    warning_type: str = ""
    warning_details: dict[str, Any] = field(default_factory=dict)


class TeacherClient(Protocol):
    def generate(self, raw_prompt: RawPrompt, *, uid: str | None = None) -> str: ...


class RubricClient(Protocol):
    def generate(
        self,
        raw_prompt: RawPrompt,
        teacher_response: str,
        student_response: str,
        *,
        uid: str | None = None,
        pair_index: int | None = None,
    ) -> Any: ...


class VerifierClient(Protocol):
    def score(
        self,
        raw_prompt: RawPrompt,
        rubric: Any,
        answer: str,
        *,
        uid: str | None = None,
        pair_index: int | None = None,
        subject: str | None = None,
    ) -> Any: ...


def normalize_raw_prompt(raw_prompt: Any) -> RawPrompt:
    if isinstance(raw_prompt, str):
        return raw_prompt

    if hasattr(raw_prompt, "tolist") and not isinstance(raw_prompt, bytes | bytearray):
        raw_prompt = raw_prompt.tolist()

    if isinstance(raw_prompt, tuple):
        raw_prompt = list(raw_prompt)

    if isinstance(raw_prompt, list):
        normalized_messages: list[dict[str, Any]] = []
        for message in raw_prompt:
            if not isinstance(message, Mapping):
                raise TypeError(f"raw_prompt messages must be mappings, got {type(message)!r}")
            normalized_messages.append(dict(message))
        return tuple(normalized_messages)

    raise TypeError(f"Unsupported raw_prompt type: {type(raw_prompt)!r}")


def canonicalize_raw_prompt(raw_prompt: Any) -> str:
    return json.dumps(normalize_raw_prompt(raw_prompt), ensure_ascii=False, sort_keys=True)


class BlackOPDPipeline:
    def __init__(
        self,
        *,
        teacher_client: TeacherClient,
        rubric_client: RubricClient,
        verifier_client: VerifierClient,
        max_pair_concurrency: int = 1,
        max_verifier_subject_concurrency: int = 1,
        artifact_exporter: BlackOPDArtifactExporter | None = None,
    ) -> None:
        self.teacher_client = teacher_client
        self.rubric_client = rubric_client
        self.verifier_client = verifier_client
        self.max_pair_concurrency = max(1, int(max_pair_concurrency))
        self.max_verifier_subject_concurrency = max(1, int(max_verifier_subject_concurrency))
        self.artifact_exporter = artifact_exporter
        self._retry_rubric_cache: dict[tuple[str, int, str, str], Any] = {}
        self._retry_rubric_cache_lock = Lock()

    def reset_retry_state(self) -> None:
        with self._retry_rubric_cache_lock:
            self._retry_rubric_cache.clear()

    def snapshot_runtime_metrics(self) -> dict[str, Any]:
        provider_snapshots: dict[int, dict[str, Any]] = {}
        stage_providers = {
            "teacher": getattr(self.teacher_client, "provider", None),
            "rubricator": getattr(self.rubric_client, "provider", None),
            "verifier": getattr(self.verifier_client, "provider", None),
        }

        for provider in stage_providers.values():
            snapshot_fn = getattr(provider, "snapshot_metrics", None)
            if provider is None or not callable(snapshot_fn):
                continue
            provider_snapshots.setdefault(id(provider), snapshot_fn())

        if not provider_snapshots:
            return {}
        if len(provider_snapshots) == 1:
            return next(iter(provider_snapshots.values()))

        return {
            f"{stage}_provider": provider_snapshots[id(provider)]
            for stage, provider in stage_providers.items()
            if provider is not None and id(provider) in provider_snapshots
        }

    def evaluate_group(self, group: BlackOPDGroup) -> tuple[BlackOPDPairResult, ...]:
        _, pair_results = self.evaluate_selected_pairs(group=group)
        return pair_results

    def evaluate_selected_pairs(
        self,
        *,
        group: BlackOPDGroup,
        pair_items: tuple[tuple[int, BlackOPDRollout], ...] | None = None,
        teacher_response: str | None = None,
        max_pair_concurrency: int | None = None,
    ) -> tuple[str | None, tuple[BlackOPDPairResult, ...]]:
        resolved_pair_items = tuple(enumerate(group.rollouts)) if pair_items is None else tuple(pair_items)
        resolved_pair_concurrency = (
            self.max_pair_concurrency if max_pair_concurrency is None else max(1, int(max_pair_concurrency))
        )
        pair_concurrency_limited = len(group.rollouts) > resolved_pair_concurrency
        verifier_subject_concurrency_limited = self.max_verifier_subject_concurrency < 2
        resolved_teacher_response = teacher_response

        if resolved_teacher_response is None:
            try:
                resolved_teacher_response = self.teacher_client.generate(group.raw_prompt, uid=group.uid)
            except BlackOPDClientError as exc:
                fallback_results = []
                for pair_index, rollout in resolved_pair_items:
                    pair_result = self._build_fallback_pair_result(
                        rollout=rollout,
                        pair_index=pair_index,
                        error_stage=exc.stage,
                        error_type=exc.error_type,
                        error_details=self._error_details_from_client_error(exc),
                        pair_concurrency_limited=pair_concurrency_limited,
                        verifier_subject_concurrency_limited=verifier_subject_concurrency_limited,
                    )
                    self._maybe_export_pair_artifacts(
                        group=group,
                        rollout=rollout,
                        pair_result=pair_result,
                        rubric=None,
                        teacher_response=None,
                        teacher_result=None,
                        student_result=None,
                    )
                    fallback_results.append(pair_result)
                return None, tuple(fallback_results)

        if resolved_pair_concurrency == 1 or len(resolved_pair_items) <= 1:
            pair_results = tuple(
                self._evaluate_pair(
                    group=group,
                    teacher_response=resolved_teacher_response,
                    pair_index=pair_index,
                    rollout=rollout,
                    max_pair_concurrency=resolved_pair_concurrency,
                )
                for pair_index, rollout in resolved_pair_items
            )
            return resolved_teacher_response, pair_results

        max_workers = min(resolved_pair_concurrency, len(resolved_pair_items))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pair_results = tuple(
                executor.map(
                    lambda pair_item: self._evaluate_pair(
                        group=group,
                        teacher_response=resolved_teacher_response,
                        pair_index=pair_item[0],
                        rollout=pair_item[1],
                        max_pair_concurrency=resolved_pair_concurrency,
                    ),
                    resolved_pair_items,
                )
            )
        return resolved_teacher_response, pair_results

    def _evaluate_pair(
        self,
        *,
        group: BlackOPDGroup,
        teacher_response: str,
        pair_index: int,
        rollout: BlackOPDRollout,
        max_pair_concurrency: int | None = None,
    ) -> BlackOPDPairResult:
        rubric: Any | None = None
        resolved_pair_concurrency = self.max_pair_concurrency if max_pair_concurrency is None else max_pair_concurrency
        pair_concurrency_limited = len(group.rollouts) > resolved_pair_concurrency
        verifier_subject_concurrency_limited = self.max_verifier_subject_concurrency < 2
        try:
            rubric = self._get_cached_rubric(
                uid=group.uid,
                pair_index=pair_index,
                teacher_response=teacher_response,
                student_response=rollout.response_text,
            )
            if rubric is None:
                rubric = self.rubric_client.generate(
                    group.raw_prompt,
                    teacher_response,
                    rollout.response_text,
                    uid=group.uid,
                    pair_index=pair_index,
                )
                self._store_cached_rubric(
                    uid=group.uid,
                    pair_index=pair_index,
                    teacher_response=teacher_response,
                    student_response=rollout.response_text,
                    rubric=rubric,
                )
            teacher_result, student_result = self._score_teacher_and_student(
                group=group,
                rubric=rubric,
                teacher_response=teacher_response,
                rollout=rollout,
                pair_index=pair_index,
            )
        except BlackOPDClientError as exc:
            pair_result = self._build_fallback_pair_result(
                rollout=rollout,
                pair_index=pair_index,
                error_stage=exc.stage,
                error_type=exc.error_type,
                error_details=self._error_details_from_client_error(exc),
                pair_concurrency_limited=pair_concurrency_limited,
                verifier_subject_concurrency_limited=verifier_subject_concurrency_limited,
            )
            self._maybe_export_pair_artifacts(
                group=group,
                rollout=rollout,
                pair_result=pair_result,
                rubric=rubric,
                teacher_response=teacher_response,
                teacher_result=None,
                student_result=None,
            )
            return pair_result

        teacher_score = float(teacher_result.final_score)
        student_score = float(student_result.final_score)
        reward_scale = self._resolve_reward_scale(rubric)
        pair_result = BlackOPDPairResult(
            batch_index=rollout.batch_index,
            pair_index=pair_index,
            teacher_score=teacher_score,
            student_score=student_score,
            reward=(student_score - teacher_score) / reward_scale,
            student_win=student_score > teacher_score,
            pair_concurrency_limited=pair_concurrency_limited,
            verifier_subject_concurrency_limited=verifier_subject_concurrency_limited,
        )
        self._maybe_export_pair_artifacts(
            group=group,
            rollout=rollout,
            pair_result=pair_result,
            rubric=rubric,
            teacher_response=teacher_response,
            teacher_result=teacher_result,
            student_result=student_result,
        )
        return pair_result

    def _score_teacher_and_student(
        self,
        *,
        group: BlackOPDGroup,
        rubric: Any,
        teacher_response: str,
        rollout: BlackOPDRollout,
        pair_index: int,
    ) -> tuple[BlackOPDVerifierScore, BlackOPDVerifierScore]:
        if self.max_verifier_subject_concurrency == 1:
            return (
                self._score_single_subject(
                    group=group,
                    rubric=rubric,
                    answer=teacher_response,
                    pair_index=pair_index,
                    subject="teacher",
                ),
                self._score_single_subject(
                    group=group,
                    rubric=rubric,
                    answer=rollout.response_text,
                    pair_index=pair_index,
                    subject="student",
                ),
            )

        with ThreadPoolExecutor(max_workers=min(2, self.max_verifier_subject_concurrency)) as executor:
            teacher_future = executor.submit(
                self._score_single_subject,
                group=group,
                rubric=rubric,
                answer=teacher_response,
                pair_index=pair_index,
                subject="teacher",
            )
            student_future = executor.submit(
                self._score_single_subject,
                group=group,
                rubric=rubric,
                answer=rollout.response_text,
                pair_index=pair_index,
                subject="student",
            )
            return teacher_future.result(), student_future.result()

    def _score_single_subject(
        self,
        *,
        group: BlackOPDGroup,
        rubric: Any,
        answer: str,
        pair_index: int,
        subject: str,
    ) -> BlackOPDVerifierScore:
        return self._coerce_verifier_score(
            self.verifier_client.score(
                group.raw_prompt,
                rubric,
                answer,
                uid=group.uid,
                pair_index=pair_index,
                subject=subject,
            )
        )

    def _coerce_verifier_score(self, value: Any) -> BlackOPDVerifierScore:
        if isinstance(value, BlackOPDVerifierScore):
            return value
        if isinstance(value, int | float):
            return BlackOPDVerifierScore(
                schema_version="ropd.verifier.v1",
                judgement=[],
                final_score=float(value),
            )
        raise TypeError(f"Unsupported verifier score type: {type(value)!r}")

    def _resolve_reward_scale(self, rubric: Any) -> float:
        maximum_score = getattr(rubric, "maximum_score", None)
        if maximum_score is None:
            return 1.0
        reward_scale = float(maximum_score)
        if reward_scale <= 0:
            raise ValueError("rubric maximum_score must be positive to normalize reward.")
        return reward_scale

    def _get_cached_rubric(
        self,
        *,
        uid: str,
        pair_index: int,
        teacher_response: str,
        student_response: str,
    ) -> Any | None:
        cache_key = self._build_retry_rubric_cache_key(
            uid=uid,
            pair_index=pair_index,
            teacher_response=teacher_response,
            student_response=student_response,
        )
        with self._retry_rubric_cache_lock:
            return self._retry_rubric_cache.get(cache_key)

    def _store_cached_rubric(
        self,
        *,
        uid: str,
        pair_index: int,
        teacher_response: str,
        student_response: str,
        rubric: Any,
    ) -> None:
        cache_key = self._build_retry_rubric_cache_key(
            uid=uid,
            pair_index=pair_index,
            teacher_response=teacher_response,
            student_response=student_response,
        )
        with self._retry_rubric_cache_lock:
            self._retry_rubric_cache[cache_key] = rubric

    @staticmethod
    def _build_retry_rubric_cache_key(
        *,
        uid: str,
        pair_index: int,
        teacher_response: str,
        student_response: str,
    ) -> tuple[str, int, str, str]:
        return (
            uid,
            pair_index,
            hashlib.sha256(teacher_response.encode("utf-8")).hexdigest(),
            hashlib.sha256(student_response.encode("utf-8")).hexdigest(),
        )

    def _maybe_export_pair_artifacts(
        self,
        *,
        group: BlackOPDGroup,
        rollout: BlackOPDRollout,
        pair_result: BlackOPDPairResult,
        rubric: Any,
        teacher_response: str | None,
        teacher_result: BlackOPDVerifierScore | None,
        student_result: BlackOPDVerifierScore | None,
    ) -> None:
        if self.artifact_exporter is None:
            return
        rubric_model = getattr(getattr(self.rubric_client, "role_config", None), "model", "") or type(
            self.rubric_client
        ).__name__
        self.artifact_exporter.record_pair(
            uid=group.uid,
            pair_index=pair_result.pair_index,
            raw_prompt=group.raw_prompt,
            rubric=rubric,
            teacher_answer=teacher_response,
            student_answer=rollout.response_text,
            rubric_model=str(rubric_model),
            teacher_score=pair_result.teacher_score,
            student_score=pair_result.student_score,
            reward_gap=pair_result.reward,
            student_win=pair_result.student_win,
            fallback_used=pair_result.fallback_used,
            judge_error=pair_result.judge_error,
            error_stage=pair_result.error_stage,
            error_type=pair_result.error_type,
            error_details=pair_result.error_details,
            warning_stage=pair_result.warning_stage,
            warning_type=pair_result.warning_type,
            warning_details=pair_result.warning_details,
            teacher_verifier_judgement=None if teacher_result is None else list(teacher_result.judgement),
            student_verifier_judgement=None if student_result is None else list(student_result.judgement),
        )

    def _build_fallback_pair_result(
        self,
        *,
        rollout: BlackOPDRollout,
        pair_index: int,
        error_stage: str,
        error_type: str,
        error_details: dict[str, Any] | None = None,
        pair_concurrency_limited: bool = False,
        verifier_subject_concurrency_limited: bool = False,
    ) -> BlackOPDPairResult:
        return BlackOPDPairResult(
            batch_index=rollout.batch_index,
            pair_index=pair_index,
            teacher_score=0.0,
            student_score=0.0,
            reward=0.0,
            fallback_used=True,
            judge_error=True,
            pair_concurrency_limited=pair_concurrency_limited,
            verifier_subject_concurrency_limited=verifier_subject_concurrency_limited,
            error_stage=error_stage,
            error_type=error_type,
            error_details=dict(error_details or {}),
        )

    @staticmethod
    def _error_details_from_client_error(exc: BlackOPDClientError) -> dict[str, Any]:
        error_details = dict(exc.details)
        if exc.status_code is not None:
            error_details.setdefault("status_code", exc.status_code)
        error_details.setdefault("retriable", exc.retriable)
        return error_details


__all__ = [
    "BlackOPDGroup",
    "BlackOPDPairResult",
    "BlackOPDPipeline",
    "BlackOPDRollout",
    "RawPrompt",
    "canonicalize_raw_prompt",
    "normalize_raw_prompt",
]
