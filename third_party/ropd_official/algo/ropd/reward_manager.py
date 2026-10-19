import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from algo.ropd.client import (
    ROPD_BATCH_SCHEMA_VERSION,
    RopdAnswerScore,
    RopdJudgeConfig,
    RopdVerifierScores,
    build_ropd_clients,
    build_ropd_judge_config,
    coerce_ropd_teacher_answer_count,
)
from algo.ropd_clients import BlackOPDClientError, BlackOPDStructuredRubric
from algo.ropd_pipeline import BlackOPDGroup, BlackOPDRollout, canonicalize_raw_prompt, normalize_raw_prompt
from verl.protocol import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn

RECOVERABLE_REWARD_ERROR_TYPES = frozenset({"timeout", "http_error", "empty_response", "incomplete"})
STEP_RECOVERABLE_REWARD_ERROR_TYPES = RECOVERABLE_REWARD_ERROR_TYPES | frozenset({"circuit_open"})


@dataclass(frozen=True, slots=True)
class RopdRewardQualityGateConfig:
    enabled: bool = True
    max_fallback_rate: float = 0.4
    max_retry_rounds: int = 1
    retry_pair_concurrency: int = 1
    max_step_judge_retry_attempts: int = 2
    step_judge_retry_initial_backoff_seconds: float = 1.0
    step_judge_retry_backoff_multiplier: float = 2.0
    step_judge_retry_max_backoff_seconds: float = 4.0


@dataclass(frozen=True, slots=True)
class RopdGroupResult:
    student_scores: tuple[float, ...]
    reward_scores: tuple[float, ...]
    judge_error: bool
    fallback_used: bool
    teacher_scores: tuple[float, ...] = ()
    rubric_hash: str = ""
    error_type: str = ""
    error_stage: str = ""
    error_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RopdGroupRecord:
    uid: str
    group: BlackOPDGroup
    result: RopdGroupResult
    retry_count: int = 0
    final_status: str = "success"
    first_error_stage: str = ""
    first_error_type: str = ""


@dataclass(frozen=True, slots=True)
class RopdAnswerItem:
    source: str
    source_index: int
    text: str


class RopdRewardManager(AbstractRewardManager):
    EXTRA_INFO_DEFAULTS = {
        "student_score": None,
        "reward_score": None,
        "judge_error": False,
        "fallback_used": False,
        "group_size": None,
        "rubric_hash": None,
    }

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: RawRewardFn | None,
        reward_fn_key: str = "data_source",
        *,
        teacher_client: Any | None = None,
        rubric_client: Any | None = None,
        verifier_client: Any | None = None,
        ropd: RopdJudgeConfig | dict[str, Any] | None = None,
        client_config: RopdJudgeConfig | dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.client_config = self._resolve_client_config(ropd=ropd, client_config=client_config)
        self.teacher_answer_count = self._resolve_teacher_answer_count(ropd=ropd, client_config=client_config)
        self.verifier_answer_chunk_size = self._resolve_verifier_answer_chunk_size(
            ropd=ropd,
            client_config=client_config,
        )
        self.max_concurrency = self._resolve_max_concurrency(ropd=ropd, client_config=client_config)
        self.reward_quality_gate = self._resolve_reward_quality_gate_config(
            ropd=ropd,
            client_config=client_config,
        )
        self.debug_output_dir = self._resolve_debug_output_dir(ropd=ropd, client_config=client_config)
        self._debug_call_index = 0
        provided_clients = (teacher_client, rubric_client, verifier_client)
        if any(client is None for client in provided_clients):
            if self.client_config is None:
                raise ValueError(
                    "ropd.provider_resolution or explicit teacher/rubricator/verifier clients are required."
                )
            self.teacher_client, self.rubric_client, self.verifier_client = build_ropd_clients(
                self.client_config
            )
        else:
            self.teacher_client = teacher_client
            self.rubric_client = rubric_client
            self.verifier_client = verifier_client
        self._validate_teacher_client_capabilities()

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            if return_dict:
                assert isinstance(reward_from_rm_scores, dict)
                reward_extra_info = reward_from_rm_scores.get("reward_extra_info", {})
                return {
                    "reward_tensor": reward_from_rm_scores["reward_tensor"],
                    "reward_extra_info": {
                        key: reward_extra_info[key] for key in self.EXTRA_INFO_DEFAULTS if key in reward_extra_info
                    },
                    "reward_control": self._build_passthrough_reward_control(batch_size=len(data)),
                }
            return reward_from_rm_scores

        groups = self._build_groups(data)
        group_records = self._evaluate_initial_groups(groups)
        reward_control = self._build_reward_control(group_records, batch_size=len(data))
        self._write_debug_record(group_records=group_records, reward_control=reward_control)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = self._init_reward_extra_info(batch_size=len(data))
        group_results = {record.uid: record.result for record in group_records}
        for group in groups:
            result = group_results[group.uid]
            for student_index, rollout in enumerate(group.rollouts):
                student_score = float(result.student_scores[student_index])
                reward_value = float(result.reward_scores[student_index])
                reward_tensor[rollout.batch_index, rollout.response_length - 1] = reward_value
                reward_extra_info["student_score"][rollout.batch_index] = student_score
                reward_extra_info["reward_score"][rollout.batch_index] = reward_value
                reward_extra_info["judge_error"][rollout.batch_index] = result.judge_error
                reward_extra_info["fallback_used"][rollout.batch_index] = result.fallback_used
                reward_extra_info["group_size"][rollout.batch_index] = len(group.rollouts)
                reward_extra_info["rubric_hash"][rollout.batch_index] = result.rubric_hash

        self._validate_reward_extra_info(reward_extra_info)
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "reward_control": reward_control,
            }
        return reward_tensor

    def _resolve_client_config(
        self,
        *,
        ropd: RopdJudgeConfig | dict[str, Any] | None,
        client_config: RopdJudgeConfig | dict[str, Any] | None,
    ) -> RopdJudgeConfig | None:
        resolved_client_source = ropd if ropd is not None else client_config
        if resolved_client_source is None:
            return None
        if isinstance(resolved_client_source, RopdJudgeConfig):
            return resolved_client_source
        if not isinstance(resolved_client_source, Mapping):
            return None
        if (
            "provider_resolution" not in resolved_client_source
            and "teacher" not in resolved_client_source
            and "rubricator" not in resolved_client_source
            and "verifier" not in resolved_client_source
        ):
            return None
        return build_ropd_judge_config(dict(resolved_client_source))

    def _resolve_teacher_answer_count(
        self,
        *,
        ropd: RopdJudgeConfig | dict[str, Any] | None,
        client_config: RopdJudgeConfig | dict[str, Any] | None,
    ) -> int:
        resolved_client_source = ropd if ropd is not None else client_config
        if isinstance(resolved_client_source, RopdJudgeConfig):
            return coerce_ropd_teacher_answer_count(resolved_client_source.teacher_answer_count)
        if isinstance(self.client_config, RopdJudgeConfig):
            return coerce_ropd_teacher_answer_count(self.client_config.teacher_answer_count)
        if isinstance(resolved_client_source, Mapping) and "teacher_answer_count" in resolved_client_source:
            teacher_answer_count = resolved_client_source["teacher_answer_count"]
        else:
            teacher_answer_count = None
        return coerce_ropd_teacher_answer_count(teacher_answer_count)

    def _resolve_verifier_answer_chunk_size(
        self,
        *,
        ropd: RopdJudgeConfig | dict[str, Any] | None,
        client_config: RopdJudgeConfig | dict[str, Any] | None,
    ) -> int | None:
        resolved_client_source = ropd if ropd is not None else client_config
        if isinstance(resolved_client_source, RopdJudgeConfig):
            return resolved_client_source.verifier_answer_chunk_size
        if isinstance(self.client_config, RopdJudgeConfig):
            return self.client_config.verifier_answer_chunk_size
        if isinstance(resolved_client_source, Mapping):
            value = resolved_client_source.get("verifier_answer_chunk_size")
            if value is None:
                return None
            resolved_value = int(value)
            if resolved_value < 1:
                raise ValueError("ropd.verifier_answer_chunk_size must be at least 1.")
            return resolved_value
        return None

    def _resolve_max_concurrency(
        self,
        *,
        ropd: RopdJudgeConfig | dict[str, Any] | None,
        client_config: RopdJudgeConfig | dict[str, Any] | None,
    ) -> int:
        resolved_client_source = ropd if ropd is not None else client_config
        if isinstance(resolved_client_source, RopdJudgeConfig):
            return resolved_client_source.max_concurrency
        if isinstance(resolved_client_source, Mapping):
            return max(1, int(resolved_client_source.get("max_concurrency", 4)))
        return 4

    def _resolve_reward_quality_gate_config(
        self,
        *,
        ropd: RopdJudgeConfig | dict[str, Any] | None,
        client_config: RopdJudgeConfig | dict[str, Any] | None,
    ) -> RopdRewardQualityGateConfig:
        resolved_client_source = ropd if ropd is not None else client_config
        gate_config: dict[str, Any] = {}
        if isinstance(resolved_client_source, Mapping):
            gate_config = dict(resolved_client_source.get("reward_quality_gate", {}))

        return RopdRewardQualityGateConfig(
            enabled=bool(gate_config.get("enabled", True)),
            max_fallback_rate=float(gate_config.get("max_fallback_rate", 0.4)),
            max_retry_rounds=int(gate_config.get("max_retry_rounds", 1)),
            retry_pair_concurrency=max(1, int(gate_config.get("retry_pair_concurrency", 1))),
            max_step_judge_retry_attempts=max(1, int(gate_config.get("max_step_judge_retry_attempts", 2))),
            step_judge_retry_initial_backoff_seconds=float(
                gate_config.get("step_judge_retry_initial_backoff_seconds", 1.0)
            ),
            step_judge_retry_backoff_multiplier=float(gate_config.get("step_judge_retry_backoff_multiplier", 2.0)),
            step_judge_retry_max_backoff_seconds=float(gate_config.get("step_judge_retry_max_backoff_seconds", 4.0)),
        )

    def _resolve_debug_output_dir(
        self,
        *,
        ropd: RopdJudgeConfig | dict[str, Any] | None,
        client_config: RopdJudgeConfig | dict[str, Any] | None,
    ) -> Path | None:
        resolved_client_source = ropd if ropd is not None else client_config
        if not isinstance(resolved_client_source, Mapping):
            return None
        debug_config = resolved_client_source.get("debug", {})
        if not isinstance(debug_config, Mapping):
            return None
        output_dir = debug_config.get("output_dir")
        if output_dir is None or str(output_dir).strip() == "":
            return None
        return Path(str(output_dir))

    def _validate_teacher_client_capabilities(self) -> None:
        if self.teacher_answer_count != 1:
            return
        if isinstance(self.teacher_client, (list, tuple)):
            return
        if callable(getattr(self.teacher_client, "generate_many", None)):
            return
        raise ValueError("ropd.teacher_answer_count=1 requires a multi-answer teacher client with generate_many().")

    def _evaluate_initial_groups(self, groups: tuple[BlackOPDGroup, ...]) -> list[RopdGroupRecord]:
        results = self._evaluate_groups(groups)
        group_records: list[RopdGroupRecord] = []
        for group, result in zip(groups, results, strict=True):
            record = RopdGroupRecord(
                uid=group.uid,
                group=group,
                result=result,
            )
            if result.fallback_used:
                record.first_error_stage = result.error_stage
                record.first_error_type = result.error_type
                record.final_status = "terminal_failure"
            group_records.append(record)
        return group_records

    def _evaluate_groups(
        self,
        groups: tuple[BlackOPDGroup, ...],
        *,
        max_concurrency: int | None = None,
    ) -> tuple[RopdGroupResult, ...]:
        if not groups:
            return tuple()

        resolved_concurrency = self.max_concurrency if max_concurrency is None else min(self.max_concurrency, max_concurrency)
        if resolved_concurrency <= 1 or len(groups) <= 1:
            return tuple(self._evaluate_group(group) for group in groups)

        with ThreadPoolExecutor(max_workers=min(resolved_concurrency, len(groups))) as executor:
            return tuple(executor.map(self._evaluate_group, groups))

    def _evaluate_group(self, group: BlackOPDGroup) -> RopdGroupResult:
        try:
            student_answers = tuple(rollout.response_text for rollout in group.rollouts)
            teacher_answer = self._generate_teacher_answer(group.raw_prompt, uid=group.uid)
            teacher_answers = self._normalize_teacher_answers(teacher_answer)
            rubric_student_answers = tuple(self._trim_answer_for_judge(answer) for answer in student_answers)
            rubric_teacher_answers = tuple(self._trim_answer_for_judge(answer) for answer in teacher_answers)
            rubric_teacher_answer: str | tuple[str, ...]
            if isinstance(teacher_answer, str) and len(teacher_answers) == 1:
                rubric_teacher_answer = rubric_teacher_answers[0]
            else:
                rubric_teacher_answer = rubric_teacher_answers
            rubric = self.rubric_client.generate(
                group.raw_prompt,
                rubric_teacher_answer,
                rubric_student_answers,
                uid=group.uid,
            )
            answer_items = self._build_shuffled_answer_items(
                uid=group.uid,
                teacher_answers=rubric_teacher_answers,
                student_answers=rubric_student_answers,
            )
            answers = tuple(item.text for item in answer_items)
            verifier_scores = self._score_answers_with_step_retry(
                group=group,
                rubric=rubric,
                answers=answers,
            )
            ordered_answer_scores = tuple(float(item.final_score) for item in verifier_scores.answers)
            if len(ordered_answer_scores) != len(answers):
                raise ValueError(
                    f"uid={group.uid!r} verifier returned {len(ordered_answer_scores)} answer scores for "
                    f"{len(answers)} answers."
                )
            teacher_scores, ordered_student_scores = self._restore_scores_by_source(
                answer_items=answer_items,
                ordered_answer_scores=ordered_answer_scores,
                teacher_count=len(teacher_answers),
                student_count=len(student_answers),
            )
            if len(ordered_student_scores) != len(group.rollouts):
                raise ValueError(
                    f"uid={group.uid!r} verifier returned {len(ordered_student_scores)} student scores for "
                    f"{len(group.rollouts)} rollouts."
                )
            maximum_score = float(getattr(rubric, "maximum_score", 0.0))
            if maximum_score <= 0.0:
                raise ValueError(f"uid={group.uid!r} rubric maximum_score must be positive.")
            reward_scores = tuple(float(score) / maximum_score for score in ordered_student_scores)
            return RopdGroupResult(
                student_scores=ordered_student_scores,
                reward_scores=reward_scores,
                judge_error=False,
                fallback_used=False,
                teacher_scores=teacher_scores,
                rubric_hash=getattr(rubric, "rubric_hash", ""),
            )
        except BlackOPDClientError as exc:
            error_details = dict(exc.details)
            if exc.status_code is not None:
                error_details.setdefault("status_code", exc.status_code)
            error_details.setdefault("retriable", exc.retriable)
            error_details.setdefault("message", str(exc))
            return RopdGroupResult(
                student_scores=tuple(0.0 for _ in group.rollouts),
                reward_scores=tuple(0.0 for _ in group.rollouts),
                judge_error=True,
                fallback_used=True,
                error_type=exc.error_type,
                error_stage=exc.stage,
                error_details=error_details,
            )
        except Exception as exc:
            return RopdGroupResult(
                student_scores=tuple(0.0 for _ in group.rollouts),
                reward_scores=tuple(0.0 for _ in group.rollouts),
                judge_error=True,
                fallback_used=True,
                error_type="runtime_error",
                error_stage="verifier",
                error_details={"message": str(exc)},
            )

    def _write_debug_record(
        self,
        *,
        group_records: list[RopdGroupRecord],
        reward_control: dict[str, Any],
    ) -> None:
        if self.debug_output_dir is None:
            return
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        self._debug_call_index += 1
        debug_path = self.debug_output_dir / "ropd_reward_debug.jsonl"
        debug_record = {
            "schema_version": "ropd.reward_debug.v1",
            "call_index": self._debug_call_index,
            "reward_control": self._json_safe(reward_control),
            "groups": [self._build_debug_group_record(record) for record in group_records],
        }
        with debug_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(debug_record, ensure_ascii=False, sort_keys=True) + "\n")

    def _build_debug_group_record(self, record: RopdGroupRecord) -> dict[str, Any]:
        result = record.result
        return {
            "uid": record.uid,
            "final_status": record.final_status,
            "retry_count": record.retry_count,
            "first_error_stage": record.first_error_stage,
            "first_error_type": record.first_error_type,
            "error_stage": result.error_stage,
            "error_type": result.error_type,
            "error_details": self._json_safe(result.error_details),
            "judge_error": bool(result.judge_error),
            "fallback_used": bool(result.fallback_used),
            "rubric_hash": result.rubric_hash,
            "teacher_scores": [float(score) for score in result.teacher_scores],
            "student_scores": [float(score) for score in result.student_scores],
            "reward_scores": [float(score) for score in result.reward_scores],
            "rollouts": [
                {"batch_index": rollout.batch_index, "response_length": rollout.response_length}
                for rollout in record.group.rollouts
            ],
        }

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        return str(value)

    def _generate_teacher_answer(self, raw_prompt: Any, *, uid: str) -> str | tuple[str, ...]:
        if isinstance(self.teacher_client, (list, tuple)):
            if not self.teacher_client:
                raise BlackOPDClientError(
                    stage="teacher",
                    error_type="validation_error",
                    message="teacher_client sequence must not be empty.",
                )
            return tuple(client.generate(raw_prompt, uid=uid) for client in self.teacher_client)
        generate_many = getattr(self.teacher_client, "generate_many", None)
        if self.teacher_answer_count == 1:
            if not callable(generate_many):
                raise BlackOPDClientError(
                    stage="teacher",
                    error_type="validation_error",
                    message="teacher_answer_count=1 requires teacher.generate_many to select from a multi-answer pool.",
                )
            teacher_answers = tuple(generate_many(raw_prompt, uid=uid))
            return self._select_stable_teacher_answer(
                raw_prompt=raw_prompt,
                uid=uid,
                teacher_answers=teacher_answers,
            )
        if callable(generate_many):
            teacher_answers = tuple(generate_many(raw_prompt, uid=uid, count=self.teacher_answer_count))
            if len(teacher_answers) != self.teacher_answer_count:
                raise BlackOPDClientError(
                    stage="teacher",
                    error_type="validation_error",
                    message=(
                        f"teacher.generate_many returned {len(teacher_answers)} answers, "
                        f"expected {self.teacher_answer_count}."
                    ),
                )
            return teacher_answers
        return tuple(self.teacher_client.generate(raw_prompt, uid=uid) for _ in range(self.teacher_answer_count))

    def _select_stable_teacher_answer(
        self,
        *,
        raw_prompt: Any,
        uid: str,
        teacher_answers: tuple[str, ...],
    ) -> str:
        normalized_answers = self._normalize_teacher_answers(teacher_answers)
        if not normalized_answers:
            raise BlackOPDClientError(
                stage="teacher",
                error_type="validation_error",
                message="teacher.generate_many returned no teacher answers for stable single-answer selection.",
            )
        return min(
            normalized_answers,
            key=lambda answer: self._teacher_answer_selection_key(uid=uid, raw_prompt=raw_prompt, answer=answer),
        )

    def _teacher_answer_selection_key(self, *, uid: str, raw_prompt: Any, answer: str) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"{uid}\x1f{canonicalize_raw_prompt(raw_prompt)}\x1f{answer}".encode("utf-8")
        ).hexdigest()
        return digest, answer

    def _normalize_teacher_answers(self, teacher_answer: str | tuple[str, ...]) -> tuple[str, ...]:
        if isinstance(teacher_answer, str):
            return (teacher_answer,)
        normalized_answers = tuple(str(answer) for answer in teacher_answer)
        return tuple(dict.fromkeys(normalized_answers))

    def _trim_answer_for_judge(self, answer: str) -> str:
        answer_text = str(answer)
        if "</think>" not in answer_text:
            return answer_text
        trimmed_answer = answer_text.rsplit("</think>", 1)[1].strip()
        if trimmed_answer:
            return trimmed_answer
        return answer_text

    def _build_shuffled_answer_items(
        self,
        *,
        uid: str,
        teacher_answers: tuple[str, ...],
        student_answers: tuple[str, ...],
    ) -> tuple[RopdAnswerItem, ...]:
        answer_items = tuple(
            RopdAnswerItem(source="teacher", source_index=index, text=answer)
            for index, answer in enumerate(teacher_answers)
        ) + tuple(
            RopdAnswerItem(source="student", source_index=index, text=answer)
            for index, answer in enumerate(student_answers)
        )
        return tuple(sorted(answer_items, key=lambda item: self._answer_shuffle_key(uid=uid, item=item)))

    def _answer_shuffle_key(self, *, uid: str, item: RopdAnswerItem) -> tuple[str, str, int]:
        digest = hashlib.sha256(
            f"{uid}\x1f{item.source}\x1f{item.source_index}\x1f{item.text}".encode("utf-8")
        ).hexdigest()
        return digest, item.source, item.source_index

    def _restore_scores_by_source(
        self,
        *,
        answer_items: tuple[RopdAnswerItem, ...],
        ordered_answer_scores: tuple[float, ...],
        teacher_count: int,
        student_count: int,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        teacher_scores: list[float | None] = [None] * teacher_count
        student_scores: list[float | None] = [None] * student_count
        for item, score in zip(answer_items, ordered_answer_scores, strict=True):
            if item.source == "teacher":
                teacher_scores[item.source_index] = score
            elif item.source == "student":
                student_scores[item.source_index] = score
            else:
                raise ValueError(f"Unknown answer source: {item.source!r}")
        if any(score is None for score in teacher_scores):
            raise ValueError("Failed to restore all teacher scores from shuffled verifier outputs.")
        if any(score is None for score in student_scores):
            raise ValueError("Failed to restore all student scores from shuffled verifier outputs.")
        return (
            tuple(float(score) for score in teacher_scores if score is not None),
            tuple(float(score) for score in student_scores if score is not None),
        )

    def _score_answers_with_step_retry(
        self,
        *,
        group: BlackOPDGroup,
        rubric: BlackOPDStructuredRubric,
        answers: tuple[str, ...],
    ):
        if self.verifier_answer_chunk_size is not None and len(answers) > self.verifier_answer_chunk_size:
            return self._score_answer_chunks_with_step_retry(group=group, rubric=rubric, answers=answers)

        return self._score_answer_chunk_with_step_retry(group=group, rubric=rubric, answers=answers)

    def _score_answer_chunks_with_step_retry(
        self,
        *,
        group: BlackOPDGroup,
        rubric: BlackOPDStructuredRubric,
        answers: tuple[str, ...],
    ) -> RopdVerifierScores:
        assert self.verifier_answer_chunk_size is not None
        all_scores: list[RopdAnswerScore] = []
        for offset in range(0, len(answers), self.verifier_answer_chunk_size):
            chunk = answers[offset : offset + self.verifier_answer_chunk_size]
            chunk_scores = self._score_answer_chunk_with_step_retry(group=group, rubric=rubric, answers=chunk)
            for chunk_score in chunk_scores.answers:
                all_scores.append(
                    RopdAnswerScore(
                        answer_index=offset + int(chunk_score.answer_index),
                        judgement=list(chunk_score.judgement),
                        final_score=float(chunk_score.final_score),
                    )
                )
        return RopdVerifierScores(
            schema_version=ROPD_BATCH_SCHEMA_VERSION,
            answers=all_scores,
        )

    def _score_answer_chunk_with_step_retry(
        self,
        *,
        group: BlackOPDGroup,
        rubric: BlackOPDStructuredRubric,
        answers: tuple[str, ...],
    ) -> RopdVerifierScores:
        last_error: BlackOPDClientError | None = None
        for _attempt in range(self.reward_quality_gate.max_step_judge_retry_attempts):
            try:
                return self.verifier_client.score_group(
                    group.raw_prompt,
                    rubric,
                    answers,
                    uid=group.uid,
                )
            except BlackOPDClientError as exc:
                last_error = exc
                if exc.error_type not in STEP_RECOVERABLE_REWARD_ERROR_TYPES:
                    raise
        assert last_error is not None
        raise last_error

    def _build_groups(self, data: DataProto) -> tuple[BlackOPDGroup, ...]:
        responses = data.batch["responses"]
        response_width = responses.shape[-1]
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_width:]
        valid_response_lengths = response_mask.sum(dim=-1)

        raw_prompts = data.non_tensor_batch["raw_prompt"]
        uids = data.non_tensor_batch["uid"]

        grouped_rollouts: OrderedDict[str, list[BlackOPDRollout]] = OrderedDict()
        grouped_prompts: dict[str, Any] = {}
        grouped_prompt_keys: dict[str, str] = {}

        for batch_index in range(len(data)):
            uid = str(uids[batch_index])
            raw_prompt = normalize_raw_prompt(raw_prompts[batch_index])
            raw_prompt_key = canonicalize_raw_prompt(raw_prompt)
            valid_response_length = int(valid_response_lengths[batch_index].item())
            if valid_response_length <= 0:
                raise ValueError(f"Sample at batch index {batch_index} has no valid response tokens.")
            if uid in grouped_prompt_keys and grouped_prompt_keys[uid] != raw_prompt_key:
                raise ValueError(f"Found multiple raw_prompt values for the same uid={uid!r}.")

            grouped_prompt_keys.setdefault(uid, raw_prompt_key)
            grouped_prompts.setdefault(uid, raw_prompt)
            grouped_rollouts.setdefault(uid, []).append(
                BlackOPDRollout(
                    batch_index=batch_index,
                    response_text=self._decode_response_text(responses[batch_index], valid_response_length),
                    response_length=valid_response_length,
                )
            )

        return tuple(
            BlackOPDGroup(uid=uid, raw_prompt=grouped_prompts[uid], rollouts=tuple(rollouts))
            for uid, rollouts in grouped_rollouts.items()
        )

    def _decode_response_text(self, response_ids: torch.Tensor, valid_response_length: int) -> str:
        valid_response_ids = response_ids[:valid_response_length]
        return self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    def _init_reward_extra_info(self, batch_size: int) -> dict[str, list[Any]]:
        return {key: [default_value] * batch_size for key, default_value in self.EXTRA_INFO_DEFAULTS.items()}

    def _validate_reward_extra_info(self, reward_extra_info: dict[str, list[Any]]) -> None:
        for key in ("student_score", "reward_score", "group_size", "rubric_hash"):
            if any(value is None for value in reward_extra_info[key]):
                raise ValueError(f"reward_extra_info[{key!r}] contains unset entries.")

    def _build_reward_control(
        self,
        group_records: list[RopdGroupRecord],
        *,
        batch_size: int,
    ) -> dict[str, Any]:
        failed_records = [record for record in group_records if record.final_status != "success"]
        fallback_rate = 0.0 if not group_records else len(failed_records) / len(group_records)
        effective_student_count = sum(
            len(record.group.rollouts) for record in group_records if record.final_status == "success"
        )
        failed_students = self._build_failed_students(group_records)
        teacher_score_groups = self._build_teacher_score_groups(group_records)
        teacher_below_student_uids = [
            group["uid"]
            for group in teacher_score_groups
            if group["teacher_scores"] and group["student_scores"]
            and min(group["teacher_scores"]) < max(group["student_scores"])
        ]
        teacher_score_group_count = len(teacher_score_groups)
        return {
            "fallback_rate_initial": fallback_rate,
            "fallback_rate_repaired": fallback_rate,
            "retry_round_count": 0,
            "retried_group_count": 0,
            "retryable_group_uids": [],
            "terminal_group_uids": [record.uid for record in failed_records],
            "update_mask": [True for _ in range(batch_size)],
            "total_group_count": len(group_records),
            "effective_uid_count": len(group_records) - len(failed_records),
            "effective_group_rate": 0.0 if not group_records else (len(group_records) - len(failed_records)) / len(group_records),
            "effective_student_count": effective_student_count,
            "group_excluded_count": len(failed_records),
            "group_excluded_uids": [record.uid for record in failed_records],
            "excluded_student_count": batch_size - effective_student_count,
            "quality_gate_stop": False,
            "stop_reason": "",
            "fallback_rate_history": [fallback_rate],
            "effective_group_rate_history": [
                0.0 if not group_records else (len(group_records) - len(failed_records)) / len(group_records)
            ],
            "step_retry_attempt": self.reward_quality_gate.max_step_judge_retry_attempts,
            "step_retry_exhausted": bool(failed_records),
            "failed_students": failed_students,
            "teacher_score_groups": teacher_score_groups,
            "teacher_below_student_group_count": len(teacher_below_student_uids),
            "teacher_below_student_group_rate": (
                0.0 if teacher_score_group_count == 0 else len(teacher_below_student_uids) / teacher_score_group_count
            ),
            "teacher_below_student_uids": teacher_below_student_uids,
            "ropd_mode": True,
            "scoring_mode": "ropd",
        }

    def _build_teacher_score_groups(
        self,
        group_records: list[RopdGroupRecord],
    ) -> list[dict[str, Any]]:
        teacher_score_groups: list[dict[str, Any]] = []
        for record in group_records:
            if record.final_status != "success" or not record.result.teacher_scores:
                continue
            teacher_score_groups.append(
                {
                    "uid": record.uid,
                    "teacher_scores": [float(score) for score in record.result.teacher_scores],
                    "student_scores": [float(score) for score in record.result.student_scores],
                }
            )
        return teacher_score_groups

    def _build_failed_students(self, group_records: list[RopdGroupRecord]) -> list[dict[str, Any]]:
        failed_students: list[dict[str, Any]] = []
        for record in group_records:
            if record.final_status == "success":
                continue
            for student_index, rollout in enumerate(record.group.rollouts):
                failed_students.append(
                    {
                        "uid": record.uid,
                        "batch_index": rollout.batch_index,
                        "student_index": student_index,
                        "error_stage": record.result.error_stage,
                        "error_type": record.result.error_type,
                        "first_error_stage": record.first_error_stage or record.result.error_stage,
                        "first_error_type": record.first_error_type or record.result.error_type,
                        "retry_count": record.retry_count,
                        "final_status": record.final_status,
                    }
                )
        failed_students.sort(key=lambda item: (item["uid"], item["student_index"]))
        return failed_students

    def _build_passthrough_reward_control(self, batch_size: int) -> dict[str, Any]:
        return {
            "fallback_rate_initial": 0.0,
            "fallback_rate_repaired": 0.0,
            "retry_round_count": 0,
            "retried_group_count": 0,
            "retryable_group_uids": [],
            "terminal_group_uids": [],
            "update_mask": [True for _ in range(batch_size)],
            "total_group_count": 0,
            "effective_uid_count": 0,
            "effective_group_rate": 0.0,
            "effective_student_count": batch_size,
            "group_excluded_count": 0,
            "group_excluded_uids": [],
            "excluded_student_count": 0,
            "quality_gate_stop": False,
            "stop_reason": "",
            "fallback_rate_history": [0.0],
            "effective_group_rate_history": [0.0],
            "step_retry_attempt": 0,
            "step_retry_exhausted": False,
            "failed_students": [],
            "teacher_score_groups": [],
            "teacher_below_student_group_count": 0,
            "teacher_below_student_group_rate": 0.0,
            "teacher_below_student_uids": [],
        }
