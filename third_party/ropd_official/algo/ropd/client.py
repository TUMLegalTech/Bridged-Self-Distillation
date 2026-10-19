from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from algo.ropd.prompts import (
    build_ropd_rubricator_prompt,
    build_ropd_verifier_prompt,
)
from algo.ropd_clients import (
    PROMPT_TEMPLATE_VERSION,
    BlackOPDClientError,
    BlackOPDDebugConfig,
    BlackOPDProviderCircuitBreakerConfig,
    BlackOPDProviderLimitsConfig,
    BlackOPDRequestSchedulerConfig,
    BlackOPDStageBreakerConfigSet,
    BlackOPDRubricCriterion,
    BlackOPDStructuredRubric,
    BlackOPDVerifierScore,
    OpenAICompatibleProvider,
    OpenAIRoleConfig,
    OpenAITransportConfig,
    OpenAITeacherClient,
    StaticTeacherClient,
    _build_circuit_breaker_config,
    _build_role_config,
    _coerce_mapping,
    _coerce_optional_positive_int,
    _coerce_optional_string,
    _coerce_positive_int,
    _json_schema_for_model,
    _merge_nested_mappings,
    _parse_structured_rubric,
    _parse_json_payload,
    _prepare_repo_environment,
    _resolve_profiled_provider_limits_config,
    _role_config_from_resolved_role,
    _validate_structured_rubric,
    _validate_verifier_score,
)
from algo.ropd_judge_provider_resolver import BlackOPDJudgeProviderResolver
from algo.ropd_teacher_index import (
    OfflineTeacherIndex,
    OfflineTeacherIndexClient,
    build_teacher_fingerprint_payload,
)

ROPD_BATCH_SCHEMA_VERSION = "ropd.batch_verifier.v2"


ROPD_MAX_TEACHER_ANSWER_COUNT = 4


def coerce_ropd_teacher_answer_count(
    value: Any,
    *,
    default: int = 1,
    field_name: str = "ropd.teacher_answer_count",
) -> int:
    resolved_value = int(value if value is not None else default)
    if not 1 <= resolved_value <= ROPD_MAX_TEACHER_ANSWER_COUNT:
        raise ValueError(
            f"{field_name} must be between 1 and {ROPD_MAX_TEACHER_ANSWER_COUNT}."
        )
    return resolved_value
class RopdAnswerScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer_index: int
    judgement: list[bool]
    final_score: float

    @field_validator("answer_index")
    @classmethod
    def _validate_answer_index(cls, value: int) -> int:
        if value < 1:
            raise ValueError("answer_index must be positive")
        return value


class RopdVerifierScores(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    answers: list[RopdAnswerScore]


class RopdTeacherClient(Protocol):
    def generate(self, raw_prompt: Any, *, uid: str | None = None) -> str: ...


class RopdRubricatorClient(Protocol):
    def generate(
        self,
        raw_prompt: Any,
        teacher_answer: str | Sequence[str],
        student_answers: Sequence[str],
        *,
        uid: str | None = None,
    ) -> BlackOPDStructuredRubric: ...


class RopdVerifierClient(Protocol):
    def score_group(
        self,
        raw_prompt: Any,
        rubric: BlackOPDStructuredRubric,
        answers: Sequence[str],
        *,
        uid: str | None = None,
    ) -> RopdVerifierScores: ...


@dataclass(frozen=True, slots=True)
class RopdJudgeConfig:
    teacher: OpenAIRoleConfig
    rubricator: OpenAIRoleConfig
    verifier: OpenAIRoleConfig
    transport: OpenAITransportConfig
    teacher_answer_count: int = 1
    verifier_answer_chunk_size: int | None = None
    max_concurrency: int = 4
    provider_limits: BlackOPDProviderLimitsConfig = field(default_factory=BlackOPDProviderLimitsConfig)
    request_scheduler: BlackOPDRequestSchedulerConfig = field(default_factory=BlackOPDRequestSchedulerConfig)
    extra_rubric_instructions: str = ""
    extra_scoring_instructions: str = ""


def parse_ropd_scores(
    raw_text: str,
    *,
    rubric: BlackOPDStructuredRubric,
    expected_answer_count: int,
) -> RopdVerifierScores:
    payload = _parse_json_payload(raw_text, stage="verifier")
    try:
        parsed = RopdVerifierScores.model_validate(payload)
    except ValidationError as exc:
        raise BlackOPDClientError(
            stage="verifier",
            error_type="schema_error",
            message=f"group absolute shared rubric verifier payload does not match schema: {exc}",
        ) from exc

    if parsed.schema_version != ROPD_BATCH_SCHEMA_VERSION:
        raise BlackOPDClientError(stage="verifier", error_type="validation_error", message="schema_version mismatch")
    if len(parsed.answers) != expected_answer_count:
        raise BlackOPDClientError(stage="verifier", error_type="validation_error", message="unexpected answer count")
    expected_answer_indices = list(range(1, expected_answer_count + 1))
    if [item.answer_index for item in parsed.answers] != expected_answer_indices:
        raise BlackOPDClientError(
            stage="verifier",
            error_type="validation_error",
            message="answer indices must cover 1..n in order",
        )

    for answer in parsed.answers:
        score = BlackOPDVerifierScore(
            schema_version="ropd.verifier.v1",
            judgement=list(answer.judgement),
            final_score=float(answer.final_score),
        )
        _validate_verifier_score(score, rubric=rubric)
    return parsed


class OpenAIRopdRubricatorClient:
    def __init__(
        self,
        *,
        provider: OpenAICompatibleProvider,
        role_config: OpenAIRoleConfig,
        extra_rubric_instructions: str,
    ) -> None:
        self.provider = provider
        self.role_config = role_config
        self.extra_rubric_instructions = extra_rubric_instructions

    def generate(
        self,
        raw_prompt: Any,
        teacher_answer: str | Sequence[str],
        student_answers: Sequence[str],
        *,
        uid: str | None = None,
    ) -> BlackOPDStructuredRubric:
        try:
            prompt_text = build_ropd_rubricator_prompt(
                raw_prompt,
                teacher_answer=teacher_answer,
                student_answers=student_answers,
                extra_rubric_instructions=self.extra_rubric_instructions,
                model=self.role_config.model,
            )
        except (TypeError, ValueError) as exc:
            raise BlackOPDClientError(
                stage="rubricator",
                error_type="validation_error",
                message=f"group absolute shared rubric prompt construction failed: {exc}",
            ) from exc

        try:
            return self.provider.create_text(
                stage="rubricator",
                role=self.role_config,
                input_payload=prompt_text,
                text_format=_json_schema_for_model(BlackOPDStructuredRubric, name="ropd"),
                output_validator=_parse_structured_rubric,
            )
        except BlackOPDClientError as exc:
            exc.add_context(uid=uid)
            raise


class OpenAIRopdVerifierClient:
    def __init__(
        self,
        *,
        provider: OpenAICompatibleProvider,
        role_config: OpenAIRoleConfig,
        extra_scoring_instructions: str,
    ) -> None:
        self.provider = provider
        self.role_config = role_config
        self.extra_scoring_instructions = extra_scoring_instructions

    def score_group(
        self,
        raw_prompt: Any,
        rubric: BlackOPDStructuredRubric,
        answers: Sequence[str],
        *,
        uid: str | None = None,
    ) -> RopdVerifierScores:
        try:
            prompt_text = build_ropd_verifier_prompt(
                raw_prompt,
                rubrics=[criterion.model_dump(mode="json") for criterion in rubric.rubrics],
                answers=answers,
                extra_scoring_instructions=self.extra_scoring_instructions,
                model=self.role_config.model,
            )
        except (TypeError, ValueError) as exc:
            raise BlackOPDClientError(
                stage="verifier",
                error_type="validation_error",
                message=f"group absolute shared rubric verifier prompt construction failed: {exc}",
            ) from exc

        try:
            return self.provider.create_text(
                stage="verifier",
                role=self.role_config,
                input_payload=prompt_text,
                text_format=_json_schema_for_model(
                    RopdVerifierScores,
                    name="ropd_batch_verifier",
                ),
                output_validator=lambda raw_text: parse_ropd_scores(
                    raw_text,
                    rubric=rubric,
                    expected_answer_count=len(answers),
                ),
            )
        except BlackOPDClientError as exc:
            exc.add_context(uid=uid)
            raise


class StaticRopdRubricatorClient:
    def __init__(self, *, debug_config: BlackOPDDebugConfig, role_config: OpenAIRoleConfig) -> None:
        self.debug_config = debug_config
        self.role_config = role_config

    def generate(
        self,
        raw_prompt: Any,
        teacher_answer: str | Sequence[str],
        student_answers: Sequence[str],
        *,
        uid: str | None = None,
    ) -> BlackOPDStructuredRubric:
        del raw_prompt, teacher_answer, student_answers, uid
        rubric = BlackOPDStructuredRubric(
            schema_version="ropd.rubric.v1",
            rubrics=[
                BlackOPDRubricCriterion(criterion_id="c1", category="Task", criterion="criterion 1", points=1),
                BlackOPDRubricCriterion(criterion_id="c2", category="Task", criterion="criterion 2", points=1),
                BlackOPDRubricCriterion(criterion_id="c3", category="Task", criterion="criterion 3", points=1),
                BlackOPDRubricCriterion(criterion_id="c4", category="Task", criterion="criterion 4", points=1),
            ],
            maximum_score=4,
        )
        return _validate_structured_rubric(rubric)


class StaticRopdVerifierClient:
    def score_group(
        self,
        raw_prompt: Any,
        rubric: BlackOPDStructuredRubric,
        answers: Sequence[str],
        *,
        uid: str | None = None,
    ) -> RopdVerifierScores:
        del raw_prompt, uid
        answer_scores: list[RopdAnswerScore] = []
        for position, _answer in enumerate(answers):
            answer_index = position + 1
            if position % 2 == 0:
                judgement = [True] * len(rubric.rubrics)
                final_score = float(rubric.maximum_score)
            else:
                judgement = [True] + [False] * (len(rubric.rubrics) - 1)
                final_score = float(rubric.rubrics[0].points)
            _validate_verifier_score(
                BlackOPDVerifierScore(
                    schema_version="ropd.verifier.v1",
                    judgement=judgement,
                    final_score=final_score,
                ),
                rubric=rubric,
            )
            answer_scores.append(
                RopdAnswerScore(
                    answer_index=answer_index,
                    judgement=judgement,
                    final_score=final_score,
                )
            )
        return RopdVerifierScores(
            schema_version=ROPD_BATCH_SCHEMA_VERSION,
            answers=answer_scores,
        )


def build_ropd_judge_config(
    config: dict[str, Any] | None = None,
) -> RopdJudgeConfig:
    _prepare_repo_environment()
    config = {} if config is None else dict(config)
    resolution_config = _coerce_mapping(config.get("provider_resolution"))
    teacher_config = _coerce_mapping(config.get("teacher"))
    rubricator_config = _coerce_mapping(config.get("rubricator"))
    verifier_config = _coerce_mapping(config.get("verifier"))
    transport_config = _coerce_mapping(config.get("transport"))
    provider_limits_config = _resolve_profiled_provider_limits_config(_coerce_mapping(config.get("provider_limits")))
    request_scheduler_config = _coerce_mapping(config.get("request_scheduler"))

    if resolution_config:
        merged_resolution_overrides = _merge_nested_mappings(
            _coerce_mapping(resolution_config.get("overrides")),
            {
                role_name: role_config
                for role_name, role_config in (
                    ("teacher", teacher_config),
                    ("rubricator", rubricator_config),
                    ("verifier", verifier_config),
                )
                if role_config
            },
        )
        resolved_provider_config = BlackOPDJudgeProviderResolver(
            spec_path=(
                _coerce_optional_string(
                    resolution_config.get("spec_path"),
                    default="verl/trainer/config/ropd/ropd_judge_providers.yaml",
                )
                or "verl/trainer/config/ropd/ropd_judge_providers.yaml"
            ),
            entrypoint=_coerce_optional_string(resolution_config.get("entrypoint"), default="train") or "train",
            overrides=merged_resolution_overrides,
        ).resolve()
        teacher_role = _role_config_from_resolved_role(resolved_provider_config.roles.teacher)
        rubricator_role = _role_config_from_resolved_role(resolved_provider_config.roles.rubricator)
        verifier_role = _role_config_from_resolved_role(resolved_provider_config.roles.verifier)
        primary_online_role = next(
            (
                role
                for role in (
                    resolved_provider_config.roles.teacher,
                    resolved_provider_config.roles.rubricator,
                    resolved_provider_config.roles.verifier,
                )
                if role.provider == "openai_compatible"
            ),
            resolved_provider_config.roles.teacher,
        )
        transport_config = _merge_nested_mappings(primary_online_role.transport, transport_config)
        provider_limits_config = _resolve_profiled_provider_limits_config(
            _merge_nested_mappings(primary_online_role.provider_limits, provider_limits_config)
        )
    else:
        if not teacher_config and not rubricator_config and not verifier_config:
            raise ValueError("ropd.provider_resolution or ropd.teacher is required.")
        teacher_role = _build_role_config(
            teacher_config,
            role_name="teacher",
            default_provider="openai_compatible",
            default_model="gpt-5.2-chat-latest",
            default_api_key=None,
            default_base_url=None,
            default_reasoning_effort=None,
            default_timeout_seconds=90.0,
            default_max_output_tokens=8192,
            default_temperature=None,
            default_top_p=None,
            default_empty_response_retries=0,
            default_incomplete_retries=0,
            default_parse_error_retries=0,
            default_schema_error_retries=0,
            default_validation_error_retries=0,
        )
        rubricator_role = _build_role_config(
            rubricator_config,
            role_name="rubricator",
            default_provider="openai_compatible" if teacher_role.provider == "offline_index" else teacher_role.provider,
            default_model=teacher_role.model,
            default_api_key=teacher_role.api_key,
            default_base_url=teacher_role.base_url,
            default_reasoning_effort=None,
            default_timeout_seconds=90.0,
            default_max_output_tokens=2048,
            default_temperature=None,
            default_top_p=None,
            default_empty_response_retries=1,
            default_incomplete_retries=1,
            default_parse_error_retries=1,
            default_schema_error_retries=1,
            default_validation_error_retries=2,
        )
        verifier_role = _build_role_config(
            verifier_config,
            role_name="verifier",
            default_provider="openai_compatible" if teacher_role.provider == "offline_index" else teacher_role.provider,
            default_model=teacher_role.model,
            default_api_key=teacher_role.api_key,
            default_base_url=teacher_role.base_url,
            default_reasoning_effort=None,
            default_timeout_seconds=30.0,
            default_max_output_tokens=1024,
            default_temperature=None,
            default_top_p=None,
            default_empty_response_retries=1,
            default_incomplete_retries=1,
            default_parse_error_retries=1,
            default_schema_error_retries=1,
            default_validation_error_retries=0,
        )

    if rubricator_role.provider == "offline_index":
        raise ValueError("ropd does not support rubricator.provider=offline_index.")
    if verifier_role.provider == "offline_index":
        raise ValueError("ropd does not support verifier.provider=offline_index.")

    transport = OpenAITransportConfig(
        max_retries=int(transport_config.get("max_retries", 2)),
        initial_backoff_seconds=float(transport_config.get("initial_backoff_seconds", 1.0)),
        backoff_multiplier=float(transport_config.get("backoff_multiplier", 2.0)),
        max_backoff_seconds=float(transport_config.get("max_backoff_seconds", 8.0)),
        max_in_flight_requests=_coerce_positive_int(
            transport_config.get("max_in_flight_requests"),
            default=32,
            field_name="ropd.transport.max_in_flight_requests",
        ),
    )
    circuit_breaker = _build_circuit_breaker_config(
        provider_limits_config.get("circuit_breaker"),
        defaults=BlackOPDProviderCircuitBreakerConfig(),
        field_name_prefix="ropd.provider_limits.circuit_breaker",
    )
    stage_breaker_config = _coerce_mapping(provider_limits_config.get("stage_breakers"))
    provider_limits = BlackOPDProviderLimitsConfig(
        max_concurrent_requests=_coerce_positive_int(
            provider_limits_config.get("max_concurrent_requests"),
            default=transport.max_in_flight_requests,
            field_name="ropd.provider_limits.max_concurrent_requests",
        ),
        max_rpm=_coerce_optional_positive_int(
            provider_limits_config.get("max_rpm"),
            default=1920,
            field_name="ropd.provider_limits.max_rpm",
        ),
        max_tpm=_coerce_optional_positive_int(
            provider_limits_config.get("max_tpm"),
            default=6000000,
            field_name="ropd.provider_limits.max_tpm",
        ),
        circuit_breaker=circuit_breaker,
        stage_breakers=BlackOPDStageBreakerConfigSet(
            teacher=_build_circuit_breaker_config(
                stage_breaker_config.get("teacher"),
                defaults=circuit_breaker,
                field_name_prefix="ropd.provider_limits.stage_breakers.teacher",
            ),
            rubricator=_build_circuit_breaker_config(
                stage_breaker_config.get("rubricator"),
                defaults=circuit_breaker,
                field_name_prefix="ropd.provider_limits.stage_breakers.rubricator",
            ),
            verifier=_build_circuit_breaker_config(
                stage_breaker_config.get("verifier"),
                defaults=circuit_breaker,
                field_name_prefix="ropd.provider_limits.stage_breakers.verifier",
            ),
        ),
    )
    request_scheduler = BlackOPDRequestSchedulerConfig(
        enabled=bool(request_scheduler_config.get("enabled", True)),
        num_workers=_coerce_optional_positive_int(
            request_scheduler_config.get("num_workers"),
            default=None,
            field_name="ropd.request_scheduler.num_workers",
        ),
        max_queue_size=_coerce_optional_positive_int(
            request_scheduler_config.get("max_queue_size"),
            default=None,
            field_name="ropd.request_scheduler.max_queue_size",
        ),
        stage_priority_enabled=bool(request_scheduler_config.get("stage_priority_enabled", True)),
        record_queue_metrics=bool(request_scheduler_config.get("record_queue_metrics", True)),
    )

    return RopdJudgeConfig(
        teacher=teacher_role,
        rubricator=rubricator_role,
        verifier=verifier_role,
        transport=transport,
        teacher_answer_count=coerce_ropd_teacher_answer_count(
            config.get("teacher_answer_count"),
            default=1,
            field_name="ropd.teacher_answer_count",
        ),
        verifier_answer_chunk_size=_coerce_optional_positive_int(
            config.get("verifier_answer_chunk_size"),
            default=None,
            field_name="ropd.verifier_answer_chunk_size",
        ),
        max_concurrency=_coerce_positive_int(
            config.get("max_concurrency"),
            default=4,
            field_name="ropd.max_concurrency",
        ),
        provider_limits=provider_limits,
        request_scheduler=request_scheduler,
        extra_rubric_instructions=_coerce_optional_string(config.get("extra_rubric_instructions"), default="") or "",
        extra_scoring_instructions=_coerce_optional_string(config.get("extra_scoring_instructions"), default="") or "",
    )


def build_ropd_clients(
    config: RopdJudgeConfig | dict[str, Any] | None = None,
    *,
    provider: OpenAICompatibleProvider | None = None,
) -> tuple[
    RopdTeacherClient,
    RopdRubricatorClient,
    RopdVerifierClient,
]:
    resolved_config = (
        config
        if isinstance(config, RopdJudgeConfig)
        else build_ropd_judge_config(config)
    )
    needs_provider = any(
        role.provider == "openai_compatible"
        for role in (resolved_config.teacher, resolved_config.rubricator, resolved_config.verifier)
    )
    resolved_provider = provider
    if resolved_provider is None and needs_provider:
        resolved_provider = OpenAICompatibleProvider(
            resolved_config.transport,
            provider_limits=resolved_config.provider_limits,
            request_scheduler_config=resolved_config.request_scheduler,
        )

    if resolved_config.teacher.provider == "offline_index":
        fingerprint = build_teacher_fingerprint_payload(
            provider="openai_compatible",
            model=resolved_config.teacher.model,
            base_url=resolved_config.teacher.base_url,
            reasoning_effort=resolved_config.teacher.reasoning_effort,
            max_output_tokens=resolved_config.teacher.max_output_tokens,
            temperature=resolved_config.teacher.temperature,
            top_p=resolved_config.teacher.top_p,
            timeout_seconds=resolved_config.teacher.timeout_seconds,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
        teacher_index = OfflineTeacherIndex.load(
            index_path=resolved_config.teacher.index_path,
            expected_fingerprint=fingerprint,
        )
        teacher_client = OfflineTeacherIndexClient(teacher_index=teacher_index)
    elif resolved_config.teacher.provider == "openai_compatible":
        if resolved_provider is None:
            raise ValueError("OpenAI provider is required for openai_compatible teacher role.")
        teacher_client: RopdTeacherClient = OpenAITeacherClient(
            provider=resolved_provider,
            role_config=resolved_config.teacher,
        )
    else:
        teacher_client = StaticTeacherClient(
            debug_config=BlackOPDDebugConfig(),
            role_config=resolved_config.teacher,
        )

    if resolved_config.rubricator.provider == "openai_compatible":
        if resolved_provider is None:
            raise ValueError("OpenAI provider is required for openai_compatible rubricator role.")
        rubricator_client: RopdRubricatorClient = OpenAIRopdRubricatorClient(
            provider=resolved_provider,
            role_config=resolved_config.rubricator,
            extra_rubric_instructions=resolved_config.extra_rubric_instructions,
        )
    else:
        rubricator_client = StaticRopdRubricatorClient(
            debug_config=BlackOPDDebugConfig(),
            role_config=resolved_config.rubricator,
        )

    if resolved_config.verifier.provider == "openai_compatible":
        if resolved_provider is None:
            raise ValueError("OpenAI provider is required for openai_compatible verifier role.")
        verifier_client: RopdVerifierClient = OpenAIRopdVerifierClient(
            provider=resolved_provider,
            role_config=resolved_config.verifier,
            extra_scoring_instructions=resolved_config.extra_scoring_instructions,
        )
    else:
        verifier_client = StaticRopdVerifierClient()

    return teacher_client, rubricator_client, verifier_client
