from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any, Literal

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from algo.ropd_artifacts import BlackOPDArtifactExporter, BlackOPDExportConfig
from algo.ropd_judge_provider_resolver import (
    BlackOPDJudgeProviderResolver,
    ResolvedJudgeProviderConfig,
    ResolvedJudgeRole,
)
from algo.ropd_scheduler import BlackOPDRequestSchedulerConfig, BoundedRequestScheduler
from algo.ropd_teacher_index import (
    OfflineTeacherIndex,
    OfflineTeacherIndexClient,
    build_teacher_fingerprint_payload,
)
from algo.openai_env import apply_selected_openai_profile_to_environment
from algo.ropd_prompts import (
    PROMPT_TEMPLATE_VERSION,
    build_rubricator_prompt,
    build_teacher_input_messages,
    build_verifier_prompt,
)

RUBRIC_SCHEMA_VERSION = "ropd.rubric.v1"
VERIFIER_SCHEMA_VERSION = "ropd.verifier.v1"
REQUEST_FINGERPRINT_SCHEMA_VERSION = "ropd.request_fingerprint.v1"
MIN_RUBRIC_CRITERIA = 4
MAX_RUBRIC_CRITERIA = 12
TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
SUPPORTED_ROLE_PROVIDERS = frozenset({"openai_compatible", "static", "offline_index"})
SUPPORTED_ROLE_API_STYLES = frozenset({"responses", "chat_completions"})
ROPD_STAGES: tuple[Literal["teacher", "rubricator", "verifier"], ...] = ("teacher", "rubricator", "verifier")
TEXT_ARTIFACT_MODES: tuple[Literal["diagnostic_only", "all_pairs"], ...] = ("diagnostic_only", "all_pairs")
PROVIDER_METRIC_NAMES = (
    "requests_started",
    "requests_succeeded",
    "requests_failed",
    "retries",
    "retriable_errors",
    "timeout_errors",
    "rate_limit_wait_count",
    "rate_limit_wait_seconds",
    "circuit_open_rejections",
    "estimated_tokens",
)


class BlackOPDClientError(RuntimeError):
    def __init__(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        error_type: Literal[
            "timeout",
            "http_error",
            "circuit_open",
            "parse_error",
            "schema_error",
            "empty_response",
            "incomplete",
            "validation_error",
        ],
        message: str,
        status_code: int | None = None,
        retriable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.error_type = error_type
        self.status_code = status_code
        self.retriable = retriable
        self.details = dict(details or {})

    def add_context(self, **context: Any) -> BlackOPDClientError:
        for key, value in context.items():
            if value is not None:
                self.details[key] = value
        return self

    def clone(self) -> BlackOPDClientError:
        return BlackOPDClientError(
            stage=self.stage,
            error_type=self.error_type,
            message=str(self),
            status_code=self.status_code,
            retriable=self.retriable,
            details=copy.deepcopy(self.details),
        )


class BlackOPDRubricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str
    category: str
    criterion: str
    points: int

    @field_validator("criterion_id", "category", "criterion")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value

    @field_validator("criterion_id")
    @classmethod
    def _validate_criterion_id(cls, value: str) -> str:
        if not value.startswith("c") or not value[1:].isdigit():
            raise ValueError("criterion_id must use the form c1, c2, c3, ...")
        return value

    @field_validator("points")
    @classmethod
    def _validate_points(cls, value: int) -> int:
        if value < 1 or value > 5:
            raise ValueError("points must be between 1 and 5")
        return value


class BlackOPDStructuredRubric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[RUBRIC_SCHEMA_VERSION]
    rubrics: list[BlackOPDRubricCriterion]
    maximum_score: int

    def canonical_hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rubrics": [criterion.model_dump(mode="json") for criterion in self.rubrics],
            "maximum_score": self.maximum_score,
        }

    @property
    def rubric_hash(self) -> str:
        canonical_json = json.dumps(
            self.canonical_hash_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    @property
    def total_points(self) -> int:
        return sum(criterion.points for criterion in self.rubrics)


class BlackOPDVerifierScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[VERIFIER_SCHEMA_VERSION]
    judgement: list[bool]
    final_score: float


@dataclass(frozen=True, slots=True)
class OpenAIRoleConfig:
    model: str
    api_key: str
    base_url: str | None
    timeout_seconds: float
    provider: str = "openai_compatible"
    api_style: str = "responses"
    reasoning_effort: str | None = "none"
    max_output_tokens: int | None = None
    temperature: float | None = 1.0
    top_p: float | None = None
    empty_response_retries: int = 0
    incomplete_retries: int = 0
    parse_error_retries: int = 0
    schema_error_retries: int = 0
    validation_error_retries: int = 0
    index_path: str | None = None


@dataclass(frozen=True, slots=True)
class OpenAITransportConfig:
    max_retries: int = 2
    initial_backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 8.0
    max_in_flight_requests: int = 32


@dataclass(frozen=True, slots=True)
class BlackOPDProviderCircuitBreakerConfig:
    consecutive_retriable_errors: int = 5
    rolling_window_size: int = 20
    rolling_error_rate: float = 0.2
    cooldown_seconds: float = 30.0
    half_open_probe_requests: int = 2


@dataclass(frozen=True, slots=True)
class BlackOPDStageBreakerConfigSet:
    teacher: BlackOPDProviderCircuitBreakerConfig
    rubricator: BlackOPDProviderCircuitBreakerConfig
    verifier: BlackOPDProviderCircuitBreakerConfig

    def for_stage(
        self,
        stage: Literal["teacher", "rubricator", "verifier"],
    ) -> BlackOPDProviderCircuitBreakerConfig:
        return getattr(self, stage)


@dataclass(frozen=True, slots=True)
class BlackOPDProviderLimitsConfig:
    max_concurrent_requests: int = 32
    max_rpm: int | None = 1920
    max_tpm: int | None = 6000000
    circuit_breaker: BlackOPDProviderCircuitBreakerConfig = field(default_factory=BlackOPDProviderCircuitBreakerConfig)
    stage_breakers: BlackOPDStageBreakerConfigSet | None = None

    def __post_init__(self) -> None:
        if self.stage_breakers is None:
            shared_breakers = BlackOPDStageBreakerConfigSet(
                teacher=self.circuit_breaker,
                rubricator=self.circuit_breaker,
                verifier=self.circuit_breaker,
            )
            object.__setattr__(self, "stage_breakers", shared_breakers)


@dataclass(frozen=True, slots=True)
class BlackOPDDebugConfig:
    include_text_artifacts: bool = False
    output_dir: str = "outputs/ropd"
    retention_days: int = 14
    text_artifact_mode: Literal["diagnostic_only", "all_pairs"] = "diagnostic_only"
    static_teacher_response: str = "Teacher reference answer: 42."
    static_maximum_score: int = 8


@dataclass(frozen=True, slots=True)
class BlackOPDClientConfig:
    teacher: OpenAIRoleConfig
    rubricator: OpenAIRoleConfig
    verifier: OpenAIRoleConfig
    transport: OpenAITransportConfig = field(default_factory=OpenAITransportConfig)
    max_group_concurrency: int = 4
    max_pair_concurrency: int = 8
    max_verifier_subject_concurrency: int = 2
    provider_limits: BlackOPDProviderLimitsConfig = field(default_factory=BlackOPDProviderLimitsConfig)
    request_scheduler: BlackOPDRequestSchedulerConfig = field(default_factory=BlackOPDRequestSchedulerConfig)
    export: BlackOPDExportConfig = field(default_factory=BlackOPDExportConfig)
    debug: BlackOPDDebugConfig = field(default_factory=BlackOPDDebugConfig)


@dataclass(slots=True)
class SharedProviderResources:
    semaphore: BoundedSemaphore
    rpm_limiter: SyncTokenBucket | None
    tpm_limiter: SyncTokenBucket | None
    client_cache: dict[tuple[str, str | None, float], Any]
    client_cache_lock: Lock
    inflight_requests: dict[str, Future[Any]]
    inflight_requests_lock: Lock


@dataclass(slots=True)
class StageProviderRuntime:
    stage_name: Literal["teacher", "rubricator", "verifier"]
    circuit_breaker: BlackOPDProviderCircuitBreaker
    metrics: dict[str, float]
    breaker_config: BlackOPDProviderCircuitBreakerConfig
    first_retriable_error: dict[str, Any] | None = None
    last_retriable_error: dict[str, Any] | None = None


def _empty_provider_metrics() -> dict[str, float]:
    return {name: 0.0 for name in PROVIDER_METRIC_NAMES}


def _sum_provider_metrics(metric_snapshots: list[dict[str, float]]) -> dict[str, float]:
    totals = _empty_provider_metrics()
    for snapshot in metric_snapshots:
        for name in PROVIDER_METRIC_NAMES:
            totals[name] += snapshot.get(name, 0.0)
    return totals


def _coerce_optional_float(value: Any, *, default: float | None) -> float | None:
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        return float(stripped)
    return float(value)


def _coerce_optional_int(value: Any, *, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        return int(stripped)
    return int(value)


def _coerce_optional_string(value: Any, *, default: str | None = None) -> str | None:
    if value is None:
        return default
    stripped = str(value).strip()
    return stripped or default


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    stripped = str(value).strip().lower()
    if stripped in {"1", "true", "yes", "on"}:
        return True
    if stripped in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot coerce {value!r} to bool.")


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _merge_nested_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {key: copy.deepcopy(value) for key, value in base.items()}
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = _merge_nested_mappings(base_value, override_value)
        else:
            merged[key] = copy.deepcopy(override_value)
    return merged


def _coerce_positive_int(value: Any, *, default: int, field_name: str) -> int:
    resolved_value = int(value if value is not None else default)
    if resolved_value < 1:
        raise ValueError(f"{field_name} must be at least 1.")
    return resolved_value


def _coerce_non_negative_int(value: Any, *, default: int, field_name: str) -> int:
    resolved_value = int(value if value is not None else default)
    if resolved_value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return resolved_value


def _coerce_optional_positive_int(value: Any, *, default: int | None, field_name: str) -> int | None:
    resolved_value = _coerce_optional_int(value, default=default)
    if resolved_value is None:
        return None
    if resolved_value < 1:
        raise ValueError(f"{field_name} must be at least 1.")
    return resolved_value


def _build_circuit_breaker_config(
    breaker_config: Mapping[str, Any] | None,
    *,
    defaults: BlackOPDProviderCircuitBreakerConfig,
    field_name_prefix: str,
) -> BlackOPDProviderCircuitBreakerConfig:
    resolved_breaker_config = _coerce_mapping(breaker_config)
    rolling_error_rate = float(resolved_breaker_config.get("rolling_error_rate", defaults.rolling_error_rate))
    if not 0.0 < rolling_error_rate <= 1.0:
        raise ValueError(f"{field_name_prefix}.rolling_error_rate must be in (0, 1].")
    built_config = BlackOPDProviderCircuitBreakerConfig(
        consecutive_retriable_errors=_coerce_positive_int(
            resolved_breaker_config.get("consecutive_retriable_errors"),
            default=defaults.consecutive_retriable_errors,
            field_name=f"{field_name_prefix}.consecutive_retriable_errors",
        ),
        rolling_window_size=_coerce_positive_int(
            resolved_breaker_config.get("rolling_window_size"),
            default=defaults.rolling_window_size,
            field_name=f"{field_name_prefix}.rolling_window_size",
        ),
        rolling_error_rate=rolling_error_rate,
        cooldown_seconds=float(resolved_breaker_config.get("cooldown_seconds", defaults.cooldown_seconds)),
        half_open_probe_requests=_coerce_positive_int(
            resolved_breaker_config.get("half_open_probe_requests"),
            default=defaults.half_open_probe_requests,
            field_name=f"{field_name_prefix}.half_open_probe_requests",
        ),
    )
    if built_config.cooldown_seconds < 0:
        raise ValueError(f"{field_name_prefix}.cooldown_seconds must be non-negative.")
    return built_config


@cache
def _load_repo_dotenv_into_environment() -> bool:
    skip_repo_dotenv = os.getenv("ROPD_SKIP_REPO_DOTENV")
    if skip_repo_dotenv is not None and skip_repo_dotenv.strip().lower() in {"1", "true", "yes", "on"}:
        return False
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    if not dotenv_path.exists():
        return False
    return load_dotenv(dotenv_path=dotenv_path, override=True)


def _prepare_repo_environment() -> None:
    _load_repo_dotenv_into_environment()
    apply_selected_openai_profile_to_environment()


def _get_env_value(name: str) -> str | None:
    _prepare_repo_environment()
    env_value = os.getenv(name)
    if env_value is not None and env_value.strip():
        return env_value
    return None


def _resolve_profiled_provider_limits_config(provider_limits_config: Mapping[str, Any]) -> dict[str, Any]:
    resolved_provider_limits_config = {
        key: copy.deepcopy(value) for key, value in provider_limits_config.items() if key != "profiles"
    }
    selected_profile = _coerce_optional_string(os.getenv("OPENAI_PROFILE"), default=None)
    if selected_profile is None:
        return resolved_provider_limits_config

    profile_overrides = _coerce_mapping(provider_limits_config.get("profiles"))
    selected_profile_override = _coerce_mapping(profile_overrides.get(selected_profile.upper()))
    if not selected_profile_override:
        return resolved_provider_limits_config
    return _merge_nested_mappings(resolved_provider_limits_config, selected_profile_override)


def _require_api_key(api_key: str | None) -> str:
    if api_key is None or not api_key.strip():
        raise ValueError("ROPD OpenAI-compatible clients require a non-empty API key.")
    return api_key.strip()


def _resolve_role_env_value(
    role_config: dict[str, Any],
    *,
    value_key: str,
    env_key: str,
    default_env_name: str | None,
    default_value: str | None,
) -> str | None:
    direct_value = _coerce_optional_string(role_config.get(value_key), default=None)
    if direct_value is not None:
        return direct_value

    env_name = _coerce_optional_string(role_config.get(env_key), default=default_env_name)
    if env_name is not None:
        env_value = _get_env_value(env_name)
        if env_value is not None:
            return env_value
    return default_value


def _resolve_reasoning_effort(role_config: dict[str, Any], *, default: str | None) -> str | None:
    reasoning_config = _coerce_mapping(role_config.get("reasoning"))
    reasoning_value = role_config.get("reasoning_effort", reasoning_config.get("effort"))
    return _coerce_optional_string(reasoning_value, default=default)


def _build_role_config(
    role_config: dict[str, Any],
    *,
    role_name: Literal["teacher", "rubricator", "verifier"],
    default_provider: str,
    default_model: str,
    default_api_key: str | None,
    default_base_url: str | None,
    default_reasoning_effort: str | None,
    default_timeout_seconds: float,
    default_max_output_tokens: int | None,
    default_temperature: float | None,
    default_top_p: float | None,
    default_empty_response_retries: int,
    default_incomplete_retries: int,
    default_parse_error_retries: int,
    default_schema_error_retries: int,
    default_validation_error_retries: int,
) -> OpenAIRoleConfig:
    provider = _coerce_optional_string(role_config.get("provider"), default=default_provider) or default_provider
    if provider not in SUPPORTED_ROLE_PROVIDERS:
        raise ValueError(
            f"Unsupported ROPD provider {provider!r}. Supported providers: {sorted(SUPPORTED_ROLE_PROVIDERS)}."
        )
    api_style = _coerce_optional_string(role_config.get("api_style"), default="responses") or "responses"
    if api_style not in SUPPORTED_ROLE_API_STYLES:
        raise ValueError(
            f"Unsupported ROPD api_style {api_style!r}. Supported values: {sorted(SUPPORTED_ROLE_API_STYLES)}."
        )

    model = _coerce_optional_string(role_config.get("model"), default=default_model) or default_model
    reasoning_effort = _resolve_reasoning_effort(role_config, default=default_reasoning_effort)
    timeout_seconds = _coerce_optional_float(role_config.get("timeout_seconds"), default=default_timeout_seconds)
    max_output_tokens = _coerce_optional_int(role_config.get("max_output_tokens"), default=default_max_output_tokens)
    temperature = _coerce_optional_float(role_config.get("temperature"), default=default_temperature)
    top_p = _coerce_optional_float(role_config.get("top_p"), default=default_top_p)
    response_retry_config = _coerce_mapping(role_config.get("response_retry"))
    empty_response_retries = _coerce_non_negative_int(
        role_config.get("empty_response_retries", response_retry_config.get("empty_response_retries")),
        default=default_empty_response_retries,
        field_name=f"ropd.{role_name}.response_retry.empty_response_retries",
    )
    incomplete_retries = _coerce_non_negative_int(
        role_config.get("incomplete_retries", response_retry_config.get("incomplete_retries")),
        default=default_incomplete_retries,
        field_name=f"ropd.{role_name}.response_retry.incomplete_retries",
    )
    parse_error_retries = _coerce_non_negative_int(
        role_config.get("parse_error_retries", response_retry_config.get("parse_error_retries")),
        default=default_parse_error_retries,
        field_name=f"ropd.{role_name}.response_retry.parse_error_retries",
    )
    schema_error_retries = _coerce_non_negative_int(
        role_config.get("schema_error_retries", response_retry_config.get("schema_error_retries")),
        default=default_schema_error_retries,
        field_name=f"ropd.{role_name}.response_retry.schema_error_retries",
    )
    validation_error_retries = _coerce_non_negative_int(
        role_config.get("validation_error_retries", response_retry_config.get("validation_error_retries")),
        default=default_validation_error_retries,
        field_name=f"ropd.{role_name}.response_retry.validation_error_retries",
    )
    index_path = _coerce_optional_string(role_config.get("index_path"), default=None)

    api_key = _resolve_role_env_value(
        role_config,
        value_key="api_key",
        env_key="api_key_env",
        default_env_name="OPENAI_API_KEY",
        default_value=default_api_key,
    )
    base_url = _resolve_role_env_value(
        role_config,
        value_key="base_url",
        env_key="base_url_env",
        default_env_name="OPENAI_BASE_URL",
        default_value=default_base_url,
    )

    if provider == "offline_index":
        if role_name != "teacher":
            raise ValueError("offline_index is only supported for teacher.")
        if index_path is None:
            raise ValueError("ropd.teacher.index_path is required when teacher.provider=offline_index.")

    if provider == "openai_compatible":
        resolved_api_key = _require_api_key(api_key)
    else:
        resolved_api_key = api_key or ""

    return OpenAIRoleConfig(
        provider=provider,
        api_style=api_style,
        model=model,
        api_key=resolved_api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds or default_timeout_seconds,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        empty_response_retries=empty_response_retries,
        incomplete_retries=incomplete_retries,
        parse_error_retries=parse_error_retries,
        schema_error_retries=schema_error_retries,
        validation_error_retries=validation_error_retries,
        index_path=index_path,
    )


def _role_config_from_resolved_role(role: ResolvedJudgeRole) -> OpenAIRoleConfig:
    return OpenAIRoleConfig(
        provider=role.provider,
        api_style=role.api_style,
        model=role.model,
        api_key=role.api_key or "",
        base_url=role.base_url,
        timeout_seconds=role.timeout_seconds,
        reasoning_effort=role.reasoning_effort,
        max_output_tokens=role.max_output_tokens,
        temperature=role.temperature,
        top_p=role.top_p,
        empty_response_retries=role.response_retry.get("empty_response_retries", 0),
        incomplete_retries=role.response_retry.get("incomplete_retries", 0),
        parse_error_retries=role.response_retry.get("parse_error_retries", 0),
        schema_error_retries=role.response_retry.get("schema_error_retries", 0),
        validation_error_retries=role.response_retry.get("validation_error_retries", 0),
        index_path=role.index_path,
    )


def _select_primary_online_resolved_role(resolved_config: ResolvedJudgeProviderConfig) -> ResolvedJudgeRole:
    for role in (resolved_config.roles.teacher, resolved_config.roles.rubricator, resolved_config.roles.verifier):
        if role.provider == "openai_compatible":
            return role
    return resolved_config.roles.teacher


def _build_resolution_overrides_from_legacy_config(config: Mapping[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for role_name in ("teacher", "rubricator", "verifier"):
        role_config = _coerce_mapping(config.get(role_name))
        if role_config:
            overrides[role_name] = role_config
    return overrides


def build_ropd_client_config(config: dict[str, Any] | None = None) -> BlackOPDClientConfig:
    _prepare_repo_environment()
    config = {} if config is None else dict(config)
    resolution_config = _coerce_mapping(config.get("provider_resolution"))
    transport_config = _coerce_mapping(config.get("transport"))
    provider_limits_config = _resolve_profiled_provider_limits_config(_coerce_mapping(config.get("provider_limits")))
    request_scheduler_config = _coerce_mapping(config.get("request_scheduler"))
    export_config = _coerce_mapping(config.get("export"))
    debug_config = _coerce_mapping(config.get("debug"))

    if not resolution_config and not config:
        raise ValueError("ropd.provider_resolution is required.")

    if resolution_config:
        merged_resolution_overrides = _merge_nested_mappings(
            _coerce_mapping(resolution_config.get("overrides")),
            _build_resolution_overrides_from_legacy_config(config),
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
        primary_online_role = _select_primary_online_resolved_role(resolved_provider_config)
        transport_config = _merge_nested_mappings(primary_online_role.transport, transport_config)
        provider_limits_config = _resolve_profiled_provider_limits_config(
            _merge_nested_mappings(primary_online_role.provider_limits, provider_limits_config)
        )
    else:
        teacher_config = _coerce_mapping(config.get("teacher"))
        rubricator_config = _coerce_mapping(config.get("rubricator"))
        verifier_config = _coerce_mapping(config.get("verifier"))
        teacher_role = _build_role_config(
            teacher_config,
            role_name="teacher",
            default_provider="openai_compatible",
            default_model="gpt-5.2-chat-latest",
            default_api_key=None,
            default_base_url=None,
            default_reasoning_effort=None,
            default_timeout_seconds=45.0,
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
            default_max_output_tokens=8192,
            default_temperature=None,
            default_top_p=None,
            default_empty_response_retries=0,
            default_incomplete_retries=0,
            default_parse_error_retries=0,
            default_schema_error_retries=0,
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
            default_max_output_tokens=8192,
            default_temperature=None,
            default_top_p=None,
            default_empty_response_retries=1,
            default_incomplete_retries=1,
            default_parse_error_retries=1,
            default_schema_error_retries=1,
            default_validation_error_retries=0,
        )

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
    if transport.max_retries < 0:
        raise ValueError("ropd.transport.max_retries must be non-negative.")
    if transport.initial_backoff_seconds < 0:
        raise ValueError("ropd.transport.initial_backoff_seconds must be non-negative.")
    if transport.backoff_multiplier < 1.0:
        raise ValueError("ropd.transport.backoff_multiplier must be at least 1.0.")
    if transport.max_backoff_seconds < 0:
        raise ValueError("ropd.transport.max_backoff_seconds must be non-negative.")

    max_group_concurrency = _coerce_positive_int(
        config.get("max_group_concurrency"),
        default=4,
        field_name="ropd.max_group_concurrency",
    )
    max_pair_concurrency = _coerce_positive_int(
        config.get("max_pair_concurrency"),
        default=8,
        field_name="ropd.max_pair_concurrency",
    )
    max_verifier_subject_concurrency = _coerce_positive_int(
        config.get("max_verifier_subject_concurrency"),
        default=2,
        field_name="ropd.max_verifier_subject_concurrency",
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
        enabled=_coerce_bool(request_scheduler_config.get("enabled"), default=True),
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
        stage_priority_enabled=_coerce_bool(
            request_scheduler_config.get("stage_priority_enabled"),
            default=True,
        ),
        record_queue_metrics=_coerce_bool(
            request_scheduler_config.get("record_queue_metrics"),
            default=True,
        ),
    )

    include_text_artifacts = _coerce_bool(
        debug_config.get("include_text_artifacts"),
        default=_coerce_bool(export_config.get("enabled"), default=False),
    )
    output_dir = _coerce_optional_string(
        debug_config.get("output_dir"),
        default=_coerce_optional_string(export_config.get("output_dir"), default="outputs/ropd"),
    ) or "outputs/ropd"
    static_maximum_score = int(debug_config.get("static_maximum_score", 8))
    if static_maximum_score < 6:
        raise ValueError("ropd.debug.static_maximum_score must be at least 6.")
    retention_days = int(debug_config.get("retention_days", export_config.get("retention_days", 14)))
    if retention_days < 1:
        raise ValueError("ropd.debug.retention_days must be at least 1.")
    text_artifact_mode = (
        _coerce_optional_string(
            debug_config.get("text_artifact_mode"),
            default=_coerce_optional_string(
                export_config.get("text_artifact_mode"),
                default="diagnostic_only",
            ),
        )
        or "diagnostic_only"
    )
    if text_artifact_mode not in TEXT_ARTIFACT_MODES:
        raise ValueError(
            "ropd.debug.text_artifact_mode must be one of "
            f"{list(TEXT_ARTIFACT_MODES)!r}, got {text_artifact_mode!r}."
        )
    debug = BlackOPDDebugConfig(
        include_text_artifacts=include_text_artifacts,
        output_dir=output_dir,
        retention_days=retention_days,
        text_artifact_mode=text_artifact_mode,
        static_teacher_response=(
            _coerce_optional_string(
                debug_config.get("static_teacher_response"),
                default="Teacher reference answer: 42.",
            )
            or "Teacher reference answer: 42."
        ),
        static_maximum_score=static_maximum_score,
    )
    return BlackOPDClientConfig(
        teacher=teacher_role,
        rubricator=rubricator_role,
        verifier=verifier_role,
        transport=transport,
        max_group_concurrency=max_group_concurrency,
        max_pair_concurrency=max_pair_concurrency,
        max_verifier_subject_concurrency=max_verifier_subject_concurrency,
        provider_limits=provider_limits,
        request_scheduler=request_scheduler,
        export=BlackOPDExportConfig(
            enabled=include_text_artifacts,
            output_dir=output_dir,
            retention_days=retention_days,
            text_artifact_mode=text_artifact_mode,
        ),
        debug=debug,
    )


class SyncTokenBucket:
    def __init__(
        self,
        *,
        rate_limit: float,
        max_tokens: float | None = None,
        time_fn: Any = time.monotonic,
        sleep: Any = time.sleep,
    ) -> None:
        self.rate_limit = rate_limit
        self.max_tokens = max_tokens or rate_limit
        self._time_fn = time_fn
        self._sleep = sleep
        self._tokens = self.max_tokens
        self._last_update: float | None = None
        self._lock = Lock()

    def acquire(self, num_tokens: float = 1.0) -> float:
        if num_tokens <= 0:
            return 0.0

        total_wait_seconds = 0.0

        if num_tokens > self.max_tokens:
            wait_seconds = 0.0
            with self._lock:
                now = self._time_fn()
                if self._last_update is None:
                    self._last_update = now

                elapsed = max(0.0, now - self._last_update)
                self._tokens = min(self.max_tokens, self._tokens + elapsed * self.rate_limit)
                self._last_update = now

                tokens_needed = num_tokens - self._tokens
                if tokens_needed > 0:
                    wait_seconds = tokens_needed / self.rate_limit

                self._tokens = max(-self.max_tokens, self._tokens - num_tokens)

            if wait_seconds > 0:
                total_wait_seconds += wait_seconds
                self._sleep(wait_seconds)
            return total_wait_seconds

        while True:
            wait_seconds = 0.0
            with self._lock:
                now = self._time_fn()
                if self._last_update is None:
                    self._last_update = now

                elapsed = max(0.0, now - self._last_update)
                self._tokens = min(self.max_tokens, self._tokens + elapsed * self.rate_limit)
                self._last_update = now

                if self._tokens >= num_tokens:
                    self._tokens -= num_tokens
                    return total_wait_seconds

                tokens_needed = num_tokens - self._tokens
                wait_seconds = tokens_needed / self.rate_limit

            total_wait_seconds += wait_seconds
            self._sleep(wait_seconds)


class BlackOPDProviderCircuitBreaker:
    def __init__(
        self,
        config: BlackOPDProviderCircuitBreakerConfig,
        *,
        time_fn: Any = time.monotonic,
    ) -> None:
        self.config = config
        self._time_fn = time_fn
        self._lock = Lock()
        self._state = "closed"
        self._opened_at = 0.0
        self._consecutive_retriable_errors = 0
        self._recent_outcomes: deque[bool] = deque(maxlen=config.rolling_window_size)
        self._half_open_in_flight = 0
        self._half_open_successes = 0

    def allow_request(self) -> bool:
        with self._lock:
            now = self._time_fn()
            if self._state == "open":
                if now - self._opened_at < self.config.cooldown_seconds:
                    return False
                self._state = "half_open"
                self._half_open_in_flight = 0
                self._half_open_successes = 0

            if self._state == "half_open":
                if self._half_open_in_flight >= self.config.half_open_probe_requests:
                    return False
                self._half_open_in_flight += 1

            return True

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_retriable_errors = 0
            self._recent_outcomes.append(False)
            if self._state == "half_open":
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._half_open_successes += 1
                if (
                    self._half_open_in_flight == 0
                    and self._half_open_successes >= self.config.half_open_probe_requests
                ):
                    self._close_locked()

    def record_retriable_error(self) -> None:
        with self._lock:
            self._consecutive_retriable_errors += 1
            self._recent_outcomes.append(True)
            if self._state == "half_open":
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._open_locked()
                return

            rolling_error_rate = sum(self._recent_outcomes) / len(self._recent_outcomes)
            should_trip_by_rate = (
                len(self._recent_outcomes) >= self.config.rolling_window_size
                and rolling_error_rate >= self.config.rolling_error_rate
            )
            should_trip_by_consecutive = (
                self._consecutive_retriable_errors >= self.config.consecutive_retriable_errors
            )
            if should_trip_by_consecutive or should_trip_by_rate:
                self._open_locked()

    def record_ignored_failure(self) -> None:
        with self._lock:
            self._consecutive_retriable_errors = 0
            if self._state == "half_open":
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)

    def record_response_quality_failure(self) -> None:
        with self._lock:
            if self._state == "half_open":
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)

    def _open_locked(self) -> None:
        self._state = "open"
        self._opened_at = self._time_fn()
        self._half_open_in_flight = 0
        self._half_open_successes = 0

    def _close_locked(self) -> None:
        self._state = "closed"
        self._consecutive_retriable_errors = 0
        self._recent_outcomes.clear()
        self._half_open_in_flight = 0
        self._half_open_successes = 0


class OpenAICompatibleProvider:
    def __init__(
        self,
        transport: OpenAITransportConfig,
        *,
        provider_limits: BlackOPDProviderLimitsConfig | None = None,
        request_scheduler_config: BlackOPDRequestSchedulerConfig | None = None,
        client_factory: Any = OpenAI,
        sleep: Any = time.sleep,
        time_fn: Any = time.monotonic,
        uniform: Any = random.uniform,
    ) -> None:
        self.transport = transport
        self.provider_limits = provider_limits or BlackOPDProviderLimitsConfig()
        self.request_scheduler_config = request_scheduler_config or BlackOPDRequestSchedulerConfig()
        self._client_factory = client_factory
        self._sleep = sleep
        self._time_fn = time_fn
        self._uniform = uniform
        self._metrics_lock = Lock()
        effective_max_in_flight = min(
            self.transport.max_in_flight_requests,
            self.provider_limits.max_concurrent_requests,
        )
        self._request_scheduler = self._build_request_scheduler(effective_max_in_flight)
        self._shared_resources = SharedProviderResources(
            semaphore=BoundedSemaphore(value=effective_max_in_flight),
            rpm_limiter=(
                SyncTokenBucket(
                    rate_limit=self.provider_limits.max_rpm / 60.0,
                    max_tokens=self.provider_limits.max_rpm / 60.0,
                    time_fn=self._time_fn,
                    sleep=self._sleep,
                )
                if self.provider_limits.max_rpm is not None
                else None
            ),
            tpm_limiter=(
                SyncTokenBucket(
                    rate_limit=self.provider_limits.max_tpm / 60.0,
                    max_tokens=self.provider_limits.max_tpm / 60.0,
                    time_fn=self._time_fn,
                    sleep=self._sleep,
                )
                if self.provider_limits.max_tpm is not None
                else None
            ),
            client_cache={},
            client_cache_lock=Lock(),
            inflight_requests={},
            inflight_requests_lock=Lock(),
        )
        stage_breakers = self.provider_limits.stage_breakers or BlackOPDStageBreakerConfigSet(
            teacher=self.provider_limits.circuit_breaker,
            rubricator=self.provider_limits.circuit_breaker,
            verifier=self.provider_limits.circuit_breaker,
        )
        self._stage_runtimes = {
            stage_name: StageProviderRuntime(
                stage_name=stage_name,
                circuit_breaker=BlackOPDProviderCircuitBreaker(
                    stage_breakers.for_stage(stage_name),
                    time_fn=self._time_fn,
                ),
                metrics=_empty_provider_metrics(),
                breaker_config=stage_breakers.for_stage(stage_name),
            )
            for stage_name in ROPD_STAGES
        }

    def _build_request_scheduler(self, effective_max_in_flight: int) -> BoundedRequestScheduler | None:
        if not self.request_scheduler_config.enabled:
            return None
        resolved_num_workers = self.request_scheduler_config.num_workers or effective_max_in_flight
        resolved_max_queue_size = self.request_scheduler_config.max_queue_size or resolved_num_workers
        return BoundedRequestScheduler(
            num_workers=resolved_num_workers,
            max_queue_size=resolved_max_queue_size,
            stage_priority_enabled=self.request_scheduler_config.stage_priority_enabled,
            record_queue_metrics=self.request_scheduler_config.record_queue_metrics,
            time_fn=self._time_fn,
        )

    def close(self) -> None:
        if self._request_scheduler is not None:
            self._request_scheduler.shutdown(wait=True)

    def snapshot_metrics(self) -> dict[str, dict[str, Any]]:
        with self._metrics_lock:
            stage_snapshots: dict[str, dict[str, Any]] = {}
            for stage_name, runtime in self._stage_runtimes.items():
                stage_snapshot: dict[str, Any] = dict(runtime.metrics)
                if runtime.first_retriable_error is not None:
                    stage_snapshot["first_retriable_error"] = copy.deepcopy(runtime.first_retriable_error)
                if runtime.last_retriable_error is not None:
                    stage_snapshot["last_retriable_error"] = copy.deepcopy(runtime.last_retriable_error)
                stage_snapshots[stage_name] = stage_snapshot
        stage_snapshots["totals"] = _sum_provider_metrics(list(stage_snapshots.values()))
        if self._request_scheduler is not None and self.request_scheduler_config.record_queue_metrics:
            stage_snapshots["totals"].update(self._request_scheduler.snapshot_metrics())
        return stage_snapshots

    def _stage_runtime(
        self,
        stage: Literal["teacher", "rubricator", "verifier"],
    ) -> StageProviderRuntime:
        return self._stage_runtimes[stage]

    def create_text(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        role: OpenAIRoleConfig,
        input_payload: Any,
        text_format: dict[str, Any] | None = None,
        output_validator: Any = None,
    ) -> Any:
        request_fingerprint = self._build_request_fingerprint(
            stage=stage,
            role=role,
            input_payload=input_payload,
            text_format=text_format,
        )

        def run_direct_request() -> Any:
            return self._create_text_direct(
                stage=stage,
                role=role,
                input_payload=input_payload,
                text_format=text_format,
                output_validator=output_validator,
            )

        if request_fingerprint is not None:
            future, is_leader = self._acquire_inflight_request(request_fingerprint)
            if not is_leader:
                return self._await_inflight_result(future)
        else:
            future = None

        if self._request_scheduler is None:
            try:
                result = run_direct_request()
            except BaseException as exc:
                if request_fingerprint is not None and future is not None:
                    self._complete_inflight_request(request_fingerprint=request_fingerprint, future=future, error=exc)
                raise
            if request_fingerprint is not None and future is not None:
                self._complete_inflight_request(
                    request_fingerprint=request_fingerprint,
                    future=future,
                    result=result,
                )
            return result

        try:
            result = self._request_scheduler.submit(stage=stage, fn=run_direct_request)
        except BaseException as exc:
            if request_fingerprint is not None and future is not None:
                self._complete_inflight_request(request_fingerprint=request_fingerprint, future=future, error=exc)
            raise
        if request_fingerprint is not None and future is not None:
            self._complete_inflight_request(
                request_fingerprint=request_fingerprint,
                future=future,
                result=result,
            )
        return result

    def _create_text_direct(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        role: OpenAIRoleConfig,
        input_payload: Any,
        text_format: dict[str, Any] | None,
        output_validator: Any = None,
    ) -> Any:
        request_kwargs = self._build_request_kwargs(
            role=role,
            input_payload=input_payload,
            text_format=text_format,
        )
        request_metadata = self._build_request_metadata(role=role, text_format=text_format)

        empty_response_retry_count = 0
        incomplete_retry_count = 0
        parse_error_retry_count = 0
        schema_error_retry_count = 0
        validation_error_retry_count = 0
        while True:
            response = self._execute_with_retry(
                stage=stage,
                role=role,
                input_payload=input_payload,
                text_format=text_format,
                request_metadata=request_metadata,
                request=(
                    (lambda client: client.chat.completions.create(**request_kwargs))
                    if role.api_style == "chat_completions"
                    else (lambda client: client.responses.create(**request_kwargs))
                ),
            )
            response_metadata = self._build_response_metadata(response)
            response_status = _coerce_optional_string(response_metadata.get("status"), default=None)
            resolved_output_text = (
                _coerce_optional_string(response_metadata.get("resolved_output_text"), default="") or ""
            ).strip()
            if response_status not in (None, "completed"):
                error_type: Literal["empty_response", "incomplete"] = (
                    "incomplete" if response_status == "incomplete" else "empty_response"
                )
                error = BlackOPDClientError(
                    stage=stage,
                    error_type=error_type,
                    message=f"{stage} response status was {response_status!r}.",
                    details={"request": request_metadata, "response": response_metadata},
                )
                self._record_response_quality_failure(stage=stage, error=error)
                if error_type == "incomplete" and incomplete_retry_count < role.incomplete_retries:
                    self._record_metric(stage=stage, name="retries")
                    self._sleep(self._compute_response_retry_delay(incomplete_retry_count))
                    incomplete_retry_count += 1
                    continue
                if error_type == "empty_response" and empty_response_retry_count < role.empty_response_retries:
                    self._record_metric(stage=stage, name="retries")
                    self._sleep(self._compute_response_retry_delay(empty_response_retry_count))
                    empty_response_retry_count += 1
                    continue
                raise error
            if not resolved_output_text:
                error = BlackOPDClientError(
                    stage=stage,
                    error_type="empty_response",
                    message=f"{stage} response was empty.",
                    details={"request": request_metadata, "response": response_metadata},
                )
                self._record_response_quality_failure(stage=stage, error=error)
                if empty_response_retry_count < role.empty_response_retries:
                    self._record_metric(stage=stage, name="retries")
                    self._sleep(self._compute_response_retry_delay(empty_response_retry_count))
                    empty_response_retry_count += 1
                    continue
                raise error
            resolved_output: Any = resolved_output_text
            if output_validator is not None:
                try:
                    resolved_output = output_validator(resolved_output_text)
                except BlackOPDClientError as exc:
                    exc.add_context(
                        request=request_metadata,
                        response=response_metadata,
                        raw_output_text=resolved_output_text,
                    )
                    self._record_response_quality_failure(stage=stage, error=exc)
                    if exc.error_type == "parse_error" and parse_error_retry_count < role.parse_error_retries:
                        self._record_metric(stage=stage, name="retries")
                        self._sleep(self._compute_response_retry_delay(parse_error_retry_count))
                        parse_error_retry_count += 1
                        continue
                    if exc.error_type == "schema_error" and schema_error_retry_count < role.schema_error_retries:
                        self._record_metric(stage=stage, name="retries")
                        self._sleep(self._compute_response_retry_delay(schema_error_retry_count))
                        schema_error_retry_count += 1
                        continue
                    if (
                        exc.error_type == "validation_error"
                        and validation_error_retry_count < role.validation_error_retries
                    ):
                        self._record_metric(stage=stage, name="retries")
                        self._sleep(self._compute_response_retry_delay(validation_error_retry_count))
                        validation_error_retry_count += 1
                        continue
                    raise
            self._after_request_success(stage=stage)
            return resolved_output

    def _acquire_inflight_request(self, request_fingerprint: str) -> tuple[Future[Any], bool]:
        with self._shared_resources.inflight_requests_lock:
            existing_future = self._shared_resources.inflight_requests.get(request_fingerprint)
            if existing_future is not None:
                return existing_future, False
            created_future: Future[Any] = Future()
            self._shared_resources.inflight_requests[request_fingerprint] = created_future
            return created_future, True

    def _await_inflight_result(self, future: Future[Any]) -> Any:
        try:
            return future.result()
        except BlackOPDClientError as exc:
            raise exc.clone() from exc

    def _complete_inflight_request(
        self,
        *,
        request_fingerprint: str,
        future: Future[Any],
        result: Any | None = None,
        error: BaseException | None = None,
    ) -> None:
        with self._shared_resources.inflight_requests_lock:
            existing_future = self._shared_resources.inflight_requests.get(request_fingerprint)
            if existing_future is future:
                self._shared_resources.inflight_requests.pop(request_fingerprint, None)
        if future.done():
            return
        try:
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(result)
        except InvalidStateError:
            pass

    def _record_metric(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        name: str,
        value: float = 1.0,
    ) -> None:
        runtime = self._stage_runtime(stage)
        with self._metrics_lock:
            runtime.metrics[name] = runtime.metrics.get(name, 0.0) + value

    def _before_request(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        role: OpenAIRoleConfig,
        input_payload: Any,
        text_format: dict[str, Any] | None,
    ) -> None:
        runtime = self._stage_runtime(stage)
        if not runtime.circuit_breaker.allow_request():
            self._record_metric(stage=stage, name="circuit_open_rejections")
            raise BlackOPDClientError(
                stage=stage,
                error_type="circuit_open",
                message=f"{stage} provider circuit breaker is open; request rejected during cooldown.",
                retriable=False,
            )

        waited_seconds = 0.0
        if self._shared_resources.rpm_limiter is not None:
            waited_seconds += self._shared_resources.rpm_limiter.acquire(1.0)
        if self._shared_resources.tpm_limiter is not None:
            estimated_tokens = float(
                self._estimate_request_tokens(
                    role=role,
                    input_payload=input_payload,
                    text_format=text_format,
                )
            )
            waited_seconds += self._shared_resources.tpm_limiter.acquire(estimated_tokens)
            self._record_metric(stage=stage, name="estimated_tokens", value=estimated_tokens)
        if waited_seconds > 0:
            self._record_metric(stage=stage, name="rate_limit_wait_count")
            self._record_metric(stage=stage, name="rate_limit_wait_seconds", value=waited_seconds)
        self._record_metric(stage=stage, name="requests_started")

    def _after_request_success(self, *, stage: Literal["teacher", "rubricator", "verifier"]) -> None:
        runtime = self._stage_runtime(stage)
        runtime.circuit_breaker.record_success()
        self._record_metric(stage=stage, name="requests_succeeded")

    def _after_request_failure(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        error: BlackOPDClientError,
    ) -> None:
        runtime = self._stage_runtime(stage)
        self._record_metric(stage=stage, name="requests_failed")
        if error.error_type == "timeout":
            self._record_metric(stage=stage, name="timeout_errors")
        if error.retriable:
            error_snapshot = self._build_retriable_error_snapshot(error)
            if runtime.first_retriable_error is None:
                runtime.first_retriable_error = error_snapshot
            runtime.last_retriable_error = error_snapshot
            runtime.circuit_breaker.record_retriable_error()
            self._record_metric(stage=stage, name="retriable_errors")
            return
        runtime.circuit_breaker.record_ignored_failure()

    @staticmethod
    def _build_retriable_error_snapshot(error: BlackOPDClientError) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "stage": error.stage,
            "error_type": error.error_type,
            "retriable": error.retriable,
            "message": str(error),
        }
        if error.status_code is not None:
            snapshot["status_code"] = error.status_code
        if error.details:
            snapshot["details"] = copy.deepcopy(error.details)
        return snapshot

    def _record_response_quality_failure(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        error: BlackOPDClientError,
    ) -> None:
        """Record a response-quality failure (empty/incomplete) without affecting the circuit breaker."""
        runtime = self._stage_runtime(stage)
        runtime.circuit_breaker.record_response_quality_failure()
        self._record_metric(stage=stage, name="requests_failed")

    def _estimate_request_tokens(
        self,
        *,
        role: OpenAIRoleConfig,
        input_payload: Any,
        text_format: dict[str, Any] | None,
    ) -> int:
        def _serialize(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            try:
                return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            except (TypeError, ValueError):
                return str(value)

        prompt_char_count = len(_serialize(input_payload)) + len(_serialize(text_format))
        prompt_token_estimate = max(1, math.ceil(prompt_char_count / 4))
        completion_token_estimate = role.max_output_tokens if role.max_output_tokens is not None else 1024
        return max(1, prompt_token_estimate + int(completion_token_estimate))

    def _build_request_fingerprint(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        role: OpenAIRoleConfig,
        input_payload: Any,
        text_format: dict[str, Any] | None,
    ) -> str | None:
        fingerprint_payload = {
            "request_fingerprint_schema_version": REQUEST_FINGERPRINT_SCHEMA_VERSION,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "stage": stage,
            "provider": role.provider,
            "api_style": role.api_style,
            "model": role.model,
            "base_url": role.base_url,
            "reasoning_effort": role.reasoning_effort,
            "max_output_tokens": role.max_output_tokens,
            "temperature": role.temperature,
            "top_p": role.top_p,
            "input_payload": input_payload,
            "text_format": text_format,
        }
        try:
            canonical_json = json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError):
            return None
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def _build_request_kwargs(
        self,
        *,
        role: OpenAIRoleConfig,
        input_payload: Any,
        text_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if role.api_style == "chat_completions":
            request_kwargs: dict[str, Any] = {
                "model": role.model,
                "messages": self._normalize_chat_messages(input_payload),
            }
            if role.max_output_tokens is not None:
                request_kwargs["max_tokens"] = role.max_output_tokens
            if role.reasoning_effort in (None, "none") and role.temperature is not None:
                request_kwargs["temperature"] = role.temperature
            if role.reasoning_effort in (None, "none") and role.top_p is not None:
                request_kwargs["top_p"] = role.top_p
            if text_format is not None:
                request_kwargs["response_format"] = {"type": "json_object"}
            return request_kwargs

        request_kwargs: dict[str, Any] = {
            "model": role.model,
            "input": input_payload,
        }
        if role.reasoning_effort is not None:
            request_kwargs["reasoning"] = {"effort": role.reasoning_effort}
        if role.max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = role.max_output_tokens
        if role.reasoning_effort in (None, "none") and role.temperature is not None:
            request_kwargs["temperature"] = role.temperature
        if role.reasoning_effort in (None, "none") and role.top_p is not None:
            request_kwargs["top_p"] = role.top_p
        if text_format is not None:
            request_kwargs["text"] = {"format": text_format}
        return request_kwargs

    def _build_request_metadata(
        self,
        *,
        role: OpenAIRoleConfig,
        text_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        request_metadata: dict[str, Any] = {
            "provider": role.provider,
            "api_style": role.api_style,
            "model": role.model,
            "timeout_seconds": role.timeout_seconds,
            "reasoning_effort": role.reasoning_effort,
            "max_output_tokens": role.max_output_tokens,
            "temperature": role.temperature,
            "top_p": role.top_p,
            "response_retry": {
                "empty_response_retries": role.empty_response_retries,
                "incomplete_retries": role.incomplete_retries,
                "parse_error_retries": role.parse_error_retries,
                "schema_error_retries": role.schema_error_retries,
            },
        }
        if text_format is not None:
            request_metadata["text_format"] = {
                "type": text_format.get("type"),
                "name": text_format.get("name"),
            }
        return request_metadata

    def _build_response_metadata(self, response: Any) -> dict[str, Any]:
        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            finish_reason = self._to_json_compatible(getattr(first_choice, "finish_reason", None))
            message_content_text = self._extract_chat_completion_text(
                None if message is None else getattr(message, "content", None)
            )
            message_reasoning_text = self._extract_chat_completion_reasoning_text(message)
            resolved_output_text = message_content_text or message_reasoning_text
            resolved_output_text_source = (
                "chat_completion_message"
                if message_content_text
                else "chat_completion_reasoning"
                if message_reasoning_text
                else "none"
            )
            response_status = "completed"
            incomplete_details = None
            if finish_reason not in (None, "stop"):
                if finish_reason == "length":
                    response_status = "incomplete"
                    incomplete_details = {"reason": "length"}
                else:
                    response_status = str(finish_reason)
            response_metadata = {
                "id": self._to_json_compatible(getattr(response, "id", None)),
                "model": self._to_json_compatible(getattr(response, "model", None)),
                "status": response_status,
                "incomplete_details": incomplete_details,
                "usage": self._to_json_compatible(getattr(response, "usage", None)),
                "finish_reason": finish_reason,
                "resolved_output_text": resolved_output_text or None,
                "resolved_output_text_length": len(resolved_output_text),
                "resolved_output_text_source": resolved_output_text_source,
                "choices": self._to_json_compatible(choices),
            }
            return {
                key: value
                for key, value in response_metadata.items()
                if value is not None and value != []
            }

        direct_output_text = _coerce_optional_string(getattr(response, "output_text", None), default="") or ""
        output_payload = self._to_json_compatible(getattr(response, "output", None))
        text_fragments = self._collect_text_fragments(output_payload)
        reconstructed_output_text = "".join(text_fragments).strip()
        resolved_output_text = direct_output_text.strip() if direct_output_text.strip() else reconstructed_output_text
        resolved_output_text_source = "output_text" if direct_output_text.strip() else "output_fragments"
        if not resolved_output_text:
            resolved_output_text_source = "none"

        response_metadata = {
            "id": self._to_json_compatible(getattr(response, "id", None)),
            "model": self._to_json_compatible(getattr(response, "model", None)),
            "status": self._to_json_compatible(getattr(response, "status", None)),
            "incomplete_details": self._to_json_compatible(getattr(response, "incomplete_details", None)),
            "usage": self._to_json_compatible(getattr(response, "usage", None)),
            "output_text": direct_output_text,
            "output_text_length": len(direct_output_text),
            "output_text_fragments": text_fragments or None,
            "output_text_fragment_count": len(text_fragments),
            "reconstructed_output_text": reconstructed_output_text or None,
            "reconstructed_output_text_length": len(reconstructed_output_text),
            "resolved_output_text": resolved_output_text or None,
            "resolved_output_text_length": len(resolved_output_text),
            "resolved_output_text_source": resolved_output_text_source,
            "output": output_payload,
        }
        return {
            key: value
            for key, value in response_metadata.items()
            if value is not None and value != []
        }

    def _compute_response_retry_delay(self, attempt_index: int) -> float:
        """Fixed short delay for response-quality retries (not congestion-based)."""
        return min(self.transport.max_backoff_seconds, self.transport.initial_backoff_seconds * self._uniform(0.5, 1.5))

    def _normalize_chat_messages(self, input_payload: Any) -> list[dict[str, Any]]:
        if isinstance(input_payload, list | tuple):
            normalized_messages: list[dict[str, Any]] = []
            for item in input_payload:
                if isinstance(item, Mapping) and "role" in item and "content" in item:
                    normalized_messages.append({"role": str(item["role"]), "content": item["content"]})
                else:
                    normalized_messages.append({"role": "user", "content": self._stringify_chat_payload(item)})
            return normalized_messages
        if isinstance(input_payload, Mapping) and "role" in input_payload and "content" in input_payload:
            return [{"role": str(input_payload["role"]), "content": input_payload["content"]}]
        return [{"role": "user", "content": self._stringify_chat_payload(input_payload)}]

    def _stringify_chat_payload(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(self._to_json_compatible(value), ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)

    def _extract_chat_completion_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list | tuple):
            fragments: list[str] = []
            for item in content:
                if isinstance(item, Mapping):
                    text = _coerce_optional_string(item.get("text"), default="") or ""
                    if text:
                        fragments.append(text)
            return "".join(fragments).strip()
        return _coerce_optional_string(content, default="") or ""

    def _extract_chat_completion_reasoning_text(self, message: Any) -> str:
        if message is None:
            return ""

        for attr_name in ("reasoning", "reasoning_content"):
            fragments = self._collect_text_fragments(getattr(message, attr_name, None), allow_plain_text=True)
            text = "".join(fragment for fragment in fragments if fragment).strip()
            if text:
                return text

        message_payload = self._to_json_compatible(message)
        if isinstance(message_payload, Mapping):
            for key in ("reasoning", "reasoning_content"):
                fragments = self._collect_text_fragments(message_payload.get(key), allow_plain_text=True)
                text = "".join(fragment for fragment in fragments if fragment).strip()
                if text:
                    return text

        return ""

    def _to_json_compatible(self, value: Any) -> Any:
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): self._to_json_compatible(nested_value)
                for key, nested_value in value.items()
            }
        if isinstance(value, list | tuple):
            return [self._to_json_compatible(item) for item in value]
        if hasattr(value, "model_dump"):
            try:
                return self._to_json_compatible(value.model_dump(mode="json"))
            except TypeError:
                return self._to_json_compatible(value.model_dump())
        if hasattr(value, "dict"):
            return self._to_json_compatible(value.dict())
        return str(value)

    def _collect_text_fragments(self, value: Any, *, allow_plain_text: bool = False) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if allow_plain_text and stripped else []
        if isinstance(value, Mapping):
            fragments: list[str] = []
            text_value = value.get("text")
            if text_value is not None:
                fragments.extend(self._collect_text_fragments(text_value, allow_plain_text=True))

            content_value = value.get("content")
            if content_value is not None:
                fragments.extend(self._collect_text_fragments(content_value, allow_plain_text=True))

            for key, nested_value in value.items():
                if key in {"text", "content"}:
                    continue
                if allow_plain_text and key == "value":
                    fragments.extend(self._collect_text_fragments(nested_value, allow_plain_text=True))
                    continue
                fragments.extend(self._collect_text_fragments(nested_value, allow_plain_text=False))
            return fragments
        if isinstance(value, list | tuple):
            fragments: list[str] = []
            for item in value:
                fragments.extend(self._collect_text_fragments(item, allow_plain_text=allow_plain_text))
            return fragments
        return []

    def _retry_after_seconds_from_headers(self, headers: Any) -> float | None:
        if headers is None:
            return None

        retry_after_value: Any = None
        if isinstance(headers, Mapping):
            retry_after_value = headers.get("retry-after") or headers.get("Retry-After")
        else:
            getter = getattr(headers, "get", None)
            if callable(getter):
                retry_after_value = getter("retry-after") or getter("Retry-After")

        if retry_after_value is None:
            return None

        try:
            return max(0.0, float(str(retry_after_value).strip()))
        except (TypeError, ValueError):
            return None

    def _resolve_retry_delay_seconds(self, *, error: BlackOPDClientError, attempt_index: int) -> float:
        retry_after_seconds = _coerce_optional_float(error.details.get("retry_after_seconds"), default=None)
        if retry_after_seconds is not None:
            return max(0.0, retry_after_seconds)
        return self._compute_backoff_seconds(attempt_index)

    def _execute_with_retry(
        self,
        *,
        stage: Literal["teacher", "rubricator", "verifier"],
        role: OpenAIRoleConfig,
        input_payload: Any,
        text_format: dict[str, Any] | None,
        request_metadata: dict[str, Any],
        request: Any,
    ) -> Any:
        max_attempts = self.transport.max_retries + 1
        last_error: BlackOPDClientError | None = None

        for attempt_index in range(max_attempts):
            try:
                self._before_request(
                    stage=stage,
                    role=role,
                    input_payload=input_payload,
                    text_format=text_format,
                )
                with self._shared_resources.semaphore:
                    response = request(self._get_client(role))
                if getattr(response, "error", None) is not None:
                    raise BlackOPDClientError(
                        stage=stage,
                        error_type="http_error",
                        message=f"{stage} response contained an API-side error: {response.error}",
                        retriable=False,
                        details={
                            "request": self._build_request_metadata(role=role, text_format=text_format),
                            "response": self._build_response_metadata(response),
                        },
                    )
                return response
            except BlackOPDClientError as exc:
                last_error = exc
            except APITimeoutError as exc:
                last_error = BlackOPDClientError(
                    stage=stage,
                    error_type="timeout",
                    message=str(exc),
                    retriable=True,
                    details={"request": copy.deepcopy(request_metadata)},
                )
            except APIConnectionError as exc:
                last_error = BlackOPDClientError(
                    stage=stage,
                    error_type="http_error",
                    message=str(exc),
                    retriable=True,
                    details={"request": copy.deepcopy(request_metadata)},
                )
            except APIStatusError as exc:
                status_code = getattr(exc, "status_code", None)
                response = getattr(exc, "response", None)
                retry_after_seconds = self._retry_after_seconds_from_headers(
                    None if response is None else response.headers
                )
                error_details: dict[str, Any] = {"request": copy.deepcopy(request_metadata)}
                if retry_after_seconds is not None:
                    error_details["retry_after_seconds"] = retry_after_seconds
                last_error = BlackOPDClientError(
                    stage=stage,
                    error_type="http_error",
                    message=str(exc),
                    status_code=status_code,
                    retriable=status_code in TRANSIENT_HTTP_STATUS_CODES,
                    details=error_details,
                )

            if last_error is None:
                continue
            self._after_request_failure(stage=stage, error=last_error)
            if not last_error.retriable or attempt_index == max_attempts - 1:
                raise last_error
            self._record_metric(stage=stage, name="retries")
            self._sleep(self._resolve_retry_delay_seconds(error=last_error, attempt_index=attempt_index))

        raise last_error or RuntimeError("unreachable")

    def _compute_backoff_seconds(self, attempt_index: int) -> float:
        base_delay = self.transport.initial_backoff_seconds * (self.transport.backoff_multiplier**attempt_index)
        jitter = self._uniform(0.5, 1.5)
        return min(self.transport.max_backoff_seconds, base_delay * jitter)

    def _get_client(self, role: OpenAIRoleConfig) -> Any:
        cache_key = (role.api_key, role.base_url, role.timeout_seconds)
        with self._shared_resources.client_cache_lock:
            client = self._shared_resources.client_cache.get(cache_key)
            if client is None:
                client = self._client_factory(
                    api_key=role.api_key,
                    base_url=role.base_url,
                    timeout=role.timeout_seconds,
                    max_retries=0,
                )
                self._shared_resources.client_cache[cache_key] = client
            return client


def _json_schema_for_model(model: type[BaseModel], *, name: str) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": name,
        "schema": model.model_json_schema(),
        "strict": True,
    }


def _strip_markdown_json_fence(raw_text: str) -> str:
    stripped_text = raw_text.strip()
    if not stripped_text.startswith("```"):
        return raw_text

    first_newline_index = stripped_text.find("\n")
    if first_newline_index < 0:
        return raw_text

    fence_language = stripped_text[3:first_newline_index].strip().lower()
    if fence_language not in {"", "json"}:
        return raw_text

    fenced_body = stripped_text[first_newline_index + 1 :]
    if not fenced_body.endswith("```"):
        return raw_text
    return fenced_body[:-3].strip()


def _parse_json_payload(
    raw_text: str,
    *,
    stage: Literal["rubricator", "verifier"],
) -> dict[str, Any]:
    normalized_raw_text = _strip_markdown_json_fence(raw_text)
    try:
        payload = json.loads(normalized_raw_text)
    except json.JSONDecodeError as exc:
        raise BlackOPDClientError(
            stage=stage,
            error_type="parse_error",
            message=f"{stage} returned invalid JSON: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise BlackOPDClientError(
            stage=stage,
            error_type="schema_error",
            message=f"{stage} returned a non-object JSON payload.",
        )
    return payload


def _validate_structured_rubric(rubric: BlackOPDStructuredRubric) -> BlackOPDStructuredRubric:
    if len(rubric.rubrics) < MIN_RUBRIC_CRITERIA or len(rubric.rubrics) > MAX_RUBRIC_CRITERIA:
        raise BlackOPDClientError(
            stage="rubricator",
            error_type="validation_error",
            message=(
                "rubricator returned an invalid number of rubric criteria "
                f"(expected {MIN_RUBRIC_CRITERIA} to {MAX_RUBRIC_CRITERIA})."
            ),
        )
    criterion_ids = [criterion.criterion_id for criterion in rubric.rubrics]
    if len(set(criterion_ids)) != len(criterion_ids):
        raise BlackOPDClientError(
            stage="rubricator",
            error_type="validation_error",
            message="rubricator returned duplicate criterion_id values.",
        )
    if rubric.maximum_score != rubric.total_points:
        rubric = rubric.model_copy(update={"maximum_score": rubric.total_points})
    return rubric


def _normalize_structured_rubric_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized_payload = copy.deepcopy(payload)
    rubrics = normalized_payload.get("rubrics")
    if isinstance(rubrics, list):
        for criterion in rubrics:
            if not isinstance(criterion, dict):
                continue
            criterion_id = criterion.get("criterion_id")
            if isinstance(criterion_id, str):
                stripped_id = criterion_id.strip()
                if len(stripped_id) > 1 and stripped_id[0].lower() == "c" and stripped_id[1:].isdigit():
                    criterion["criterion_id"] = stripped_id.lower()
    return normalized_payload


def _parse_structured_rubric(raw_text: str) -> BlackOPDStructuredRubric:
    payload = _normalize_structured_rubric_payload(_parse_json_payload(raw_text, stage="rubricator"))
    try:
        rubric = BlackOPDStructuredRubric.model_validate(payload)
    except ValidationError as exc:
        raise BlackOPDClientError(
            stage="rubricator",
            error_type="schema_error",
            message=f"rubricator payload does not match the rubric schema: {exc}",
        ) from exc
    return _validate_structured_rubric(rubric)


def _validate_verifier_score(
    score: BlackOPDVerifierScore,
    *,
    rubric: BlackOPDStructuredRubric,
) -> BlackOPDVerifierScore:
    if len(score.judgement) != len(rubric.rubrics):
        raise BlackOPDClientError(
            stage="verifier",
            error_type="validation_error",
            message="verifier judgement length does not match rubric length.",
        )
    recomputed_final_score = sum(
        criterion.points for criterion, judgement in zip(rubric.rubrics, score.judgement, strict=True) if judgement
    )
    if float(score.final_score) != float(recomputed_final_score):
        score = score.model_copy(update={"final_score": float(recomputed_final_score)})
    return score


def _parse_verifier_score(raw_text: str, *, rubric: BlackOPDStructuredRubric) -> BlackOPDVerifierScore:
    payload = _parse_json_payload(raw_text, stage="verifier")
    try:
        score = BlackOPDVerifierScore.model_validate(payload)
    except ValidationError as exc:
        raise BlackOPDClientError(
            stage="verifier",
            error_type="schema_error",
            message=f"verifier payload does not match the verifier schema: {exc}",
        ) from exc
    return _validate_verifier_score(score, rubric=rubric)


class OpenAITeacherClient:
    def __init__(self, *, provider: OpenAICompatibleProvider, role_config: OpenAIRoleConfig) -> None:
        self.provider = provider
        self.role_config = role_config

    def generate(self, raw_prompt: Any, *, uid: str | None = None) -> str:
        try:
            input_messages = build_teacher_input_messages(raw_prompt)
        except (TypeError, ValueError) as exc:
            raise BlackOPDClientError(
                stage="teacher",
                error_type="validation_error",
                message=f"teacher prompt construction failed: {exc}",
            ) from exc
        try:
            return self.provider.create_text(stage="teacher", role=self.role_config, input_payload=input_messages)
        except BlackOPDClientError as exc:
            exc.add_context(uid=uid)
            raise


class OpenAIRubricatorClient:
    def __init__(self, *, provider: OpenAICompatibleProvider, role_config: OpenAIRoleConfig) -> None:
        self.provider = provider
        self.role_config = role_config

    def generate(
        self,
        raw_prompt: Any,
        teacher_response: str,
        student_response: str,
        *,
        uid: str | None = None,
        pair_index: int | None = None,
    ) -> BlackOPDStructuredRubric:
        try:
            prompt_text = build_rubricator_prompt(
                raw_prompt,
                teacher_response=teacher_response,
                student_response=student_response,
            )
        except (TypeError, ValueError) as exc:
            raise BlackOPDClientError(
                stage="rubricator",
                error_type="validation_error",
                message=f"rubricator prompt construction failed: {exc}",
            ) from exc

        try:
            return self.provider.create_text(
                stage="rubricator",
                role=self.role_config,
                input_payload=prompt_text,
                text_format=_json_schema_for_model(BlackOPDStructuredRubric, name="ropd_rubric"),
                output_validator=_parse_structured_rubric,
            )
        except BlackOPDClientError as exc:
            exc.add_context(uid=uid, pair_index=pair_index)
            raise


class OpenAIVerifierClient:
    def __init__(self, *, provider: OpenAICompatibleProvider, role_config: OpenAIRoleConfig) -> None:
        self.provider = provider
        self.role_config = role_config

    def score(
        self,
        raw_prompt: Any,
        rubric: BlackOPDStructuredRubric,
        answer: str,
        *,
        uid: str | None = None,
        pair_index: int | None = None,
        subject: str | None = None,
    ) -> BlackOPDVerifierScore:
        try:
            prompt_text = build_verifier_prompt(
                raw_prompt,
                response=answer,
                rubrics=[criterion.model_dump(mode="json") for criterion in rubric.rubrics],
                model=self.role_config.model,
            )
        except (TypeError, ValueError) as exc:
            raise BlackOPDClientError(
                stage="verifier",
                error_type="validation_error",
                message=f"verifier prompt construction failed: {exc}",
            ) from exc

        try:
            return self.provider.create_text(
                stage="verifier",
                role=self.role_config,
                input_payload=prompt_text,
                text_format=_json_schema_for_model(BlackOPDVerifierScore, name="ropd_verifier"),
                output_validator=lambda raw_text: _parse_verifier_score(raw_text, rubric=rubric),
            )
        except BlackOPDClientError as exc:
            exc.add_context(uid=uid, pair_index=pair_index, subject=subject)
            raise


def _build_static_rubric_criteria(maximum_score: int) -> list[BlackOPDRubricCriterion]:
    return [
        BlackOPDRubricCriterion(
            criterion_id=f"c{index}",
            category="Correctness",
            criterion=f"criterion {index}",
            points=1,
        )
        for index in range(1, maximum_score + 1)
    ]


def _resolve_static_student_score(maximum_score: int, pair_index: int | None) -> int:
    max_student_score = max(1, maximum_score // 2 - 1)
    base_score = 2 + max(pair_index or 0, 0)
    return min(base_score, max_student_score)


class StaticTeacherClient:
    def __init__(self, *, debug_config: BlackOPDDebugConfig, role_config: OpenAIRoleConfig) -> None:
        self.debug_config = debug_config
        self.role_config = role_config

    def generate(self, raw_prompt: Any, *, uid: str | None = None) -> str:
        del raw_prompt, uid
        return self.debug_config.static_teacher_response


class StaticRubricatorClient:
    def __init__(self, *, debug_config: BlackOPDDebugConfig, role_config: OpenAIRoleConfig) -> None:
        self.debug_config = debug_config
        self.role_config = role_config
        self._criteria = _build_static_rubric_criteria(self.debug_config.static_maximum_score)

    def generate(
        self,
        raw_prompt: Any,
        teacher_response: str,
        student_response: str,
        *,
        uid: str | None = None,
        pair_index: int | None = None,
    ) -> BlackOPDStructuredRubric:
        del raw_prompt, teacher_response, student_response, uid
        rubric = BlackOPDStructuredRubric(
            schema_version=RUBRIC_SCHEMA_VERSION,
            rubrics=self._criteria,
            maximum_score=self.debug_config.static_maximum_score,
        )
        return _validate_structured_rubric(rubric)


class StaticVerifierClient:
    def __init__(self, *, debug_config: BlackOPDDebugConfig, role_config: OpenAIRoleConfig) -> None:
        self.debug_config = debug_config
        self.role_config = role_config

    def score(
        self,
        raw_prompt: Any,
        rubric: BlackOPDStructuredRubric,
        answer: str,
        *,
        uid: str | None = None,
        pair_index: int | None = None,
        subject: str | None = None,
    ) -> BlackOPDVerifierScore:
        del raw_prompt, answer, uid
        student_score = _resolve_static_student_score(rubric.maximum_score, pair_index)
        teacher_score = 1 if student_score > 0 else 0
        resolved_score = teacher_score if subject == "teacher" else student_score
        judgement = [True] * int(resolved_score) + [False] * (len(rubric.rubrics) - int(resolved_score))
        score = BlackOPDVerifierScore(
            schema_version=VERIFIER_SCHEMA_VERSION,
            judgement=judgement,
            final_score=float(resolved_score),
        )
        return _validate_verifier_score(score, rubric=rubric)


def _build_role_client(
    *,
    stage: Literal["teacher", "rubricator", "verifier"],
    role_config: OpenAIRoleConfig,
    debug_config: BlackOPDDebugConfig,
    provider: OpenAICompatibleProvider | None,
) -> Any:
    if role_config.provider == "offline_index":
        if stage != "teacher":
            raise ValueError("offline_index is only supported for teacher.")
        fingerprint = build_teacher_fingerprint_payload(
            provider="openai_compatible",
            model=role_config.model,
            base_url=role_config.base_url,
            reasoning_effort=role_config.reasoning_effort,
            max_output_tokens=role_config.max_output_tokens,
            temperature=role_config.temperature,
            top_p=role_config.top_p,
            timeout_seconds=role_config.timeout_seconds,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
        teacher_index = OfflineTeacherIndex.load(
            index_path=role_config.index_path,
            expected_fingerprint=fingerprint,
        )
        return OfflineTeacherIndexClient(teacher_index=teacher_index)

    if role_config.provider == "openai_compatible":
        if provider is None:
            raise ValueError("OpenAI-compatible ROPD roles require an initialized provider.")
        if stage == "teacher":
            return OpenAITeacherClient(provider=provider, role_config=role_config)
        if stage == "rubricator":
            return OpenAIRubricatorClient(provider=provider, role_config=role_config)
        return OpenAIVerifierClient(provider=provider, role_config=role_config)

    if role_config.provider == "static":
        if stage == "teacher":
            return StaticTeacherClient(debug_config=debug_config, role_config=role_config)
        if stage == "rubricator":
            return StaticRubricatorClient(debug_config=debug_config, role_config=role_config)
        return StaticVerifierClient(debug_config=debug_config, role_config=role_config)

    raise ValueError(f"Unsupported ROPD role provider {role_config.provider!r}.")


def build_ropd_pipeline(
    config: BlackOPDClientConfig | dict[str, Any] | None = None,
    *,
    provider: OpenAICompatibleProvider | None = None,
) -> Any:
    from algo.ropd_pipeline import BlackOPDPipeline

    resolved_config = config if isinstance(config, BlackOPDClientConfig) else build_ropd_client_config(config)
    requires_openai_provider = any(
        role.provider == "openai_compatible"
        for role in (resolved_config.teacher, resolved_config.rubricator, resolved_config.verifier)
    )
    resolved_provider = (
        provider
        or (
            OpenAICompatibleProvider(
                resolved_config.transport,
                provider_limits=resolved_config.provider_limits,
                request_scheduler_config=resolved_config.request_scheduler,
            )
            if requires_openai_provider
            else None
        )
    )
    return BlackOPDPipeline(
        teacher_client=_build_role_client(
            stage="teacher",
            role_config=resolved_config.teacher,
            debug_config=resolved_config.debug,
            provider=resolved_provider,
        ),
        rubric_client=_build_role_client(
            stage="rubricator",
            role_config=resolved_config.rubricator,
            debug_config=resolved_config.debug,
            provider=resolved_provider,
        ),
        verifier_client=_build_role_client(
            stage="verifier",
            role_config=resolved_config.verifier,
            debug_config=resolved_config.debug,
            provider=resolved_provider,
        ),
        max_pair_concurrency=resolved_config.max_pair_concurrency,
        max_verifier_subject_concurrency=resolved_config.max_verifier_subject_concurrency,
        artifact_exporter=(
            BlackOPDArtifactExporter(resolved_config.export) if resolved_config.export.enabled else None
        ),
    )


__all__ = [
    "BlackOPDClientConfig",
    "BlackOPDClientError",
    "BlackOPDDebugConfig",
    "BlackOPDProviderCircuitBreakerConfig",
    "BlackOPDProviderLimitsConfig",
    "BlackOPDRequestSchedulerConfig",
    "BlackOPDStageBreakerConfigSet",
    "MAX_RUBRIC_CRITERIA",
    "MIN_RUBRIC_CRITERIA",
    "BlackOPDRubricCriterion",
    "BlackOPDStructuredRubric",
    "BlackOPDVerifierScore",
    "BlackOPDExportConfig",
    "OpenAICompatibleProvider",
    "OpenAIRoleConfig",
    "OpenAIRubricatorClient",
    "OpenAITeacherClient",
    "OpenAITransportConfig",
    "OpenAIVerifierClient",
    "PROMPT_TEMPLATE_VERSION",
    "RUBRIC_SCHEMA_VERSION",
    "TRANSIENT_HTTP_STATUS_CODES",
    "VERIFIER_SCHEMA_VERSION",
    "build_ropd_client_config",
    "build_ropd_pipeline",
]
