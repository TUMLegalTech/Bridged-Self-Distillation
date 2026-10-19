from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


class BlackOPDJudgeProviderResolverError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ResolvedJudgeRole:
    provider: str
    profile: str | None
    model: str
    api_style: str
    reasoning_effort: str | None
    timeout_seconds: float
    max_output_tokens: int | None
    temperature: float | None
    top_p: float | None
    api_key: str | None
    base_url: str | None
    index_path: str | None
    response_retry: dict[str, int]
    transport: dict[str, Any]
    provider_limits: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ResolvedJudgeRoles:
    teacher: ResolvedJudgeRole
    rubricator: ResolvedJudgeRole
    verifier: ResolvedJudgeRole


@dataclass(frozen=True, slots=True)
class ResolvedJudgeProviderConfig:
    roles: ResolvedJudgeRoles
    sources: dict[str, str]


ROLE_NAMES: tuple[str, ...] = ("teacher", "rubricator", "verifier")
LEGACY_ENV_KEYS = {
    "OPENAI_PROFILE": "ROPD_JUDGE_PROFILE",
    "ROPD_PROVIDER_MODE": "ROPD_JUDGE_PROVIDER",
    "ROPD_API_MAX_OUTPUT_TOKENS": "ROPD_JUDGE_MAX_OUTPUT_TOKENS",
}


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def _normalize_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    raise BlackOPDJudgeProviderResolverError(f"Expected mapping, got {type(value).__name__}.")


def _env_present(environ: dict[str, str], name: str) -> bool:
    return name in environ


def _env_value(environ: dict[str, str], name: str) -> str | None:
    if name not in environ:
        return None
    return environ[name]


def _set_field(
    field_values: dict[str, Any],
    field_sources: dict[str, str],
    *,
    field_name: str,
    value: Any,
    source: str,
) -> None:
    field_values[field_name] = value
    field_sources[field_name] = source


def _build_role_env_key(role_name: str, field_name: str) -> str:
    return f"ROPD_{role_name.upper()}_{field_name.upper()}"


def _apply_mapping(
    field_values: dict[str, Any],
    field_sources: dict[str, str],
    mapping: dict[str, Any],
    *,
    source_prefix: str,
    include_field_name: bool = True,
) -> None:
    for key, value in mapping.items():
        source = f"{source_prefix}.{key}" if include_field_name else source_prefix
        _set_field(field_values, field_sources, field_name=key, value=value, source=source)


def _resolve_profile_name(
    *,
    environ: dict[str, str],
    global_overrides: dict[str, Any],
    role_overrides: dict[str, Any],
) -> tuple[str | None, str | None]:
    role_profile_key = role_overrides.get("profile")
    if role_profile_key is not None:
        return _normalize_optional_string(role_profile_key), "override:role.profile"

    role_profile_env = _build_role_env_key(role_overrides["__role_name__"], "profile")
    if _env_present(environ, role_profile_env):
        return _normalize_optional_string(_env_value(environ, role_profile_env)), f"env:{role_profile_env}"

    if global_overrides.get("profile") is not None:
        return _normalize_optional_string(global_overrides["profile"]), "override:global.profile"

    if _env_present(environ, "ROPD_JUDGE_PROFILE"):
        return _normalize_optional_string(_env_value(environ, "ROPD_JUDGE_PROFILE")), "env:ROPD_JUDGE_PROFILE"

    if _env_present(environ, "OPENAI_PROFILE"):
        return _normalize_optional_string(_env_value(environ, "OPENAI_PROFILE")), "legacy_env:OPENAI_PROFILE"

    return None, None


def _resolve_role(
    *,
    role_name: str,
    spec: dict[str, Any],
    entrypoint: str,
    overrides: dict[str, Any],
    environ: dict[str, str],
    sources: dict[str, str],
) -> ResolvedJudgeRole:
    profiles = _coerce_mapping(spec.get("profiles"))
    role_defaults = _coerce_mapping(_coerce_mapping(spec.get("roles")).get(role_name))
    entrypoint_role = _coerce_mapping(_coerce_mapping(_coerce_mapping(spec.get("entrypoints")).get(entrypoint)).get(role_name))
    global_overrides = _coerce_mapping(overrides.get("global"))
    role_overrides = _coerce_mapping(overrides.get(role_name))
    role_overrides["__role_name__"] = role_name

    field_values: dict[str, Any] = {}
    field_sources: dict[str, str] = {}

    _apply_mapping(field_values, field_sources, role_defaults, source_prefix=f"yaml:roles.{role_name}")
    _apply_mapping(
        field_values,
        field_sources,
        entrypoint_role,
        source_prefix=f"entrypoint:{entrypoint}",
        include_field_name=False,
    )

    if global_overrides.get("provider") is not None:
        _set_field(
            field_values,
            field_sources,
            field_name="provider",
            value=global_overrides["provider"],
            source="override:global.provider",
        )
    elif _env_present(environ, "ROPD_JUDGE_PROVIDER"):
        _set_field(
            field_values,
            field_sources,
            field_name="provider",
            value=_env_value(environ, "ROPD_JUDGE_PROVIDER"),
            source="env:ROPD_JUDGE_PROVIDER",
        )
    elif _env_present(environ, "ROPD_PROVIDER_MODE"):
        _set_field(
            field_values,
            field_sources,
            field_name="provider",
            value=_env_value(environ, "ROPD_PROVIDER_MODE"),
            source="legacy_env:ROPD_PROVIDER_MODE",
        )

    role_provider_env = _build_role_env_key(role_name, "provider")
    if role_overrides.get("provider") is not None:
        _set_field(field_values, field_sources, field_name="provider", value=role_overrides["provider"], source="override:role.provider")
    elif _env_present(environ, role_provider_env):
        _set_field(
            field_values,
            field_sources,
            field_name="provider",
            value=_env_value(environ, role_provider_env),
            source=f"env:{role_provider_env}",
        )

    profile_name, profile_source = _resolve_profile_name(
        environ=environ,
        global_overrides=global_overrides,
        role_overrides=role_overrides,
    )
    if profile_name is not None:
        _set_field(field_values, field_sources, field_name="profile", value=profile_name, source=profile_source or "")

    global_max_tokens = None
    global_max_tokens_source = None
    if global_overrides.get("max_output_tokens") is not None:
        global_max_tokens = global_overrides["max_output_tokens"]
        global_max_tokens_source = "override:global.max_output_tokens"
    elif _env_present(environ, "ROPD_JUDGE_MAX_OUTPUT_TOKENS"):
        global_max_tokens = _env_value(environ, "ROPD_JUDGE_MAX_OUTPUT_TOKENS")
        global_max_tokens_source = "env:ROPD_JUDGE_MAX_OUTPUT_TOKENS"
    elif _env_present(environ, "ROPD_API_MAX_OUTPUT_TOKENS"):
        global_max_tokens = _env_value(environ, "ROPD_API_MAX_OUTPUT_TOKENS")
        global_max_tokens_source = "legacy_env:ROPD_API_MAX_OUTPUT_TOKENS"
    if global_max_tokens is not None:
        _set_field(
            field_values,
            field_sources,
            field_name="max_output_tokens",
            value=global_max_tokens,
            source=global_max_tokens_source or "",
        )

    env_field_normalizers = {
        "model": _normalize_optional_string,
        "api_style": _normalize_optional_string,
        "reasoning_effort": _normalize_optional_string,
        "api_key": _normalize_optional_string,
        "timeout_seconds": _normalize_optional_float,
        "max_output_tokens": _normalize_optional_int,
        "temperature": _normalize_optional_float,
        "top_p": _normalize_optional_float,
        "index_path": _normalize_optional_string,
        "base_url": _normalize_optional_string,
        "empty_response_retries": _normalize_optional_int,
        "incomplete_retries": _normalize_optional_int,
        "parse_error_retries": _normalize_optional_int,
        "schema_error_retries": _normalize_optional_int,
        "validation_error_retries": _normalize_optional_int,
    }
    for field_name, normalizer in env_field_normalizers.items():
        role_env_key = _build_role_env_key(role_name, field_name)
        if field_name in role_overrides:
            _set_field(
                field_values,
                field_sources,
                field_name=field_name,
                value=normalizer(role_overrides[field_name]),
                source=f"override:role.{field_name}",
            )
        elif _env_present(environ, role_env_key):
            _set_field(
                field_values,
                field_sources,
                field_name=field_name,
                value=normalizer(_env_value(environ, role_env_key)),
                source=f"env:{role_env_key}",
            )

    profile_name = _normalize_optional_string(field_values.get("profile"))
    profile_spec = _coerce_mapping(profiles.get(profile_name)) if profile_name is not None else {}
    if profile_name is not None and not profile_spec:
        raise BlackOPDJudgeProviderResolverError(f"Unknown judge provider profile: {profile_name}")

    provider = _normalize_optional_string(field_values.get("provider")) or _normalize_optional_string(profile_spec.get("provider"))
    if provider is None:
        raise BlackOPDJudgeProviderResolverError(f"{role_name}.provider is required.")

    model = _normalize_optional_string(field_values.get("model"))
    if model is None:
        raise BlackOPDJudgeProviderResolverError(f"{role_name}.model is required.")

    api_style = _normalize_optional_string(field_values.get("api_style"))
    if api_style is None:
        api_style = _normalize_optional_string(profile_spec.get("default_api_style")) or "responses"
        sources.setdefault(f"{role_name}.api_style", f"profile:{profile_name}.default_api_style" if profile_name else "default:responses")

    reasoning_effort = _normalize_optional_string(field_values.get("reasoning_effort"))
    timeout_seconds = _normalize_optional_float(field_values.get("timeout_seconds"))
    if timeout_seconds is None:
        raise BlackOPDJudgeProviderResolverError(f"{role_name}.timeout_seconds is required.")

    max_output_tokens = _normalize_optional_int(field_values.get("max_output_tokens"))
    temperature = _normalize_optional_float(field_values.get("temperature"))
    top_p = _normalize_optional_float(field_values.get("top_p"))

    response_retry = copy.deepcopy(_coerce_mapping(field_values.get("response_retry")))
    for field_name in (
        "empty_response_retries",
        "incomplete_retries",
        "parse_error_retries",
        "schema_error_retries",
        "validation_error_retries",
    ):
        field_value = _normalize_optional_int(field_values.get(field_name))
        if field_value is not None:
            response_retry[field_name] = field_value
    transport = copy.deepcopy(_coerce_mapping(profile_spec.get("transport")))
    provider_limits = copy.deepcopy(_coerce_mapping(profile_spec.get("provider_limits")))

    base_url = _normalize_optional_string(field_values.get("base_url"))
    if base_url is None:
        base_url = _normalize_optional_string(profile_spec.get("base_url"))
    if base_url is None:
        base_url_env_name = _normalize_optional_string(profile_spec.get("base_url_env"))
        if base_url_env_name is not None:
            base_url = _normalize_optional_string(_env_value(environ, base_url_env_name))

    api_key = _normalize_optional_string(field_values.get("api_key"))
    if api_key is None:
        api_key_env_name = _normalize_optional_string(profile_spec.get("api_key_env"))
        if api_key_env_name is not None:
            api_key = _normalize_optional_string(_env_value(environ, api_key_env_name))

    index_path = _normalize_optional_string(field_values.get("index_path"))
    if provider == "offline_index" and index_path is None:
        default_index_path = _normalize_optional_string(role_defaults.get("index_path"))
        if _normalize_optional_string(field_sources.get("index_path")) == f"entrypoint:{entrypoint}":
            index_path = default_index_path
        if index_path is None:
            raise BlackOPDJudgeProviderResolverError(f"{role_name}.index_path is required when provider=offline_index.")

    for field_name, source in field_sources.items():
        sources[f"{role_name}.{field_name}"] = source
    if f"{role_name}.api_style" not in sources:
        sources[f"{role_name}.api_style"] = (
            f"profile:{profile_name}.default_api_style" if profile_name else "default:responses"
        )

    return ResolvedJudgeRole(
        provider=provider,
        profile=profile_name,
        model=model,
        api_style=api_style,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key,
        base_url=base_url,
        index_path=index_path,
        response_retry={key: int(value) for key, value in response_retry.items()},
        transport=transport,
        provider_limits=provider_limits,
    )


def _resolve_judge_provider_config(
    spec: dict[str, Any],
    *,
    entrypoint: str,
    overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> ResolvedJudgeProviderConfig:
    resolved_overrides = {} if overrides is None else dict(overrides)
    resolved_environ = dict(os.environ if environ is None else environ)
    sources: dict[str, str] = {}
    resolved_roles = {
        role_name: _resolve_role(
            role_name=role_name,
            spec=spec,
            entrypoint=entrypoint,
            overrides=resolved_overrides,
            environ=resolved_environ,
            sources=sources,
        )
        for role_name in ROLE_NAMES
    }
    return ResolvedJudgeProviderConfig(
        roles=ResolvedJudgeRoles(
            teacher=resolved_roles["teacher"],
            rubricator=resolved_roles["rubricator"],
            verifier=resolved_roles["verifier"],
        ),
        sources=sources,
    )


class BlackOPDJudgeProviderResolver:
    def __init__(
        self,
        *,
        spec_path: str | Path,
        entrypoint: str,
        overrides: dict[str, Any] | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.spec_path = Path(spec_path)
        self.entrypoint = entrypoint
        self.overrides = {} if overrides is None else dict(overrides)
        self.environ = environ

    def resolve(self) -> ResolvedJudgeProviderConfig:
        if not self.spec_path.exists():
            raise BlackOPDJudgeProviderResolverError(f"Judge provider spec not found: {self.spec_path}")
        raw_spec = OmegaConf.to_container(OmegaConf.load(self.spec_path), resolve=True)
        if not isinstance(raw_spec, dict):
            raise BlackOPDJudgeProviderResolverError("Judge provider spec must resolve to a mapping.")
        return _resolve_judge_provider_config(
            raw_spec,
            entrypoint=self.entrypoint,
            overrides=self.overrides,
            environ=self.environ,
        )


__all__ = [
    "BlackOPDJudgeProviderResolver",
    "BlackOPDJudgeProviderResolverError",
    "ResolvedJudgeProviderConfig",
    "ResolvedJudgeRole",
    "ResolvedJudgeRoles",
]
