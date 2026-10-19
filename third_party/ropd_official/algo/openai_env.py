from __future__ import annotations

import os

SUPPORTED_OPENAI_PROFILES = frozenset({"A", "B"})


def _read_non_empty_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def apply_selected_openai_profile_to_environment() -> str | None:
    profile = _read_non_empty_env("OPENAI_PROFILE")
    if profile is None:
        return None

    normalized_profile = profile.upper()
    if normalized_profile not in SUPPORTED_OPENAI_PROFILES:
        supported_profiles = ", ".join(sorted(SUPPORTED_OPENAI_PROFILES))
        raise ValueError(
            f"OPENAI_PROFILE must be one of [{supported_profiles}], got {profile!r}."
        )

    api_key_env = f"OPENAI_{normalized_profile}_API_KEY"
    base_url_env = f"OPENAI_{normalized_profile}_BASE_URL"
    api_key = _read_non_empty_env(api_key_env)
    base_url = _read_non_empty_env(base_url_env)

    missing_envs: list[str] = []
    if api_key is None:
        missing_envs.append(api_key_env)
    if base_url is None:
        missing_envs.append(base_url_env)
    if missing_envs:
        missing_names = ", ".join(missing_envs)
        raise ValueError(
            f"OPENAI_PROFILE={normalized_profile} requires non-empty values for [{missing_names}]."
        )

    os.environ["OPENAI_PROFILE"] = normalized_profile
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = base_url
    return normalized_profile
