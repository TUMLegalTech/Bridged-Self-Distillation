from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any

OFFLINE_TEACHER_INDEX_SCHEMA_VERSION = "ropd.teacher_index.v1"
OFFLINE_TEACHER_MULTI_ANSWER_SCHEMA_VERSION = "ropd.teacher_index.v2"


class BlackOPDTeacherIndexError(RuntimeError):
    pass


def _stable_json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_canonical_raw_prompt(raw_prompt: Any) -> str:
    from algo.ropd_pipeline import canonicalize_raw_prompt

    return _sha256_text(canonicalize_raw_prompt(raw_prompt))


def hash_teacher_answer(teacher_answer: str) -> str:
    return _sha256_text(teacher_answer)


def hash_teacher_answers(teacher_answers: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(hash_teacher_answer(answer) for answer in teacher_answers)


def build_teacher_fingerprint_payload(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    reasoning_effort: str | None,
    max_output_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    timeout_seconds: float,
    prompt_template_version: str,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "timeout_seconds": timeout_seconds,
        "prompt_template_version": prompt_template_version,
    }


def hash_teacher_fingerprint(payload: Mapping[str, Any]) -> str:
    return _sha256_text(_stable_json_dumps(payload))


@dataclass(frozen=True, slots=True)
class OfflineTeacherIndexRecord:
    uid: str
    raw_prompt_hash: str
    teacher_answers: tuple[str, ...]
    teacher_fingerprint: dict[str, Any]

    @property
    def teacher_answer(self) -> str:
        return self.teacher_answers[0]


class OfflineTeacherIndex:
    def __init__(
        self,
        *,
        index_path: Path,
        expected_fingerprint: Mapping[str, Any],
        records: dict[tuple[str, str], OfflineTeacherIndexRecord],
    ) -> None:
        self.index_path = index_path
        self.expected_fingerprint = dict(expected_fingerprint)
        self._records = records

    @classmethod
    def load(
        cls,
        *,
        index_path: str | Path,
        expected_fingerprint: Mapping[str, Any],
    ) -> "OfflineTeacherIndex":
        path = Path(index_path).expanduser()
        if not path.exists():
            raise BlackOPDTeacherIndexError(f"Offline teacher index not found: {path}")

        records: dict[tuple[str, str], OfflineTeacherIndexRecord] = {}
        expected_digest = hash_teacher_fingerprint(expected_fingerprint)
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw_line.strip():
                continue

            row = json.loads(raw_line)
            schema_version = row.get("schema_version")
            if schema_version not in {
                OFFLINE_TEACHER_INDEX_SCHEMA_VERSION,
                OFFLINE_TEACHER_MULTI_ANSWER_SCHEMA_VERSION,
            }:
                raise BlackOPDTeacherIndexError(
                    f"Unsupported offline teacher index schema at {path}:{line_number}: {schema_version!r}"
                )

            teacher_answers = cls._load_teacher_answers(row=row, path=path, line_number=line_number)

            record = OfflineTeacherIndexRecord(
                uid=str(row["uid"]),
                raw_prompt_hash=str(row["raw_prompt_hash"]),
                teacher_answers=teacher_answers,
                teacher_fingerprint=dict(row["teacher_fingerprint"]),
            )
            if hash_teacher_fingerprint(record.teacher_fingerprint) != expected_digest:
                raise BlackOPDTeacherIndexError(
                    f"Offline teacher index teacher fingerprint mismatch at {path}:{line_number}."
                )

            key = (record.uid, record.raw_prompt_hash)
            if key in records:
                raise BlackOPDTeacherIndexError(f"Duplicate offline teacher index key at {path}:{line_number}: {key!r}")
            records[key] = record

        return cls(index_path=path, expected_fingerprint=expected_fingerprint, records=records)

    def lookup(self, *, uid: str, raw_prompt: Any) -> str:
        return self.lookup_many(uid=uid, raw_prompt=raw_prompt, count=1)[0]

    def lookup_many(self, *, uid: str, raw_prompt: Any, count: int | None = None) -> tuple[str, ...]:
        key = (str(uid), hash_canonical_raw_prompt(raw_prompt))
        record = self._records.get(key)
        if record is None:
            raise BlackOPDTeacherIndexError(f"Offline teacher index miss for uid={uid!r}, raw_prompt_hash={key[1]!r}.")
        if count is None:
            return record.teacher_answers
        resolved_count = int(count)
        if resolved_count < 1:
            raise BlackOPDTeacherIndexError("Offline teacher lookup count must be positive.")
        if resolved_count > len(record.teacher_answers):
            raise BlackOPDTeacherIndexError(
                f"Offline teacher index requested {resolved_count} teacher answers for uid={uid!r}, "
                f"but only {len(record.teacher_answers)} are available."
            )
        return record.teacher_answers[:resolved_count]

    @staticmethod
    def _load_teacher_answers(*, row: dict[str, Any], path: Path, line_number: int) -> tuple[str, ...]:
        if row.get("schema_version") == OFFLINE_TEACHER_INDEX_SCHEMA_VERSION:
            teacher_answer = str(row["teacher_answer"]).strip()
            if not teacher_answer:
                raise BlackOPDTeacherIndexError(
                    f"Offline teacher index contains an empty teacher_answer at {path}:{line_number}."
                )
            return (teacher_answer,)

        raw_teacher_answers = row.get("teacher_answers")
        if not isinstance(raw_teacher_answers, list) or not raw_teacher_answers:
            raise BlackOPDTeacherIndexError(
                f"Offline teacher index requires a non-empty teacher_answers list at {path}:{line_number}."
            )
        teacher_answers = tuple(str(answer).strip() for answer in raw_teacher_answers)
        if any(not answer for answer in teacher_answers):
            raise BlackOPDTeacherIndexError(
                f"Offline teacher index contains an empty teacher_answers entry at {path}:{line_number}."
            )
        return teacher_answers


class OfflineTeacherIndexClient:
    def __init__(self, *, teacher_index: OfflineTeacherIndex) -> None:
        self.teacher_index = teacher_index

    def generate(self, raw_prompt: Any, *, uid: str | None = None) -> str:
        if uid is None or not str(uid).strip():
            raise BlackOPDTeacherIndexError("Offline teacher lookup requires a stable uid.")
        return self.teacher_index.lookup(uid=str(uid), raw_prompt=raw_prompt)

    def generate_many(self, raw_prompt: Any, *, uid: str | None = None, count: int | None = None) -> tuple[str, ...]:
        if uid is None or not str(uid).strip():
            raise BlackOPDTeacherIndexError("Offline teacher lookup requires a stable uid.")
        return self.teacher_index.lookup_many(uid=str(uid), raw_prompt=raw_prompt, count=count)
