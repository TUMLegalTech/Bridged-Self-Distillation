"""Legacy PPO trainer extensions for Bridged Self-Distillation."""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.bsd.cross_vocab_opd import CrossVocabOPDScorer
from verl.trainer.bsd.rlsd import compute_opd_pg_advantages, compute_truncation_aware_grpo_advantages
from verl.trainer.bsd.prompt_assembly import (
    DEFAULT_PI_TRACE_FIELD,
    PI_SOURCE_GROUND_TRUTH,
    PI_SOURCE_TRACE_FIELD,
    assert_no_pi_in_student_prompt,
    build_answer_teacher_prompt,
    build_pi_teacher_prompt,
    build_student_prompt,
    normalize_pi_source,
    optional_bool,
    render_chat_messages,
    render_pi_teacher_user_prompt,
    resolve_emit_think,
    strip_trace_answer,
)
from verl.trainer.bsd.scorer import (
    InlineHFTeacherScorer,
    TeacherScoreRequest,
    assert_left_padded_prefix_mask,
    assert_tokenizer_compatible,
)
from verl.utils import tensordict_utils as tu
from verl.utils.attention_utils import default_attn_implementation
from verl.utils.torch_functional import masked_mean
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

try:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
except ModuleNotFoundError as exc:
    RayPPOTrainer = object
    _RAY_TRAINER_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _RAY_TRAINER_IMPORT_ERROR = None


@dataclass(frozen=True)
class PiTeacherBatchBuild:
    """PI teacher scoring batch plus construction diagnostics."""

    batch: DataProto
    metrics: dict[str, float]
    prompts: list[str]


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected extra_info dict, got {type(value)!r}")


def _raw_prompt_to_text(raw_prompt: Any) -> str | None:
    if raw_prompt is None:
        return None
    if isinstance(raw_prompt, str):
        return raw_prompt
    if isinstance(raw_prompt, list):
        parts = []
        for item in raw_prompt:
            if isinstance(item, dict):
                content = item.get("content", "")
                parts.append(content if isinstance(content, str) else str(content))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(raw_prompt)


def _encode_serialized_prompt(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer([prompt], add_special_tokens=False, padding=False, truncation=False)
    return [int(token_id) for token_id in encoded["input_ids"][0]]


def _fit_pi_teacher_prompt(
    tokenizer: Any,
    problem: str,
    pi_trace: str,
    *,
    max_length: int,
    allow_truncation: bool,
    teacher_thinking: bool,
    emit_think: bool | None = None,
    template: str | None = None,
) -> tuple[str, list[int], bool]:
    prompt = build_pi_teacher_prompt(
        tokenizer, problem, pi_trace, enable_thinking=teacher_thinking, emit_think=emit_think, template=template
    )
    input_ids = _encode_serialized_prompt(tokenizer, prompt)
    if len(input_ids) <= max_length:
        return prompt, input_ids, False
    if not allow_truncation:
        raise ValueError(
            f"PI teacher prompt has {len(input_ids)} tokens, exceeding teacher_max_prompt_length={max_length}"
        )

    empty_prompt = build_pi_teacher_prompt(
        tokenizer, problem, "", enable_thinking=teacher_thinking, emit_think=emit_think, template=template
    )
    empty_ids = _encode_serialized_prompt(tokenizer, empty_prompt)
    if len(empty_ids) > max_length:
        raise ValueError(
            "PI teacher prompt structure without PI trace exceeds teacher_max_prompt_length="
            f"{max_length}; refusing to truncate the problem or chat-template structure"
        )

    best_prompt = empty_prompt
    best_ids = empty_ids
    low, high = 0, len(pi_trace)
    while low <= high:
        midpoint = (low + high) // 2
        candidate_prompt = build_pi_teacher_prompt(
            tokenizer,
            problem,
            pi_trace[:midpoint],
            enable_thinking=teacher_thinking,
            emit_think=emit_think,
            template=template,
        )
        candidate_ids = _encode_serialized_prompt(tokenizer, candidate_prompt)
        if len(candidate_ids) <= max_length:
            best_prompt = candidate_prompt
            best_ids = candidate_ids
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best_prompt, best_ids, True


def _fit_answer_teacher_prompt(
    tokenizer: Any,
    problem: str,
    ground_truth: str,
    *,
    max_length: int,
    teacher_thinking: bool,
    emit_think: bool | None = None,
) -> tuple[str, list[int], bool]:
    prompt = build_answer_teacher_prompt(
        tokenizer,
        problem,
        ground_truth,
        enable_thinking=teacher_thinking,
        emit_think=emit_think,
    )
    input_ids = _encode_serialized_prompt(tokenizer, prompt)
    if len(input_ids) > max_length:
        raise ValueError(
            f"Answer teacher prompt has {len(input_ids)} tokens, exceeding "
            f"teacher_max_prompt_length={max_length}; refusing to truncate reward_model.ground_truth"
        )
    return prompt, input_ids, False


def _left_pad_prefixes(
    input_ids_list: list[list[int]],
    *,
    max_length: int,
    pad_token_id: int,
) -> tuple[torch.LongTensor, torch.Tensor]:
    for input_ids in input_ids_list:
        if len(input_ids) > max_length:
            raise ValueError(
                f"Serialized teacher prompt has {len(input_ids)} tokens, exceeding max_length={max_length}; "
                "serialized chat prefixes must never be truncated"
            )

    batch_size = len(input_ids_list)
    input_ids = torch.full((batch_size, max_length), fill_value=pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    for row_idx, ids in enumerate(input_ids_list):
        if not ids:
            raise ValueError("PI teacher prompt tokenized to an empty prefix")
        width = len(ids)
        start = max_length - width
        input_ids[row_idx, start : start + width] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row_idx, start : start + width] = 1
    return input_ids, attention_mask


def _assert_student_prompt_prefixes(
    source_batch: DataProto,
    tokenizer: Any,
    extra_infos: list[dict[str, Any]],
    *,
    student_thinking: bool,
    emit_think: bool | None = None,
) -> None:
    if "prompts" not in source_batch.batch or "attention_mask" not in source_batch.batch:
        raise ValueError("BSD prompt serialization check requires prompts and attention_mask")
    prefix_ids = source_batch.batch["prompts"].detach().cpu().long()
    prefix_mask = source_batch.batch["attention_mask"][:, : prefix_ids.shape[1]].detach().cpu().bool()
    assert_left_padded_prefix_mask(prefix_mask)
    for row_idx, extra in enumerate(extra_infos):
        problem = extra.get("problem")
        expected_prompt = build_student_prompt(
            tokenizer, str(problem), enable_thinking=student_thinking, emit_think=emit_think
        )
        expected_ids = _encode_serialized_prompt(tokenizer, expected_prompt)
        actual_ids = prefix_ids[row_idx][prefix_mask[row_idx]].tolist()
        if actual_ids != expected_ids:
            sample_id = extra.get("id", row_idx)
            raise ValueError(
                "Student prompt token prefix does not match the declared chat-template thinking mode: "
                f"sample={sample_id}, student_thinking={student_thinking}, "
                f"actual_tokens={len(actual_ids)}, expected_tokens={len(expected_ids)}"
            )


def build_pi_teacher_score_batch(
    *,
    source_batch: DataProto,
    tokenizer: Any,
    max_prompt_length: int,
    allow_prompt_truncation: bool = False,
    teacher_thinking: bool = False,
    student_thinking: bool | None = None,
    emit_think: bool | None = None,
    pi_trace_field: str = DEFAULT_PI_TRACE_FIELD,
    pi_source: str = PI_SOURCE_TRACE_FIELD,
    pi_teacher_template: str | None = None,
) -> PiTeacherBatchBuild:
    """Build a temporary batch that scores original responses under PI prefixes.

    ``pi_teacher_template`` swaps the trace-field teacher wording for a context-specific one (BSD-hint, BSD-guide)."""

    pi_source = normalize_pi_source(pi_source)
    for key in ("responses", "response_mask"):
        if key not in source_batch.batch:
            raise ValueError(f"source batch is missing required tensor key {key!r}")
    if "extra_info" not in source_batch.non_tensor_batch:
        raise ValueError("Bridged Self-Distillation/RLSD requires extra_info.problem")

    extra_infos = [_as_dict(item) for item in source_batch.non_tensor_batch["extra_info"]]
    reward_models: list[dict[str, Any]] | None = None
    if pi_source == PI_SOURCE_GROUND_TRUTH:
        if "reward_model" not in source_batch.non_tensor_batch:
            raise ValueError("Answer-PI RLSD requires reward_model.ground_truth")
        reward_model_items = source_batch.non_tensor_batch["reward_model"]
        if len(reward_model_items) != len(extra_infos):
            raise ValueError("reward_model batch size does not match extra_info batch size")
        reward_models = []
        for row_idx, item in enumerate(reward_model_items):
            try:
                reward_models.append(_as_dict(item))
            except TypeError as exc:
                sample_id = str(extra_infos[row_idx].get("id", row_idx))
                raise ValueError(f"Answer-PI RLSD sample {sample_id} has an invalid reward_model mapping") from exc

    raw_prompts = source_batch.non_tensor_batch.get("raw_prompt")
    prompts: list[str] = []
    prefix_id_rows: list[list[int]] = []
    missing_context_ids: list[str] = []
    context_token_counts: list[int] = []
    leak_checked_count = 0
    leak_count = 0
    truncated = 0
    for row_idx, extra in enumerate(extra_infos):
        problem = extra.get("problem")
        sample_id = str(extra.get("id", row_idx))
        if not problem:
            raise ValueError(f"Bridged Self-Distillation sample {sample_id} is missing extra_info.problem")

        if pi_source == PI_SOURCE_TRACE_FIELD:
            privileged_context = extra.get(pi_trace_field)
        else:
            privileged_context = reward_models[row_idx].get("ground_truth") if reward_models is not None else None
        if not isinstance(privileged_context, str) or not privileged_context.strip():
            missing_context_ids.append(sample_id)
            continue

        if raw_prompts is not None:
            leak_checked_count += 1
            try:
                leak_text = privileged_context if pi_source == PI_SOURCE_TRACE_FIELD else None
                assert_no_pi_in_student_prompt(_raw_prompt_to_text(raw_prompts[row_idx]) or "", leak_text)
            except ValueError:
                leak_count += 1
                raise

        if pi_source == PI_SOURCE_TRACE_FIELD:
            prompt, prompt_ids, was_truncated = _fit_pi_teacher_prompt(
                tokenizer,
                str(problem),
                privileged_context,
                max_length=max_prompt_length,
                allow_truncation=allow_prompt_truncation,
                teacher_thinking=teacher_thinking,
                emit_think=emit_think,
                template=pi_teacher_template,
            )
        else:
            prompt, prompt_ids, was_truncated = _fit_answer_teacher_prompt(
                tokenizer,
                str(problem),
                privileged_context,
                max_length=max_prompt_length,
                teacher_thinking=teacher_thinking,
                emit_think=emit_think,
            )
        prompts.append(prompt)
        prefix_id_rows.append(prompt_ids)
        context_token_counts.append(len(_encode_serialized_prompt(tokenizer, privileged_context)))
        truncated += int(was_truncated)
    if missing_context_ids:
        preview = ", ".join(missing_context_ids[:5])
        field_path = (
            f"extra_info.{pi_trace_field}"
            if pi_source == PI_SOURCE_TRACE_FIELD
            else "reward_model.ground_truth"
        )
        raise ValueError(
            f"Bridged Self-Distillation/RLSD requires non-empty {field_path} for every sample; "
            f"missing or invalid ids: {preview}"
        )

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0) or 0
    if student_thinking is not None:
        _assert_student_prompt_prefixes(
            source_batch,
            tokenizer,
            extra_infos,
            student_thinking=student_thinking,
            emit_think=emit_think,
        )
    prefix_ids, prefix_mask = _left_pad_prefixes(
        prefix_id_rows,
        max_length=max_prompt_length,
        pad_token_id=int(pad_token_id),
    )

    responses = source_batch.batch["responses"].detach().cpu().long()
    response_mask = source_batch.batch["response_mask"].detach().cpu().long()
    if prefix_ids.shape[0] != responses.shape[0]:
        raise ValueError("PI prompt batch size does not match response batch size")
    response_mask_bool = response_mask.bool()
    response_input_ids = responses.masked_fill(~response_mask_bool, int(pad_token_id))
    input_ids = torch.cat([prefix_ids, response_input_ids], dim=1)
    attention_mask = torch.cat([prefix_mask, response_mask_bool.long()], dim=1)
    position_ids = attention_mask.long().cumsum(dim=1).sub(1).clamp_min(0)

    score_batch = DataProto.from_dict(
        tensors={
            "prompts": prefix_ids,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
        }
    )
    metrics = {
        "bsd/pi_teacher_prompt_truncated_count": float(truncated),
        "bsd/pi_trace_truncated_count": (
            float(truncated) if pi_source == PI_SOURCE_TRACE_FIELD else 0.0
        ),
        "bsd/privileged_context_truncated_count": float(truncated),
        "bsd/privileged_context_tokens_mean": float(
            sum(context_token_counts) / len(context_token_counts)
        ),
        "bsd/pi_teacher_prompt_tokens_mean": float(prefix_mask.sum(dim=-1).float().mean().item()),
        "bsd/pi_teacher_response_tokens": float(response_mask.sum().item()),
        "bsd/pi_source_trace_field": float(pi_source == PI_SOURCE_TRACE_FIELD),
        "bsd/pi_source_ground_truth": float(pi_source == PI_SOURCE_GROUND_TRUTH),
        "privacy/student_prompt_pi_leak_checked_count": float(leak_checked_count),
        "privacy/student_prompt_pi_leak_count": float(leak_count),
    }
    return PiTeacherBatchBuild(batch=score_batch, metrics=metrics, prompts=prompts)


def build_direct_opd_teacher_score_batch(
    *,
    source_batch: DataProto,
    tokenizer: Any,
    max_prompt_length: int,
    allow_prompt_truncation: bool = False,
    teacher_thinking: bool = False,
) -> PiTeacherBatchBuild:
    """Build a temporary batch that scores original responses under direct teacher prompts."""

    for key in ("prompts", "responses", "attention_mask", "response_mask"):
        if key not in source_batch.batch:
            raise ValueError(f"source batch is missing required tensor key {key!r}")
    if teacher_thinking:
        raise ValueError("direct_opd_teacher_thinking=True is incompatible with exact student prompt reuse")

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0) or 0

    prefix_ids = source_batch.batch["prompts"].detach().cpu().long()
    prefix_width = prefix_ids.shape[1]
    full_attention_mask = source_batch.batch["attention_mask"].detach().cpu().long()
    if full_attention_mask.shape[1] < prefix_width:
        raise ValueError(
            "Direct OPD-PG requires attention_mask to include the prompt segment: "
            f"attention width {full_attention_mask.shape[1]} < prompt width {prefix_width}"
        )
    prefix_mask = full_attention_mask[:, :prefix_width]
    assert_left_padded_prefix_mask(prefix_mask)

    truncated = 0
    if prefix_width > max_prompt_length:
        raise ValueError(
            "Direct OPD-PG requires the exact student prompt prefix and cannot truncate it: "
            f"prompt width {prefix_width} exceeds teacher_max_prompt_length={max_prompt_length}"
        )

    responses = source_batch.batch["responses"].detach().cpu().long()
    response_mask = source_batch.batch["response_mask"].detach().cpu().long()
    response_mask_bool = response_mask.bool()
    response_input_ids = responses.masked_fill(~response_mask_bool, int(pad_token_id))
    input_ids = torch.cat([prefix_ids, response_input_ids], dim=1)
    attention_mask = torch.cat([prefix_mask, response_mask_bool.long()], dim=1)
    position_ids = attention_mask.long().cumsum(dim=1).sub(1).clamp_min(0)

    score_batch = DataProto.from_dict(
        tensors={
            "prompts": prefix_ids,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
        }
    )
    metrics = {
        "opd/teacher_prompt_truncated_count": float(truncated),
        "opd/teacher_prompt_tokens_mean": float(prefix_mask.sum(dim=-1).float().mean().item()),
        "opd/teacher_response_tokens": float(response_mask.sum().item()),
    }
    return PiTeacherBatchBuild(batch=score_batch, metrics=metrics, prompts=[])


BRIDGED_CRITIC_QUESTION = (
    "check the student's attempt. Is everything the student has written so far correct, and is the attempt on a "
    "path that reaches the correct final answer? Answer with exactly one word: Yes or No."
)
_BRIDGED_CRITIC_SLOT = "<<<BRIDGED_CRITIC_ATTEMPT>>>"


@dataclass(frozen=True)
class BridgedCriticConfig:
    """Bridged-critic settings (``bsd.bridged_critic_*``); see ``BSDRayPPOTrainer._apply_bridged_critic``."""

    cuts: tuple[int, ...]
    min_tail: int
    beta: float
    calibration: tuple[float, float, float]
    max_prompt_length: int
    pi_trace_field: str

    @classmethod
    def from_config(cls, bsd_config: dict[str, Any]) -> BridgedCriticConfig:
        cuts = tuple(sorted(int(c) for c in bsd_config.get("bridged_critic_cuts", (1000, 2000, 3000, 4000, 6000, 8000))))
        if not cuts or cuts[0] <= 0 or len(set(cuts)) != len(cuts):
            raise ValueError(f"bridged_critic_cuts must be distinct positive token positions, got {cuts}")
        beta = float(bsd_config.get("bridged_critic_beta", 0.5))
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"bridged_critic_beta must lie in [0, 1], got {beta}")
        missing = [k for k in ("bridged_critic_calib_a", "bridged_critic_calib_w", "bridged_critic_calib_u") if k not in bsd_config]
        if missing:
            raise ValueError(f"The bridged critic needs an explicit calibration; missing {missing}")
        return cls(
            cuts=cuts,
            min_tail=int(bsd_config.get("bridged_critic_min_tail", 64)),
            beta=beta,
            calibration=tuple(float(bsd_config[k]) for k in ("bridged_critic_calib_a", "bridged_critic_calib_w", "bridged_critic_calib_u")),
            max_prompt_length=int(bsd_config.get("bridged_critic_max_prompt_length", 20480)),
            pi_trace_field=str(bsd_config.get("pi_trace_field", DEFAULT_PI_TRACE_FIELD)),
        )


@dataclass(frozen=True)
class BridgedCriticBatchBuild:
    batch: DataProto | None  # None when every probe was skipped
    probes: list[tuple[int, int]]  # (row, cut) of the scored rows, in batch order; padding rows follow them
    metrics: dict[str, float]


def bridged_critic_user_text(problem: str, reference: str) -> str:
    """User turn of the privileged self-critic: a probe that is answered with the single token Yes or No."""
    return (
        f"Problem: {problem}\n\n"
        f"Here is a reference solution to this problem:\n\n<reference_solution>\n{reference}\n</reference_solution>\n\n"
        "Here is a student's attempt at this problem. It may be cut off before the end.\n\n"
        f"<attempt>\n{_BRIDGED_CRITIC_SLOT}\n</attempt>\n\n"
        f"Using the information above, {BRIDGED_CRITIC_QUESTION}"
    )


def plan_bridged_critic_probes(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    rows: list[int],
    cuts: tuple[int, ...],
    min_tail: int,
    close_id: int | None,
) -> list[tuple[int, int]]:
    """(row, cut) for every absolute cut that leaves at least ``min_tail`` tokens before the row's last ``</think>``.

    An absolute cut does not reveal how long the rollout will become (failed rollouts are longer on average).
    """
    probes: list[tuple[int, int]] = []
    for row in rows:
        length = int(response_mask[row].sum())
        ids = responses[row, :length]
        hits = torch.nonzero(ids.eq(close_id)).flatten() if close_id is not None else torch.empty(0, dtype=torch.long)
        close = int(hits.max()) if hits.numel() else length
        probes.extend((int(row), int(cut)) for cut in cuts if cut <= close - min_tail)
    return probes


def build_bridged_critic_score_batch(
    *,
    source_batch: DataProto,
    tokenizer: Any,
    probes: list[tuple[int, int]],
    pi_trace_field: str,
    max_prompt_length: int,
    pad_multiple: int = 1,
) -> BridgedCriticBatchBuild:
    """One row per probe: the critic prompt around the row's first ``cut`` response ids, then the response "Yes".

    The response prefix enters as its original token ids. The privileged trace appears only in this critic prompt;
    the training prompt is checked to be free of it, as for the PI teacher.
    """
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    if len(yes_ids) != 1:
        raise ValueError(f"The bridged critic reads log P(Yes) of one token; the tokenizer splits 'Yes' into {yes_ids}")
    if "extra_info" not in source_batch.non_tensor_batch:
        raise ValueError("The bridged critic requires extra_info.problem and the privileged trace field")
    extra_infos = [_as_dict(item) for item in source_batch.non_tensor_batch["extra_info"]]
    raw_prompts = source_batch.non_tensor_batch.get("raw_prompt")
    responses = source_batch.batch["responses"].detach().cpu().long()
    frames: dict[int, tuple[list[int], list[int]]] = {}
    rows: list[list[int]] = []
    kept: list[tuple[int, int]] = []
    skipped = 0
    for row, cut in probes:
        if row not in frames:
            extra = extra_infos[row]
            sample_id = str(extra.get("id", row))
            problem, reference = extra.get("problem"), extra.get(pi_trace_field)
            if not problem:
                raise ValueError(f"Bridged-critic sample {sample_id} is missing extra_info.problem")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"The bridged critic requires non-empty extra_info.{pi_trace_field}; missing for sample {sample_id}")
            if raw_prompts is not None:
                assert_no_pi_in_student_prompt(_raw_prompt_to_text(raw_prompts[row]) or "", reference)
            rendered = render_chat_messages(
                tokenizer, [{"role": "user", "content": bridged_critic_user_text(str(problem), reference)}], enable_thinking=False
            )
            head, slot, tail = rendered.partition(_BRIDGED_CRITIC_SLOT)
            if not slot or _BRIDGED_CRITIC_SLOT in tail:
                raise ValueError("The rendered critic prompt must contain the attempt slot exactly once")
            frames[row] = (_encode_serialized_prompt(tokenizer, head), _encode_serialized_prompt(tokenizer, tail))
        head_ids, tail_ids = frames[row]
        ids = head_ids + responses[row, :cut].tolist() + tail_ids
        if len(ids) > max_prompt_length:
            skipped += 1
            continue
        rows.append(ids)
        kept.append((int(row), int(cut)))
    metrics = {
        "bridged_critic/probes_skipped_long": float(skipped),
        "bridged_critic/prompt_tokens_mean": float(np.mean([len(r) for r in rows])) if rows else 0.0,
        "bridged_critic/prompt_tokens_max": float(max((len(r) for r in rows), default=0)),
    }
    if not rows:
        return BridgedCriticBatchBuild(batch=None, probes=[], metrics=metrics)
    padded = rows + [rows[-1]] * ((-len(rows)) % max(int(pad_multiple), 1))  # fill the data-parallel split
    metrics["bridged_critic/padding_rows"] = float(len(padded) - len(rows))
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0) or 0
    prefix_ids, prefix_mask = _left_pad_prefixes(padded, max_length=max(len(r) for r in padded), pad_token_id=int(pad_token_id))
    response_ids = torch.full((len(padded), 1), int(yes_ids[0]), dtype=torch.long)
    response_mask = torch.ones((len(padded), 1), dtype=torch.long)
    attention_mask = torch.cat([prefix_mask, response_mask], dim=1)
    batch = DataProto.from_dict(
        tensors={
            "prompts": prefix_ids,
            "responses": response_ids,
            "input_ids": torch.cat([prefix_ids, response_ids], dim=1),
            "attention_mask": attention_mask,
            "position_ids": attention_mask.long().cumsum(dim=1).sub(1).clamp_min(0),
            "response_mask": response_mask,
        }
    )
    return BridgedCriticBatchBuild(batch=batch, probes=kept, metrics=metrics)


def bridged_critic_segment_weights(bounds: list[int], values: list[float], beta: float) -> list[float] | None:
    """Weights of the segments [bounds[j], bounds[j+1]) of one failed row whose value passes ``values``.

    Blame of a segment = its value drop, max(0, values[j] - values[j+1]); weight = (1 - beta) + beta * blame / mean
    blame, the mean weighted by segment length, so the weights average to one over the row. ``None`` when no segment
    loses value (the row keeps weight one).
    """
    seg = np.diff(bounds).astype(float)
    blame = np.maximum(0.0, -np.diff(np.asarray(values, dtype=float)))
    mean_blame = float((blame * seg).sum() / seg.sum())
    if mean_blame <= 0:
        return None
    return [(1.0 - beta) + beta * float(x) / mean_blame for x in blame]


def compute_bridged_critic_weights(
    *,
    lengths: list[int],
    uids: Any,
    completed: list[bool],
    passed: list[bool],
    scores: dict[tuple[int, int], float],
    calibration: tuple[float, float, float],
    beta: float,
    width: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-token weights of the bridged critic (one where nothing changes) and diagnostics.

    Within a group whose completed rows contain both outcomes, the critic score s = log P(Yes) of the probed rows at a
    cut is standardised over the group, z = (s - mean) / (std + 0.1), and calibrated to a value
    V = sigmoid(a + w * z + u * logit(b)), with b the group's success rate over its completed rows. Along a failed
    completed row the value starts at b, passes V at the row's cuts and ends at the outcome 0. The blame of a segment
    is its value drop, max(0, V_start - V_end), and its tokens get the weight (1 - beta) + beta * blame / mean blame,
    the mean taken over the row's tokens. The weights average to one over the row: the reweighted advantage keeps the
    verifier's sign and the row's total and only moves blame between segments. Correct rows, truncated rows and rows
    of single-outcome groups keep weight one.
    """
    a, w, u = calibration
    weights = torch.ones((len(lengths), width), dtype=torch.float32)
    groups: dict[Any, list[int]] = defaultdict(list)
    for row, done in enumerate(completed):
        if done:
            groups[uids[row]].append(row)
    values: dict[tuple[int, int], float] = {}
    aucs: list[float] = []
    mixed_groups = 0
    for rows in groups.values():
        n_pass = sum(bool(passed[r]) for r in rows)
        if not 0 < n_pass < len(rows):
            continue
        mixed_groups += 1
        b = min(max(n_pass / len(rows), 1e-3), 1 - 1e-3)
        logit_b = math.log(b / (1 - b))
        for cut in sorted({c for (r, c) in scores if r in rows}):
            at = [r for r in rows if (r, cut) in scores]
            if len(at) < 2:
                continue
            s = np.array([scores[(r, cut)] for r in at])
            z = (s - s.mean()) / (s.std() + 0.1)
            for r, zz in zip(at, z):
                values[(r, cut)] = 1.0 / (1.0 + math.exp(-(a + w * float(zz) + u * logit_b)))
            good = [scores[(r, cut)] for r in at if passed[r]]
            bad = [scores[(r, cut)] for r in at if not passed[r]]
            if good and bad:
                aucs.append(float(np.mean([(g > f) + 0.5 * (g == f) for g in good for f in bad])))
        for row in rows:
            if passed[row]:
                continue
            cuts = sorted(c for (r, c) in values if r == row)
            if not cuts:
                continue
            bounds = [0] + cuts + [int(lengths[row])]
            seg_weights = bridged_critic_segment_weights(
                bounds, [n_pass / len(rows)] + [values[(row, c)] for c in cuts] + [0.0], beta
            )
            if seg_weights is None:
                continue
            for (lo, hi), seg_weight in zip(zip(bounds[:-1], bounds[1:]), seg_weights):
                weights[row, lo:hi] = seg_weight
    reweighted = [r for r in range(len(lengths)) if (weights[r, : int(lengths[r])] != 1).any()]
    dev = [float((weights[r, : int(lengths[r])] - 1).abs().mean()) for r in reweighted]
    metrics = {
        "bridged_critic/mixed_groups": float(mixed_groups),
        "bridged_critic/rows_reweighted": float(len(reweighted)),
        "bridged_critic/weight_abs_dev_mean": float(np.mean(dev)) if dev else 0.0,
        "bridged_critic/weight_max": float(weights.max()),
        "bridged_critic/online_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "bridged_critic/last_segment_weight_mean": float(
            np.mean([float(weights[r, int(lengths[r]) - 1]) for r in reweighted])
        )
        if reweighted
        else float("nan"),
    }
    return weights, metrics


def _group_diagnostic_metrics(
    scores: torch.Tensor,
    *,
    uids: Any | None,
    rollout_n: int,
) -> tuple[dict[str, float], bool, bool]:
    if uids is not None:
        if len(uids) != scores.shape[0]:
            return {"train/group_metric_uid_mismatch": 1.0}, True, False
        uid_to_scores: dict[str, list[float]] = defaultdict(list)
        for uid, score in zip(uids, scores.detach().cpu().tolist(), strict=True):
            uid_to_scores[str(uid)].append(float(score))
        groups = [torch.tensor(values, dtype=torch.float32) for values in uid_to_scores.values() if len(values) >= 2]
        if not groups:
            return {}, True, True
        group_std = torch.tensor([group.std(unbiased=False).item() for group in groups], dtype=torch.float32)
        all_correct = torch.tensor([(group >= 1.0).all().item() for group in groups], dtype=torch.float32)
        all_wrong = torch.tensor([(group < 1.0).all().item() for group in groups], dtype=torch.float32)
        return {
            "train/group_nonzero_std_ratio": float((group_std > 0).float().mean().item()),
            "train/group_all_correct_ratio": float(all_correct.mean().item()),
            "train/group_all_wrong_ratio": float(all_wrong.mean().item()),
        }, True, True

    if rollout_n <= 0:
        return {}, False, False
    usable = (scores.shape[0] // rollout_n) * rollout_n
    if usable == 0:
        return {}, False, False
    grouped = scores[:usable].view(-1, rollout_n)
    group_std = grouped.std(dim=-1, unbiased=False)
    correct = grouped >= 1.0
    return {
        "train/group_nonzero_std_ratio": float((group_std > 0).float().mean().detach().item()),
        "train/group_all_correct_ratio": float(correct.all(dim=-1).float().mean().detach().item()),
        "train/group_all_wrong_ratio": float((~correct).all(dim=-1).float().mean().detach().item()),
    }, False, True


def _last_complete_boxed_span(text: str) -> tuple[int, int] | None:
    """Return the character span of the last balanced ``\\boxed{...}`` expression."""

    marker = "\\boxed{"
    last_span = None
    search_start = 0
    while True:
        marker_start = text.find(marker, search_start)
        if marker_start < 0:
            return last_span
        brace_start = marker_start + len(marker) - 1
        depth = 0
        for position in range(brace_start, len(text)):
            character = text[position]
            preceding_slashes = 0
            slash_position = position - 1
            while slash_position >= 0 and text[slash_position] == "\\":
                preceding_slashes += 1
                slash_position -= 1
            if preceding_slashes % 2:
                continue
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    last_span = (marker_start, position + 1)
                    break
        search_start = marker_start + 1


def _decode_individual_token_pieces(tokenizer: Any, token_ids: list[int]) -> list[str]:
    """Decode original token IDs individually, caching context-independent token pieces."""

    cache = getattr(tokenizer, "_bsd_single_token_decode_cache", None)
    if cache is None:
        cache = {}
        tokenizer._bsd_single_token_decode_cache = cache
    missing_ids = list(dict.fromkeys(token_id for token_id in token_ids if token_id not in cache))
    if missing_ids:
        pieces = tokenizer.batch_decode(
            [[token_id] for token_id in missing_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        cache.update(zip(missing_ids, pieces, strict=True))
    return [cache[token_id] for token_id in token_ids]


def _boxed_token_mask(responses: torch.Tensor, response_mask: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    """Mask token boundaries overlapping the last complete boxed expression in each row."""

    boxed_mask = torch.zeros(response_mask.shape, dtype=torch.bool)
    responses_cpu = responses.detach().cpu()
    response_mask_cpu = response_mask.detach().cpu().bool()
    for row in range(response_mask.shape[0]):
        valid_positions = response_mask_cpu[row].nonzero(as_tuple=False).flatten().tolist()
        token_ids = [int(responses_cpu[row, position].item()) for position in valid_positions]
        pieces = _decode_individual_token_pieces(tokenizer, token_ids)
        span = _last_complete_boxed_span("".join(pieces))
        if span is None:
            continue
        span_start, span_end = span
        piece_start = 0
        for valid_position, piece in zip(valid_positions, pieces, strict=True):
            piece_end = piece_start + len(piece)
            if piece_end > span_start and piece_start < span_end:
                boxed_mask[row, valid_position] = True
            piece_start = piece_end
    return boxed_mask.to(response_mask.device)


def _build_rlsd_token_mask(
    response_mask: torch.Tensor,
    spec: str,
    *,
    responses: torch.Tensor | None = None,
    tokenizer: Any | None = None,
) -> torch.Tensor:
    """Build a response-token mask for RLSD credit reweighting from original token IDs."""

    normalized = str(spec or "all").strip().lower().replace("_", "-")
    aliases: dict[str, tuple[str, int | None]] = {
        "all": ("all", None),
        "full": ("all", None),
        "all8k": ("all", None),
        "all-8k": ("all", None),
        "4k": ("prefix", 4096),
        "tail4k": ("tail", 4096),
        "prefix30": ("prefix-ratio", 30),
        "prefix-30": ("prefix-ratio", 30),
        "tail30": ("tail-ratio", 30),
        "tail-30": ("tail-ratio", 30),
        "prefix30+boxed": ("prefix-ratio+boxed", 30),
        "prefix-30+boxed": ("prefix-ratio+boxed", 30),
        "2k+ans": ("prefix+answer", 2048),
        "first2k+tail256": ("prefix+answer", 2048),
        "first-2k+tail-256": ("prefix+answer", 2048),
        "4k+ans": ("prefix+answer", 4096),
        "first4k+tail256": ("prefix+answer", 4096),
        "first-4k+tail-256": ("prefix+answer", 4096),
        "6k+ans": ("prefix+answer", 6144),
        "first6k+tail256": ("prefix+answer", 6144),
        "first-6k+tail-256": ("prefix+answer", 6144),
    }
    if normalized not in aliases:
        raise ValueError(
            "actor.policy_loss.rlsd_token_mask must be one of "
            "all, full, prefix30, prefix30+boxed, tail30, 4k, tail4k, "
            "2k+ans, 4k+ans, or 6k+ans; "
            f"got {spec!r}"
        )
    mode, width = aliases[normalized]
    mask = response_mask.bool()
    if mode == "all":
        return mask.long()

    token_index = mask.long().cumsum(dim=-1)
    valid_lengths = mask.long().sum(dim=-1, keepdim=True)
    if mode.endswith("ratio") or mode == "prefix-ratio+boxed":
        width = (valid_lengths * int(width) + 99) // 100
    if mode == "prefix":
        return (mask & (token_index <= int(width))).long()
    if mode == "tail":
        return (mask & (token_index > (valid_lengths - int(width)).clamp_min(0))).long()
    if mode == "prefix-ratio":
        return (mask & (token_index <= width)).long()
    if mode == "tail-ratio":
        return (mask & (token_index > valid_lengths - width)).long()
    if mode == "prefix-ratio+boxed":
        if responses is None or tokenizer is None:
            raise ValueError("prefix30+boxed requires original responses and tokenizer")
        prefix_mask = mask & (token_index <= width)
        return (prefix_mask | _boxed_token_mask(responses, mask, tokenizer)).long()

    first_mask = token_index <= int(width)
    answer_mask = token_index > (valid_lengths - 256).clamp_min(0)
    return (mask & (first_mask | answer_mask)).long()


# ---------------------------------------------------------------------------
# Guided rescue of k = 0 groups.
#
# Every prompt also gets ``guided_samples`` completions sampled from
# pi_theta(. | x, answer-stripped trace) in the same rollout call. For groups whose
# unguided rollouts all failed the verifier (k = 0), guided completions that pass
# the verifier, a relative length floor and a meta-phrase filter replace unguided
# rows of that group. The replaced row keeps the PI-free prompt x, so the guided
# prompt is never a log-prob context for the loss; the completion is appended as
# its original token ids and tagged by GUIDED_ROW_FLAG. Before the update the
# rollout IS weight of tagged rows becomes rho_t = P(y_t|x) / P(y_t|x, trace),
# truncated to [rho_min, rho_max].
# ---------------------------------------------------------------------------

GUIDED_ROW_FLAG = "guided_rescue_row"
DEFAULT_GUIDED_REJECT_REGEX = r"reference solution|given solution|reference comments|solution above|as requested"
_PROMPT_DERIVED_KEYS = ("prompts", "input_ids", "attention_mask", "position_ids")


@dataclass(frozen=True)
class GuidedRescueConfig:
    """Resolved ``bsd.guided_*`` settings."""

    samples: int
    rho_min: float
    rho_max: float
    min_length_ratio: float
    reject_regex: str
    pi_trace_field: str
    pi_teacher_template: str | None
    student_thinking: bool
    emit_think: bool
    prompt_length: int

    @classmethod
    def from_config(cls, bsd_config: dict[str, Any], *, prompt_length: int) -> GuidedRescueConfig | None:
        if not bool(optional_bool(bsd_config.get("guided_rescue_enabled"))):
            return None
        samples = int(bsd_config.get("guided_samples", 2))
        rho_min = float(bsd_config.get("guided_rho_min", 0.2))
        rho_max = float(bsd_config.get("guided_rho_max", 1.0))
        min_length_ratio = float(bsd_config.get("guided_min_length_ratio", 0.08))
        student_thinking = optional_bool(bsd_config.get("student_thinking"))
        if samples <= 0:
            raise ValueError("bsd.guided_samples must be positive")
        if not 0.0 <= rho_min <= rho_max or rho_max <= 0.0:
            raise ValueError("bsd.guided_rho_min/guided_rho_max must satisfy 0 <= min <= max, max > 0")
        if not math.isfinite(min_length_ratio) or min_length_ratio < 0.0:
            raise ValueError("bsd.guided_min_length_ratio must be a finite non-negative number")
        if student_thinking is None:
            raise ValueError("Guided rescue requires an explicit bsd.student_thinking")
        # Absent -> built-in phrase list; explicit empty string -> filter disabled.
        # Validate eagerly so a bad pattern fails at startup, not at the first rescue step.
        raw_regex = bsd_config.get("guided_reject_regex")
        reject_regex = DEFAULT_GUIDED_REJECT_REGEX if raw_regex is None else str(raw_regex)
        if reject_regex:
            try:
                re.compile(reject_regex)
            except re.error as exc:
                raise ValueError(f"bsd.guided_reject_regex is not a valid regex: {exc}") from exc
        return cls(
            samples=samples,
            rho_min=rho_min,
            rho_max=rho_max,
            min_length_ratio=min_length_ratio,
            reject_regex=reject_regex,
            pi_trace_field=str(bsd_config.get("pi_trace_field", DEFAULT_PI_TRACE_FIELD)),
            pi_teacher_template=str(bsd_config.get("pi_teacher_template") or "") or None,
            student_thinking=student_thinking,
            emit_think=resolve_emit_think(student_thinking, optional_bool(bsd_config.get("emit_think"))),
            prompt_length=int(prompt_length),
        )


_THINK_TAG = re.compile(r"</?think>")


def build_guided_gen_batch(
    gen_batch: DataProto,
    tokenizer: Any,
    cfg: GuidedRescueConfig,
) -> tuple[DataProto, dict[str, float]]:
    """Repeat every prompt ``cfg.samples`` times under a trace-conditioned sampling prompt.

    Only ``raw_prompt`` changes; ``uid`` and the other request fields are kept so the
    outputs can be matched back to their groups by position.
    """

    for key in ("extra_info", "raw_prompt", "uid"):
        if key not in gen_batch.non_tensor_batch:
            raise ValueError(f"Guided rescue requires non-tensor key {key!r} in the generation batch")
    extra_infos = [_as_dict(item) for item in gen_batch.non_tensor_batch["extra_info"]]
    raw_prompts = gen_batch.non_tensor_batch["raw_prompt"]
    reward_models = gen_batch.non_tensor_batch.get("reward_model")
    messages: list[list[dict[str, str]]] = []
    prompt_tokens: list[int] = []
    stripped_count = 0
    gt_leak_count = 0
    for row_idx, extra in enumerate(extra_infos):
        sample_id = str(extra.get("id", row_idx))
        problem = extra.get("problem")
        if not problem:
            raise ValueError(f"Guided rescue sample {sample_id} is missing extra_info.problem")
        trace = extra.get(cfg.pi_trace_field)
        if not isinstance(trace, str) or not trace.strip():
            raise ValueError(f"Guided rescue sample {sample_id} is missing extra_info.{cfg.pi_trace_field}")
        # The rollout prompt x must stay PI-free even though a guided sibling is built next to it.
        assert_no_pi_in_student_prompt(_raw_prompt_to_text(raw_prompts[row_idx]) or "", trace)
        stripped = strip_trace_answer(trace)
        if not _THINK_TAG.sub("", stripped).strip():
            raise ValueError(f"Guided rescue sample {sample_id}: nothing is left of the trace after answer stripping")
        stripped_count += int(stripped != trace)
        if reward_models is not None:
            ground_truth = str(_as_dict(reward_models[row_idx]).get("ground_truth", ""))
            gt_leak_count += int(bool(ground_truth.strip()) and ground_truth.strip() in stripped)
        # The agent loop renders raw_prompt with the student chat template; measure that
        # exact serialization because it silently left-truncates anything longer than
        # rollout.prompt_length, which would cut the trace instead of failing.
        rendered = build_pi_teacher_prompt(
            tokenizer, str(problem), stripped, enable_thinking=cfg.student_thinking, emit_think=cfg.emit_think,
            template=cfg.pi_teacher_template,
        )
        n_tokens = len(_encode_serialized_prompt(tokenizer, rendered))
        if n_tokens > cfg.prompt_length:
            raise ValueError(
                f"Guided sampling prompt for sample {sample_id} has {n_tokens} tokens, exceeding "
                f"rollout.prompt_length={cfg.prompt_length}; raise data.max_prompt_length instead of "
                "letting the agent loop truncate the trace"
            )
        content = render_pi_teacher_user_prompt(
            str(problem), stripped, emit_think=cfg.emit_think, template=cfg.pi_teacher_template
        )
        messages.append([{"role": "user", "content": content}])
        prompt_tokens.append(n_tokens)

    guided = gen_batch.repeat(repeat_times=cfg.samples, interleave=True)
    repeated = [message for message in messages for _ in range(cfg.samples)]
    guided.non_tensor_batch["raw_prompt"] = np.fromiter(repeated, dtype=object, count=len(repeated))
    metrics = {
        "guided_rescue/guided_requests": float(len(guided)),
        "guided_rescue/guided_prompt_tokens_mean": float(np.mean(prompt_tokens)),
        "guided_rescue/guided_prompt_tokens_max": float(max(prompt_tokens)),
        "guided_rescue/trace_answer_stripped_frac": stripped_count / len(extra_infos),
        # Ground-truth string still literally present in the reasoning that the guided
        # prompt shows; logged so that the rescue metric can be stratified by it.
        "guided_rescue/gt_in_guided_prompt_frac": (
            gt_leak_count / len(extra_infos) if reward_models is not None else float("nan")
        ),
    }
    return guided, metrics


def _verifier_passed(proto: DataProto) -> torch.Tensor:
    """Boolean per row: the completion was verified correct.

    The reward manager reports ``accuracy`` separately from the ``score`` (which can
    also pay an optional format bonus for a boxed-but-wrong answer; the bonus is 0 by
    default), so that field is used when present. Otherwise a verified completion scores
    exactly 1.0 and the format bonus is strictly smaller, so the sequence score is
    compared against 1.0.
    """

    accuracy = proto.non_tensor_batch.get("accuracy")
    if accuracy is not None:
        return torch.from_numpy(np.asarray(accuracy, dtype=np.float64) > 0.0)
    return (proto.batch["rm_scores"].sum(dim=-1) >= 1.0).cpu()


def _think_close_token_id(tokenizer: Any) -> int | None:
    """Vocabulary id of ``</think>`` when it is a single token (Qwen3), else ``None``.

    A tokenizer that spells the tag as plain text (``</``, ``think``, ``>`` -- where the
    ``>`` can merge with a following newline) has no single id; the rescue filter then
    falls back to a decoded-text check in :func:`_closes_think`.
    """

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    close_id = convert("</think>") if convert is not None else None
    if close_id is None or close_id == getattr(tokenizer, "unk_token_id", None):
        return None
    return int(close_id)


def _closes_think(tokenizer: Any, response_ids: torch.Tensor, close_id: int | None) -> bool:
    """Whether a guided completion contains the closing ``</think>`` tag."""

    if close_id is not None:
        return bool(response_ids.eq(close_id).any())
    return "</think>" in tokenizer.decode(response_ids.tolist(), skip_special_tokens=False)


def rescue_k0_groups(
    batch: DataProto,
    guided_batch: DataProto,
    guided_output: DataProto,
    *,
    tokenizer: Any,
    cfg: GuidedRescueConfig,
) -> tuple[DataProto, dict[str, float]]:
    """Replace unguided rows of verifier-failed groups with surviving guided completions.

    ``guided_output`` rows follow ``guided_batch`` order: the group id comes from
    ``guided_batch.non_tensor_batch['uid']`` and the completion tensors from
    ``guided_output``. A rebuilt row keeps the unguided row's prompt x and metadata,
    takes the guided response, response_mask, rm_scores and rollout_log_probs as they
    are, and re-derives input_ids/attention_mask/position_ids under x. Group size and
    batch size do not change.
    """

    for key in ("responses", "response_mask", "rm_scores", "rollout_log_probs", *_PROMPT_DERIVED_KEYS):
        if key not in batch.batch or key not in guided_output.batch:
            raise ValueError(f"Guided rescue requires tensor key {key!r} in both rollout outputs")
    if "uid" not in batch.non_tensor_batch or "uid" not in guided_batch.non_tensor_batch:
        raise ValueError("Guided rescue requires uid on the rollout batch and the guided requests")
    if len(guided_output) != len(guided_batch):
        raise ValueError(
            f"guided rollout returned {len(guided_output)} rows for {len(guided_batch)} requests"
        )
    if batch.batch["position_ids"].dim() != 2:
        raise ValueError("Guided rescue supports 1-D position ids only")

    uids = np.asarray(batch.non_tensor_batch["uid"])
    guided_uids = np.asarray(guided_batch.non_tensor_batch["uid"])
    verified = _verifier_passed(batch)
    lengths = batch.batch["response_mask"].sum(dim=-1).cpu()
    guided_verified = _verifier_passed(guided_output)
    guided_lengths = guided_output.batch["response_mask"].sum(dim=-1).cpu()
    # The rebuilt attention mask is prompt_mask ++ response_mask. That only reproduces
    # the guided rollout's own mask for single-turn completions; a multi-turn row would
    # carry tool/observation segments the loss must not see, so fail fast.
    guided_prompt_length = guided_output.batch["prompts"].shape[1]
    guided_response_attention = guided_output.batch["attention_mask"][:, guided_prompt_length:]
    if not torch.equal(guided_response_attention.bool(), guided_output.batch["response_mask"].bool()):
        raise ValueError(
            "Guided rescue expects single-turn guided rollouts: attention_mask over the "
            "response segment differs from response_mask"
        )

    group_uids = list(dict.fromkeys(uids.tolist()))
    k0_uids = [uid for uid in group_uids if not bool(verified[torch.from_numpy(uids == uid)].any())]
    close_id = _think_close_token_id(tokenizer)
    pattern = re.compile(cfg.reject_regex, re.IGNORECASE) if cfg.reject_regex else None

    guided_seen_k0 = 0
    guided_verified_k0 = 0
    think_rejects = 0
    length_rejects = 0
    regex_rejects = 0
    rescued_groups = 0
    min_lengths: list[float] = []
    targets: list[int] = []
    sources: list[int] = []
    for uid in k0_uids:
        group_rows = np.flatnonzero(uids == uid)
        # Dynamic length floor, relative to this group's own unguided rollouts
        # (truncated rows included): a guided completion must be at least
        # `min_length_ratio` of the group's mean response length. The scale follows
        # the student's current length instead of a fixed token count, and the floor
        # is a degenerate-output guard, not a quality filter: verified guided
        # completions are not shorter than failed ones.
        min_length = cfg.min_length_ratio * float(lengths[group_rows].float().mean().item())
        min_lengths.append(min_length)
        candidates = np.flatnonzero(guided_uids == uid)
        guided_seen_k0 += len(candidates)
        survivors: list[int] = []
        for g in candidates.tolist():
            if not bool(guided_verified[g]):
                continue
            guided_verified_k0 += 1
            response_mask = guided_output.batch["response_mask"][g].bool()
            # A rescue must close its reasoning block: an answer-only completion never does.
            if not _closes_think(tokenizer, guided_output.batch["responses"][g][response_mask], close_id):
                think_rejects += 1
                continue
            if float(guided_lengths[g]) < min_length:
                length_rejects += 1
                continue
            if pattern is not None:
                # Decoded for filtering only; the token ids are what enter the batch.
                response_ids = guided_output.batch["responses"][g][response_mask]
                if pattern.search(tokenizer.decode(response_ids.tolist(), skip_special_tokens=True)):
                    regex_rejects += 1
                    continue
            survivors.append(g)
        if not survivors:
            continue
        survivors = survivors[: len(group_rows)]
        rescued_groups += 1
        targets.extend(int(t) for t in group_rows[len(group_rows) - len(survivors) :])
        sources.extend(survivors)

    flag = np.zeros(len(batch), dtype=bool)
    if targets:
        t_idx = torch.tensor(targets, dtype=torch.long)
        g_idx = torch.tensor(sources, dtype=torch.long)
        for key, value in guided_output.batch.items():
            if key in _PROMPT_DERIVED_KEYS:
                continue
            if key not in batch.batch:
                raise ValueError(f"guided rollout output has tensor key {key!r} that the rollout batch lacks")
            batch.batch[key][t_idx] = value[g_idx].to(batch.batch[key])
        prompt_length = batch.batch["prompts"].shape[1]
        prompts = batch.batch["prompts"][t_idx]
        prompt_mask = batch.batch["attention_mask"][t_idx, :prompt_length]
        response_mask = batch.batch["response_mask"][t_idx].to(prompt_mask)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        batch.batch["input_ids"][t_idx] = torch.cat([prompts, batch.batch["responses"][t_idx]], dim=1)
        batch.batch["attention_mask"][t_idx] = attention_mask
        batch.batch["position_ids"][t_idx] = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0).to(
            batch.batch["position_ids"]
        )
        # Completion-level fields (reward extras, turn counts) follow the guided sample;
        # raw_prompt stays x so downstream leak checks see the training prompt.
        for key, value in guided_output.non_tensor_batch.items():
            if key == "raw_prompt" or key not in batch.non_tensor_batch:
                continue
            batch.non_tensor_batch[key][targets] = value[sources]
        flag[targets] = True
    batch.non_tensor_batch[GUIDED_ROW_FLAG] = flag

    num_groups = max(len(group_uids), 1)
    guided_verified_all = float(guided_verified.float().mean().item()) if len(guided_verified) else 0.0
    metrics = {
        "guided_rescue/k0_groups": float(len(k0_uids)),
        "guided_rescue/k0_group_frac": len(k0_uids) / num_groups,
        "guided_rescue/guided_verified_frac_all": guided_verified_all,
        "guided_rescue/guided_verified_frac_k0": guided_verified_k0 / max(guided_seen_k0, 1),
        "guided_rescue/guided_think_reject_count": float(think_rejects),
        "guided_rescue/guided_length_reject_count": float(length_rejects),
        "guided_rescue/guided_regex_reject_count": float(regex_rejects),
        "guided_rescue/guided_min_length": float(np.mean(min_lengths)) if min_lengths else 0.0,
        "guided_rescue/rescued_groups": float(rescued_groups),
        "guided_rescue/rescued_group_frac_of_k0": rescued_groups / max(len(k0_uids), 1),
        "guided_rescue/replaced_rows": float(len(targets)),
        "guided_rescue/replaced_row_frac": len(targets) / max(len(batch), 1),
        "guided_rescue/guided_response_length_mean": (
            float(batch.batch["response_mask"][torch.tensor(targets)].sum(dim=-1).float().mean().item())
            if targets
            else 0.0
        ),
    }
    return batch, metrics


def apply_guided_is_weights(batch: DataProto, cfg: GuidedRescueConfig, metrics: dict[str, Any]) -> None:
    """Replace the rollout IS weight of guided rows with the truncated ratio rho_t.

    ``old_log_probs`` are the actor's log-probs under x and ``rollout_log_probs`` the
    sampler's log-probs under (x, trace), so exp(old - rollout) is exactly
    P(y_t|x) / P(y_t|x, trace). The weight is detached and truncated; unguided rows
    keep the upstream truncated IS weight.
    """

    flag = batch.non_tensor_batch.get(GUIDED_ROW_FLAG)
    if flag is None:
        raise ValueError("Guided rescue is enabled but the batch carries no guided-row flag")
    if "rollout_is_weights" not in batch.batch:
        raise ValueError(
            "Guided rescue requires algorithm.rollout_correction.rollout_is so rho_t can replace the IS weight"
        )
    mask_all = batch.batch["response_mask"]
    guided = torch.from_numpy(np.asarray(flag, dtype=bool)).to(mask_all.device)
    metrics["guided_rescue/guided_rows"] = float(guided.sum().item())
    metrics["guided_rescue/guided_row_frac"] = float(guided.float().mean().item())
    if not bool(guided.any()):
        for key in ("rho_floor_frac", "rho_cap_frac", "rho_mean", "log_rho_raw_mean", "rho_ess_ratio_mean"):
            metrics[f"guided_rescue/{key}"] = float("nan")
        return

    mask = mask_all[guided].bool()
    log_ratio = (batch.batch["old_log_probs"][guided] - batch.batch["rollout_log_probs"][guided]).float().detach()
    raw = torch.exp(log_ratio)
    rho = raw.clamp(min=cfg.rho_min, max=cfg.rho_max) * mask
    weights = batch.batch["rollout_is_weights"]
    weights[guided] = rho.to(weights.dtype)

    num_tokens = mask.sum().clamp_min(1).float()
    seq_tokens = mask.sum(dim=-1).clamp_min(1).float()
    ess_ratio = rho.sum(dim=-1).pow(2) / (rho.pow(2).sum(dim=-1).clamp_min(1e-12) * seq_tokens)
    metrics["guided_rescue/rho_floor_frac"] = float((((raw < cfg.rho_min) & mask).sum() / num_tokens).item())
    metrics["guided_rescue/rho_cap_frac"] = float((((raw > cfg.rho_max) & mask).sum() / num_tokens).item())
    metrics["guided_rescue/rho_mean"] = masked_mean(rho, mask).item()
    metrics["guided_rescue/log_rho_raw_mean"] = masked_mean(log_ratio, mask).item()
    metrics["guided_rescue/rho_ess_ratio_mean"] = float(ess_ratio.mean().item())


def check_training_file_yields_planned_updates(*, num_batches: int, total_epochs: int, planned_updates: int) -> None:
    """Refuse a run whose training file cannot yield the planned number of updates.

    The training loop also ends when the passes over the training file are used up, before
    ``trainer.total_training_steps`` is reached and without an error. The last update of such
    a run is saved only if it is a multiple of the checkpoint interval, and the decay of the
    weight coefficient is cut short, so the shortfall is an error here, before any rollout.
    ``num_batches`` counts the complete batches that the training file holds after the
    problems with an over-long prompt are dropped.
    """

    available_updates = int(num_batches) * int(total_epochs)
    if available_updates < int(planned_updates):
        raise ValueError(
            f"The training file yields {available_updates} updates ({int(num_batches)} batches of "
            f"data.train_batch_size problems times trainer.total_epochs={int(total_epochs)}) after the "
            f"problems with an over-long prompt are dropped, fewer than "
            f"trainer.total_training_steps={int(planned_updates)}. Provide more problems, raise "
            "trainer.total_epochs (TOTAL_EPOCHS in the launchers) or lower trainer.total_training_steps."
        )


class BSDRayPPOTrainer(RayPPOTrainer):
    """Legacy PPO trainer that injects PI teacher logprobs for RLSD updates."""

    _TEACHER_SNAPSHOT_ADAPTER = "bsd_teacher"
    _TRAINER_STATE_FILENAME = "bsd_trainer_state.json"

    def __init__(self, *args, **kwargs) -> None:
        if _RAY_TRAINER_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "BSDRayPPOTrainer requires the full verl PPO trainer dependencies"
            ) from _RAY_TRAINER_IMPORT_ERROR
        super().__init__(*args, **kwargs)
        # self.total_training_steps is trainer.total_training_steps when it is set, else the
        # number of updates that the passes over the training file yield.
        check_training_file_yields_planned_updates(
            num_batches=len(self.train_dataloader),
            total_epochs=int(self.config.trainer.total_epochs),
            planned_updates=int(self.total_training_steps),
        )
        self._bsd_teacher_last_sync_step: int | None = None
        self._rollout_actor_consistency_checked = False

    def _trainer_state_path(self, *, loading: bool = False) -> str:
        if loading and self.config.trainer.resume_mode == "resume_path":
            global_step_folder = self.config.trainer.resume_from_path
        else:
            global_step_folder = os.path.join(
                self.config.trainer.default_local_dir,
                f"global_step_{self.global_steps}",
            )
        return os.path.join(global_step_folder, self._TRAINER_STATE_FILENAME)

    def _save_checkpoint(self):
        super()._save_checkpoint()
        state_path = self._trainer_state_path()
        state = {
            "version": 1,
            "global_step": int(self.global_steps),
            "teacher_update_mode": str(self.config.bsd.get("teacher_update_mode", "base_no_adapter")),
            "teacher_last_sync_step": self._bsd_teacher_last_sync_step,
        }
        tmp_path = f"{state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, state_path)

    def _load_checkpoint(self):
        result = super()._load_checkpoint()
        if int(getattr(self, "global_steps", 0)) <= 0:
            return result

        teacher_update_mode = str(self.config.bsd.get("teacher_update_mode", "base_no_adapter"))
        if teacher_update_mode != "periodic_snapshot":
            return result

        state_path = self._trainer_state_path(loading=True)
        if os.path.isfile(state_path):
            with open(state_path, encoding="utf-8") as handle:
                state = json.load(handle)
            if int(state.get("version", -1)) != 1:
                raise ValueError(f"Unsupported BSD trainer state version in {state_path}: {state.get('version')}")
            if int(state.get("global_step", -1)) != int(self.global_steps):
                raise ValueError(
                    f"BSD trainer state step mismatch in {state_path}: "
                    f"saved={state.get('global_step')}, resumed={self.global_steps}"
                )
            if state.get("teacher_update_mode") != teacher_update_mode:
                raise ValueError(
                    f"BSD teacher update mode mismatch in {state_path}: "
                    f"saved={state.get('teacher_update_mode')}, resumed={teacher_update_mode}"
                )
            last_sync_step = state.get("teacher_last_sync_step")
            if not isinstance(last_sync_step, int) or not 0 < last_sync_step <= int(self.global_steps):
                raise ValueError(f"Invalid teacher_last_sync_step in {state_path}: {last_sync_step!r}")
            self._bsd_teacher_last_sync_step = last_sync_step
            return result

        sync_interval = int(self.config.bsd.get("teacher_sync_interval", 10))
        if sync_interval <= 0:
            raise ValueError("bsd.teacher_sync_interval must be positive for periodic_snapshot")
        self._bsd_teacher_last_sync_step = 1 + ((int(self.global_steps) - 1) // sync_interval) * sync_interval
        print(
            f"Warning: {state_path} is missing; inferred periodic teacher sync step "
            f"{self._bsd_teacher_last_sync_step}. Exact recovery cannot be guaranteed for a checkpoint "
            "that was itself created after an earlier resume."
        )
        return result

    def _check_rollout_actor_consistency(
        self,
        metrics: dict[str, Any],
        bsd_config: dict[str, Any],
    ) -> None:
        """Fail before the first update when rollout and actor weights disagree."""

        threshold = bsd_config.get("rollout_actor_consistency_threshold")
        if threshold is None or getattr(self, "_rollout_actor_consistency_checked", False):
            return
        threshold = float(threshold)
        valid = metrics.get("training/rollout_probs_diff_valid")
        correlation = metrics.get("training/rollout_actor_probs_pearson_corr")
        if valid != 1 or correlation is None or not math.isfinite(float(correlation)):
            raise RuntimeError(
                "BSD rollout/actor consistency check requires valid finite first-step log-prob metrics."
            )
        if float(correlation) < threshold:
            raise RuntimeError(
                "BSD rollout/actor consistency check failed before actor update: "
                f"correlation={float(correlation):.6f}, required>={threshold:.6f}."
            )
        metrics["bsd/rollout_actor_consistency_passed"] = 1.0
        metrics["bsd/rollout_actor_consistency_threshold"] = threshold
        self._rollout_actor_consistency_checked = True

    def _guided_rescue_config(self) -> GuidedRescueConfig | None:
        bsd_config = self.config.get("bsd", {})
        if OmegaConf.is_config(bsd_config):
            bsd_config = OmegaConf.to_container(bsd_config, resolve=True)
        if not bool(optional_bool(bsd_config.get("guided_rescue_enabled"))):
            return None
        rollout_config = self.config.actor_rollout_ref.rollout
        cfg = GuidedRescueConfig.from_config(bsd_config, prompt_length=int(rollout_config.prompt_length))
        if cfg is not None and str(self.config.algorithm.adv_estimator) != "grpo":
            raise ValueError("Guided rescue rebuilds uid groups and therefore requires adv_estimator=grpo")
        # Survivors replace the last ``samples`` rows of a k=0 group; every group must
        # keep at least one unguided row so the unguided-only metrics and gate exist.
        if cfg is not None and cfg.samples >= int(rollout_config.n):
            raise ValueError(
                f"bsd.guided_samples={cfg.samples} must be smaller than rollout.n={int(rollout_config.n)}"
            )
        # The rollout tracer records every request prompt, which would include the
        # trace-conditioned guided prompts; keep it off rather than leak PI to a tracker.
        trace_backend = (rollout_config.get("trace") or {}).get("backend")
        if cfg is not None and trace_backend:
            raise ValueError(
                f"Guided rescue does not support rollout.trace.backend={trace_backend!r}: "
                "the tracer would log the trace-conditioned guided prompts"
            )
        return cfg

    def _build_extra_gen_batch(self, gen_batch: DataProto) -> DataProto | None:
        cfg = self._guided_rescue_config()
        if cfg is None:
            return None
        guided, self._guided_gen_metrics = build_guided_gen_batch(gen_batch, self.tokenizer, cfg)
        return guided

    def _after_rollout(
        self,
        batch: DataProto,
        extra_gen_batch: DataProto | None,
        extra_gen_output: DataProto | None,
        metrics: dict[str, Any],
    ) -> DataProto:
        cfg = self._guided_rescue_config()
        if cfg is None:
            return batch
        if extra_gen_batch is None or extra_gen_output is None:
            raise ValueError("Guided rescue is enabled but the rollout call carried no guided requests")
        metrics.update(getattr(self, "_guided_gen_metrics", {}))
        batch, rescue_metrics = rescue_k0_groups(
            batch, extra_gen_batch, extra_gen_output, tokenizer=self.tokenizer, cfg=cfg
        )
        metrics.update(rescue_metrics)
        return batch

    def _compute_actor_log_prob(
        self,
        batch: DataProto,
        *,
        no_lora_adapter: bool = False,
        lora_adapter_name: str | None = None,
    ) -> DataProto:
        if no_lora_adapter and lora_adapter_name is not None:
            raise ValueError("no_lora_adapter and lora_adapter_name are mutually exclusive")
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        rollout_config = self.config.actor_rollout_ref.rollout
        metadata = {
            "calculate_entropy": False,
            "compute_loss": False,
            "temperature": rollout_config.temperature,
        }
        if no_lora_adapter:
            metadata["no_lora_adapter"] = True
        if lora_adapter_name is not None:
            metadata["lora_adapter_name"] = lora_adapter_name
        tu.assign_non_tensor(batch_td, **metadata)
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        log_probs = tu.get(output, "log_probs")
        log_probs = no_padding_2_padding(log_probs, batch_td)
        return DataProto.from_tensordict(tu.get_tensordict({"ref_log_prob": log_probs.float()}))

    def _compute_effective_rlsd_lambda(self, policy_loss: Any) -> float:
        target_lambda = float(policy_loss.get("rlsd_lambda", 0.5))
        warmup_steps = int(policy_loss.get("rlsd_lambda_warmup_steps", 0))
        decay_steps = int(policy_loss.get("rlsd_lambda_decay_steps", 150))
        if warmup_steps < 0:
            raise ValueError("actor.policy_loss.rlsd_lambda_warmup_steps must be non-negative")
        if decay_steps < 0:
            raise ValueError("actor.policy_loss.rlsd_lambda_decay_steps must be non-negative")
        step = int(getattr(self, "global_steps", 0))
        if warmup_steps > 0 and step < warmup_steps:
            return target_lambda * (step / warmup_steps)
        if decay_steps > 0 and step >= warmup_steps:
            decay_progress = (step - warmup_steps) / decay_steps
            return target_lambda * max(1.0 - decay_progress, 0.0)
        return target_lambda

    def _maybe_sync_teacher_snapshot(
        self,
        *,
        bsd_config: dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        sync_interval = int(bsd_config.get("teacher_sync_interval", 10))
        if sync_interval <= 0:
            raise ValueError("bsd.teacher_sync_interval must be positive for periodic_snapshot")
        step = int(getattr(self, "global_steps", 0))
        last_sync = getattr(self, "_bsd_teacher_last_sync_step", None)
        should_sync = last_sync is None or (step - int(last_sync)) >= sync_interval
        if should_sync:
            self.actor_rollout_wg.sync_bsd_teacher_snapshot(
                source_adapter="default",
                target_adapter=self._TEACHER_SNAPSHOT_ADAPTER,
            )
            self._bsd_teacher_last_sync_step = step
        metrics["bsd/teacher_snapshot_synced"] = 1.0 if should_sync else 0.0
        metrics["bsd/teacher_snapshot_step"] = float(getattr(self, "_bsd_teacher_last_sync_step", -1))
        metrics["bsd/teacher_sync_interval"] = float(sync_interval)

    def _compute_pi_teacher_log_prob(
        self,
        score_batch: DataProto,
        *,
        teacher_update_mode: str,
        bsd_config: dict[str, Any],
        metrics: dict[str, Any],
    ) -> DataProto:
        if teacher_update_mode == "base_no_adapter":
            if not self.ref_in_actor:
                raise ValueError(
                    "teacher_update_mode=base_no_adapter requires a LoRA actor so the base model is in actor"
                )
            return self._compute_ref_log_prob(score_batch)
        if teacher_update_mode == "current_no_grad":
            return self._compute_actor_log_prob(score_batch)
        if teacher_update_mode == "periodic_snapshot":
            if not self.ref_in_actor:
                raise ValueError("teacher_update_mode=periodic_snapshot requires a LoRA actor")
            if not bool(self.config.actor_rollout_ref.model.get("bsd_teacher_adapter", False)):
                raise ValueError(
                    "teacher_update_mode=periodic_snapshot requires actor_rollout_ref.model.bsd_teacher_adapter=True"
                )
            self._maybe_sync_teacher_snapshot(bsd_config=bsd_config, metrics=metrics)
            return self._compute_actor_log_prob(
                score_batch,
                lora_adapter_name=self._TEACHER_SNAPSHOT_ADAPTER,
            )
        raise ValueError(
            "bsd.teacher_update_mode must be one of "
            "base_no_adapter, current_no_grad, periodic_snapshot; "
            f"got {teacher_update_mode!r}"
        )

    def _think_tag_token_ids(self) -> tuple[int | None, int | None]:
        """Vocabulary ids of the reasoning-block tags, or ``None`` where unavailable.

        Resolution is deliberately tolerant: this feeds an observability metric, and a
        vocabulary without the tags -- or a student tokenizer that does not expose the
        lookup, as on the cross-vocabulary paths -- must skip the metric rather than
        abort a training step.
        """
        cached = getattr(self, "_bsd_think_tag_ids", None)
        if cached is not None:
            return cached
        tokenizer = getattr(self, "tokenizer", None)
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if convert is None:
            cached = (None, None)
        else:
            unk = getattr(tokenizer, "unk_token_id", None)
            resolved = []
            for tag in ("<think>", "</think>"):
                token_id = convert(tag)
                resolved.append(None if token_id is None or token_id == unk else int(token_id))
            cached = (resolved[0], resolved[1])
        self._bsd_think_tag_ids = cached
        return cached

    def _add_think_tag_metrics(self, batch: DataProto, response_mask: Any, metrics: dict[str, Any]) -> None:
        """Rate at which rollouts emit a well-formed ``<think>`` block.

        Counted on response token ids, not decoded text: both tags are single vocabulary
        entries, so this needs no decoding and cannot be skewed by re-tokenisation. A
        model that has to open the block itself fails in distinguishable ways
        -- no tag, an open whose close was cut off by the response budget, repeated tags,
        or a block that starts late -- so the rates stay separate. The definitions match
        ``tools/bsd/evaluate_checkpoints.py`` so training and eval are comparable.
        """
        if "responses" not in batch.batch:
            return
        open_id, close_id = self._think_tag_token_ids()
        if open_id is None or close_id is None:
            return
        responses = batch.batch["responses"]
        # Padding shares ids with real tokens, so only masked-in positions may count.
        opens = (responses == open_id) & response_mask
        closes = (responses == close_id) & response_mask
        open_counts = opens.sum(dim=-1)
        close_counts = closes.sum(dim=-1)
        # With exactly one hit, argmax over the boolean map is that hit's position.
        well_formed = (
            (open_counts == 1)
            & (close_counts == 1)
            & (opens.float().argmax(dim=-1) < closes.float().argmax(dim=-1))
        )
        # Responses are right-padded, so the first masked-in position is the first token
        # the policy actually generated. Reading it through the mask keeps the metric
        # correct if that padding convention ever changes.
        has_tokens = response_mask.any(dim=-1)
        first_index = response_mask.float().argmax(dim=-1, keepdim=True)
        first_token = responses.gather(1, first_index).squeeze(1)
        metrics["think/open_rate"] = open_counts.gt(0).float().mean().detach().item()
        metrics["think/close_rate"] = close_counts.gt(0).float().mean().detach().item()
        metrics["think/wellformed_rate"] = well_formed.float().mean().detach().item()
        metrics["think/open_first_token_rate"] = (
            ((first_token == open_id) & has_tokens).float().mean().detach().item()
        )
        metrics["think/duplicate_tag_rate"] = (
            ((open_counts > 1) | (close_counts > 1)).float().mean().detach().item()
        )
        metrics["think/opens_per_response_mean"] = open_counts.float().mean().detach().item()

    def _add_training_diagnostics(self, batch: DataProto, metrics: dict[str, Any]) -> None:
        if "response_mask" in batch.batch:
            response_mask = batch.batch["response_mask"].bool()
            response_lengths = response_mask.sum(dim=-1).float()
            metrics["train/step_student_tokens"] = float(response_mask.sum().detach().item())
            metrics["train/response_length_mean"] = response_lengths.mean().detach().item()
            metrics["train/truncation_proxy_last_token_ratio"] = response_mask[:, -1].float().mean().detach().item()
            metrics["response_length/mean"] = metrics["train/response_length_mean"]
            metrics["response_length/clip_ratio"] = metrics["train/truncation_proxy_last_token_ratio"]
            self._add_think_tag_metrics(batch, response_mask, metrics)
        if "token_level_scores" not in batch.batch or "response_mask" not in batch.batch:
            return
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        scores = (batch.batch["token_level_scores"] * batch.batch["response_mask"]).sum(dim=-1).float()
        uids = batch.non_tensor_batch.get("uid") if batch.non_tensor_batch is not None else None
        group_metrics, uid_seen, metric_ready = _group_diagnostic_metrics(scores, uids=uids, rollout_n=rollout_n)
        metrics.update(group_metrics)
        if metric_ready:
            metrics["train/group_metrics_uid_based"] = 1.0 if uid_seen else 0.0

    def _get_direct_opd_scorer(self, bsd_config: dict[str, Any]) -> InlineHFTeacherScorer:
        scorer = getattr(self, "_direct_opd_scorer", None)
        if scorer is not None:
            return scorer
        teacher_model = bsd_config.get("direct_opd_teacher_model_path")
        if not teacher_model:
            raise ValueError("direct_opd_teacher_model_path is required when direct_opd_enabled=True")
        device = bsd_config.get("direct_opd_teacher_device", "cuda")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "inline_external_hf requires CUDA visible in the trainer actor. "
                "Set RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 or use the multi-GPU teacher-pool entrypoint."
            )
        dtype_name = str(bsd_config.get("direct_opd_teacher_dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name)
        load_before = (
            torch.cuda.memory_allocated() / (1024**3)
            if str(device).startswith("cuda") and torch.cuda.is_available()
            else 0.0
        )
        scorer = InlineHFTeacherScorer(
            model_path=str(teacher_model),
            dtype=dtype,
            device=device,
            trust_remote_code=bool(bsd_config.get("trust_remote_code", True)),
            attn_implementation=str(bsd_config.get("direct_opd_attn_implementation", default_attn_implementation())),
        )
        assert_tokenizer_compatible(self.tokenizer, scorer.tokenizer)
        load_after = (
            torch.cuda.memory_allocated() / (1024**3)
            if str(device).startswith("cuda") and torch.cuda.is_available()
            else 0.0
        )
        self._direct_opd_inline_teacher_load_gb = max(0.0, load_after - load_before)
        self._direct_opd_scorer = scorer
        return scorer

    def _get_cross_vocab_opd_scorer(self, bsd_config: dict[str, Any]) -> CrossVocabOPDScorer:
        scorer = getattr(self, "_cross_vocab_opd_scorer", None)
        if scorer is not None:
            return scorer
        teacher_model = bsd_config.get("cross_vocab_opd_teacher_model_path")
        if not teacher_model:
            raise ValueError(
                "bsd.cross_vocab_opd_teacher_model_path is required for cross-vocabulary OPD"
            )
        backend = str(bsd_config.get("cross_vocab_opd_teacher_backend", "zmq_vllm"))
        if backend != "zmq_vllm":
            raise NotImplementedError("Cross-vocabulary OPD currently supports only the zmq_vllm teacher backend")
        scorer = CrossVocabOPDScorer(
            student_tokenizer=self.tokenizer,
            teacher_model_path=str(teacher_model),
            endpoint=str(bsd_config.get("cross_vocab_opd_teacher_endpoint", "tcp://127.0.0.1:15555")),
            max_sequence_length=int(bsd_config.get("cross_vocab_opd_teacher_max_sequence_length", 30720)),
            large_chunk_threshold=int(bsd_config.get("cross_vocab_opd_large_chunk_threshold", 6)),
            timeout_seconds=float(bsd_config.get("cross_vocab_opd_teacher_timeout_seconds", 600.0)),
            # The mode the dataset renders prompts with. A thinking-off Qwen3 template
            # pre-opens and closes the reasoning block, which moves the response boundary.
            student_thinking=optional_bool(
                (self.config.data.get("apply_chat_template_kwargs") or {}).get("enable_thinking")
            ),
        )
        self._cross_vocab_opd_scorer = scorer
        return scorer

    def _score_direct_opd_inline(self, build: PiTeacherBatchBuild, bsd_config: dict[str, Any]) -> torch.Tensor:
        scorer = self._get_direct_opd_scorer(bsd_config)
        tensors = build.batch.batch
        prefix_width = tensors["prompts"].shape[1]
        micro_batch_size = int(bsd_config.get("direct_opd_teacher_micro_batch_size", 1))
        outputs = []
        device = str(bsd_config.get("direct_opd_teacher_device", "cuda"))
        track_cuda = device.startswith("cuda") and torch.cuda.is_available()
        if track_cuda:
            torch.cuda.reset_peak_memory_stats()
        score_start = time.perf_counter()
        for start in range(0, tensors["responses"].shape[0], micro_batch_size):
            end = start + micro_batch_size
            scored = scorer.score(
                TeacherScoreRequest(
                    prompt_input_ids=tensors["prompts"][start:end],
                    prompt_attention_mask=tensors["attention_mask"][start:end, :prefix_width],
                    response_ids=tensors["responses"][start:end],
                    response_mask=tensors["response_mask"][start:end],
                    temperature=float(bsd_config.get("direct_opd_teacher_temperature", 1.0)),
                )
            )
            outputs.append(scored.response_log_probs.detach().cpu())
        self._direct_opd_inline_teacher_score_time = time.perf_counter() - score_start
        self._direct_opd_inline_teacher_max_memory_allocated_gb = (
            torch.cuda.max_memory_allocated() / (1024**3) if track_cuda else 0.0
        )
        self._direct_opd_inline_teacher_memory_allocated_gb = (
            torch.cuda.memory_allocated() / (1024**3) if track_cuda else 0.0
        )
        self._direct_opd_inline_teacher_memory_reserved_gb = (
            torch.cuda.memory_reserved() / (1024**3) if track_cuda else 0.0
        )
        return torch.cat(outputs, dim=0)

    def _before_actor_update(
        self,
        batch: DataProto,
        metrics: dict[str, Any],
        timing_raw: dict[str, Any],
    ) -> DataProto:
        del timing_raw
        policy_loss = self.config.actor_rollout_ref.actor.policy_loss
        rlsd_enabled = bool(policy_loss.get("rlsd_enabled", False))
        opd_pg_enabled = bool(policy_loss.get("opd_pg_enabled", False))
        cross_vocab_opd_enabled = bool(policy_loss.get("cross_vocab_opd_enabled", False))
        if sum((rlsd_enabled, opd_pg_enabled, cross_vocab_opd_enabled)) > 1:
            raise ValueError("RLSD, direct OPD-PG, and cross-vocabulary OPD modes are mutually exclusive")
        self._add_training_diagnostics(batch, metrics)
        bsd_config = self.config.get("bsd", {})
        if OmegaConf.is_config(bsd_config):
            bsd_config = OmegaConf.to_container(bsd_config, resolve=True)
        guided_cfg = self._guided_rescue_config()
        if guided_cfg is not None:
            apply_guided_is_weights(batch, guided_cfg, metrics)
            # Guided rows were sampled under the trace prompt, so rollout/actor agreement
            # (and the first-step consistency gate) is only meaningful on unguided rows.
            unguided = ~np.asarray(batch.non_tensor_batch[GUIDED_ROW_FLAG], dtype=bool)
            if unguided.any() and "rollout_log_probs" in batch.batch:
                from verl.utils.debug.metrics import calculate_debug_metrics

                metrics.update(calculate_debug_metrics(batch.select_idxs(unguided)))
            # The guided branch relies on rollout/actor agreement (rho uses old_log_prob),
            # so the gate applies even when the RLSD weighting is off.
            self._check_rollout_actor_consistency(metrics, bsd_config)
        truncation_aware = bool(bsd_config.get("grpo_truncation_aware", False))
        bridged_opd = bool(bsd_config.get("bridged_opd_enabled", False))
        bridged_critic = bool(bsd_config.get("bridged_critic_enabled", False))
        if bridged_opd and (rlsd_enabled or opd_pg_enabled or cross_vocab_opd_enabled or guided_cfg is not None):
            raise ValueError(
                "Bridged OPD is mutually exclusive with RLSD, OPD-PG, cross-vocabulary OPD, and guided rescue"
            )
        if bridged_critic and (bridged_opd or rlsd_enabled or opd_pg_enabled or cross_vocab_opd_enabled or guided_cfg is not None):
            raise ValueError("The bridged critic is mutually exclusive with RLSD, OPD-PG, cross-vocabulary OPD, Bridged OPD and guided rescue")
        if bridged_critic and not truncation_aware:
            raise ValueError("The bridged critic reweights the truncation-aware GRPO advantage: set grpo_truncation_aware=True")
        if truncation_aware and (opd_pg_enabled or cross_vocab_opd_enabled):
            raise ValueError("grpo_truncation_aware applies to verifier advantages; OPD modes do not use them")
        if truncation_aware:
            self._apply_truncation_aware_grpo(batch, metrics)
        if bridged_critic:
            self._apply_bridged_critic(batch, bsd_config, metrics)
            return batch
        if bridged_opd:
            self._check_rollout_actor_consistency(metrics, bsd_config)
            self._attach_pi_teacher_log_probs(batch, bsd_config, metrics)
            self._apply_bridged_opd(batch, bsd_config, metrics)
            return batch
        if not rlsd_enabled and not opd_pg_enabled and not cross_vocab_opd_enabled:
            return batch

        self._check_rollout_actor_consistency(metrics, bsd_config)
        if cross_vocab_opd_enabled:
            score = self._get_cross_vocab_opd_scorer(bsd_config).score_batch(batch)
            teacher_log_probs = score.teacher_log_probs.to(
                device=batch.batch["responses"].device,
                dtype=batch.batch["old_log_probs"].dtype,
            )
            chunk_ids = score.chunk_ids.to(device=batch.batch["responses"].device)
            expected_shape = batch.batch["response_mask"].shape
            if teacher_log_probs.shape != expected_shape or chunk_ids.shape != expected_shape:
                raise ValueError(
                    "Cross-vocabulary teacher tensor shape mismatch: "
                    f"log_probs={tuple(teacher_log_probs.shape)}, chunk_ids={tuple(chunk_ids.shape)}, "
                    f"response_mask={tuple(expected_shape)}"
                )
            aligned_mask = batch.batch["response_mask"].bool() & (chunk_ids >= 0)
            batch.batch["cross_vocab_teacher_log_probs"] = teacher_log_probs
            batch.batch["cross_vocab_chunk_ids"] = chunk_ids
            metrics.update(score.metrics)
            metrics["train/step_teacher_tokens"] = (
                score.metrics["cross_vocab_opd/teacher_prompt_tokens"]
                + score.metrics["cross_vocab_opd/teacher_response_tokens"]
            )
            metrics["cross_vocab_opd/teacher_logp_mean"] = masked_mean(
                teacher_log_probs, aligned_mask
            ).detach().item()
            metrics["cross_vocab_opd/teacher_delta_mean"] = masked_mean(
                teacher_log_probs - batch.batch["old_log_probs"], aligned_mask
            ).detach().item()
            metrics["cross_vocab_opd/uses_task_rewards"] = 0.0
            return batch
        if opd_pg_enabled:
            backend = bsd_config.get("direct_opd_teacher_backend", "inline_external_hf")
            if backend != "inline_external_hf":
                raise NotImplementedError(
                    "Direct OPD-PG only supports direct_opd_teacher_backend=inline_external_hf"
                )
            max_prompt_length = int(
                bsd_config.get("direct_opd_teacher_max_prompt_length", self.config.data.max_prompt_length)
            )
            allow_truncation = bool(bsd_config.get("allow_teacher_prompt_truncation", False))
            teacher_thinking = bool(bsd_config.get("direct_opd_teacher_thinking", False))
            build = build_direct_opd_teacher_score_batch(
                source_batch=batch,
                tokenizer=self.tokenizer,
                max_prompt_length=max_prompt_length,
                allow_prompt_truncation=allow_truncation,
                teacher_thinking=teacher_thinking,
            )
            teacher_log_probs = self._score_direct_opd_inline(build, bsd_config).to(
                device=batch.batch["responses"].device,
                dtype=batch.batch["old_log_probs"].dtype,
            )
            if teacher_log_probs.shape != batch.batch["response_mask"].shape:
                raise ValueError(
                    "teacher_opd_log_probs shape mismatch: "
                    f"{tuple(teacher_log_probs.shape)} vs response_mask {tuple(batch.batch['response_mask'].shape)}"
                )
            batch.batch["teacher_opd_log_probs"] = teacher_log_probs * batch.batch["response_mask"].to(
                teacher_log_probs
            )
            mask = batch.batch["response_mask"].bool()
            metrics.update(build.metrics)
            metrics["train/step_teacher_tokens"] = metrics["opd/teacher_response_tokens"]
            metrics["opd/teacher_logp_mean"] = masked_mean(teacher_log_probs, mask).detach().item()
            metrics["opd/teacher_delta_mean"] = masked_mean(
                teacher_log_probs - batch.batch["old_log_probs"], mask
            ).detach().item()
            metrics["opd/inline_teacher_load_gb"] = float(getattr(self, "_direct_opd_inline_teacher_load_gb", 0.0))
            metrics["opd/inline_teacher_score_time"] = float(
                getattr(self, "_direct_opd_inline_teacher_score_time", 0.0)
            )
            metrics["opd/inline_teacher_max_memory_allocated_gb"] = float(
                getattr(self, "_direct_opd_inline_teacher_max_memory_allocated_gb", 0.0)
            )
            metrics["opd/inline_teacher_memory_allocated_gb"] = float(
                getattr(self, "_direct_opd_inline_teacher_memory_allocated_gb", 0.0)
            )
            metrics["opd/inline_teacher_memory_reserved_gb"] = float(
                getattr(self, "_direct_opd_inline_teacher_memory_reserved_gb", 0.0)
            )
            return batch

        effective_lambda = self._compute_effective_rlsd_lambda(policy_loss)
        batch.meta_info["rlsd_effective_lambda"] = effective_lambda
        metrics["rlsd/effective_lambda"] = effective_lambda
        rlsd_token_mask_spec = str(policy_loss.get("rlsd_token_mask", "full"))
        batch.batch["rlsd_token_mask"] = _build_rlsd_token_mask(
            batch.batch["response_mask"],
            rlsd_token_mask_spec,
            responses=batch.batch["responses"],
            tokenizer=self.tokenizer,
        ).to(device=batch.batch["response_mask"].device)
        if guided_cfg is not None:
            # Guided rows carry no privileged-self credit: rho_t is their only reweighting.
            guided_rows = torch.from_numpy(np.asarray(batch.non_tensor_batch[GUIDED_ROW_FLAG], dtype=bool))
            batch.batch["rlsd_token_mask"][guided_rows.to(batch.batch["rlsd_token_mask"].device)] = 0
        metrics["rlsd/token_mask_ratio"] = (
            batch.batch["rlsd_token_mask"].sum()
            / batch.batch["response_mask"].to(batch.batch["rlsd_token_mask"]).sum().clamp_min(1)
        ).detach().item()

        self._attach_pi_teacher_log_probs(batch, bsd_config, metrics)
        return batch

    def _completed_rows(self, batch: DataProto) -> torch.Tensor:
        """Per row: generation ended before the response-length budget (not a budget stop).

        A response that fills the budget still counts as completed when its last token is a stop token. Besides the
        tokenizer EOS this includes the pad id: Qwen-family generation configs list ``<|endoftext|>`` (the pad
        token) as a second EOS, and a pad id can only be the last valid token when the model generated it.
        """
        response_mask = batch.batch["response_mask"].bool()
        lengths = response_mask.sum(dim=-1)
        budget = int(self.config.data.max_response_length)
        last = batch.batch["responses"].gather(1, (lengths - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        stop_ids: list[int] = []
        for value in (getattr(self.tokenizer, "eos_token_id", None), getattr(self.tokenizer, "pad_token_id", None)):
            stop_ids.extend(value if isinstance(value, (list, tuple)) else [value] if value is not None else [])
        ends_with_stop = torch.isin(last.cpu(), torch.tensor(stop_ids)) if stop_ids else torch.zeros_like(last.cpu(), dtype=bool)
        return ((lengths.cpu() < budget) | ends_with_stop) & (lengths.cpu() > 0)

    def _apply_truncation_aware_grpo(self, batch: DataProto, metrics: dict[str, Any]) -> None:
        """Replace the upstream GRPO advantages with completed-only group statistics."""
        if str(self.config.algorithm.adv_estimator) != "grpo":
            raise ValueError("grpo_truncation_aware requires algorithm.adv_estimator=grpo")
        response_mask = batch.batch["response_mask"]
        scores = (batch.batch["token_level_rewards"] * response_mask).sum(dim=-1)
        completed = self._completed_rows(batch)
        advantages, ta_metrics = compute_truncation_aware_grpo_advantages(
            scores=scores.detach().cpu(),
            index=batch.non_tensor_batch["uid"],
            completed=completed,
            response_mask=response_mask.detach().cpu(),
            norm_adv_by_std=bool(self.config.algorithm.get("norm_adv_by_std_in_grpo", True)),
        )
        batch.batch["advantages"] = advantages.to(device=response_mask.device, dtype=batch.batch["advantages"].dtype)
        batch.batch["returns"] = batch.batch["advantages"].clone()
        metrics.update(ta_metrics)

    def _apply_bridged_opd(self, batch: DataProto, bsd_config: dict[str, Any], metrics: dict[str, Any]) -> None:
        """Bridged OPD: dense sampled-token reverse-KL credit toward the PI-conditioned self-teacher.

        A_t = verifier_coef * A_grpo + coef * clip(log T(y_t|x,tau,y<t) - log pi_old(y_t|x,y<t)) * scope,
        with scope = completed verifier-failed rows (``failed_completed``) or every row (``all``).
        pi_old is exact for the update only when each step makes one optimizer update (ppo_epochs=1 and
        ppo_mini_batch_size = train_batch_size); this method checks both.
        """
        actor_cfg = self.config.actor_rollout_ref.actor
        if int(actor_cfg.get("ppo_epochs", 1)) != 1 or int(actor_cfg.get("ppo_mini_batch_size", -1)) != int(
            self.config.data.train_batch_size
        ):
            raise ValueError(
                "Bridged OPD builds its dense advantage from old-policy log-probs, which equal the current policy only "
                "with one optimizer update per step: set ppo_epochs=1 and ppo_mini_batch_size=train_batch_size"
            )
        scope_name = str(bsd_config.get("bridged_opd_scope", "failed_completed"))
        coef = float(bsd_config.get("bridged_opd_coef", 1.0))
        verifier_coef = float(bsd_config.get("bridged_opd_verifier_coef", 1.0))
        clip_abs = float(bsd_config.get("bridged_opd_clip_abs", 2.0))
        response_mask = batch.batch["response_mask"]
        mask = response_mask.bool()
        teacher = batch.batch["teacher_pi_log_probs"]
        old = batch.batch["old_log_probs"]
        if not torch.isfinite(teacher[mask]).all() or not torch.isfinite(old[mask]).all():
            raise ValueError("Bridged OPD received non-finite teacher or student log-probabilities")
        completed = self._completed_rows(batch).to(mask.device)
        passed = _verifier_passed(batch).to(mask.device)
        if scope_name == "failed_completed":
            scope_rows = completed & ~passed
        elif scope_name == "all":
            scope_rows = torch.ones_like(completed)
        else:
            raise ValueError(f"bridged_opd_scope must be failed_completed or all, got {scope_name!r}")
        scope = mask & scope_rows.unsqueeze(1)
        dense, opd_metrics = compute_opd_pg_advantages(
            teacher_log_probs=teacher,
            student_old_log_probs=old,
            response_mask=scope.to(response_mask.dtype),
            logratio_clip_abs=clip_abs,
        )
        verifier = batch.batch["advantages"]
        batch.batch["advantages"] = (verifier_coef * verifier + coef * dense.to(verifier.dtype)) * response_mask.to(
            verifier.dtype
        )
        batch.batch["returns"] = batch.batch["advantages"].clone()
        delta = (teacher - old).detach()
        metrics.update({f"bridged_opd/{k.split('/', 1)[-1]}": v for k, v in opd_metrics.items()})
        metrics.update(
            {
                "bridged_opd/coef": coef,
                "bridged_opd/verifier_coef": verifier_coef,
                "bridged_opd/scope_row_ratio": scope_rows.float().mean().item(),
                "bridged_opd/scope_token_ratio": (scope.sum() / mask.sum().clamp_min(1)).item(),
                "bridged_opd/dense_abs_mean": masked_mean((coef * dense).abs(), scope).item() if scope.any() else 0.0,
                "bridged_opd/verifier_abs_mean": masked_mean((verifier_coef * verifier).abs(), mask).item(),
                # Online inversion check: mean delta on correct vs failed completed rows.
                "bridged_opd/delta_mean_correct": masked_mean(delta, mask & (completed & passed).unsqueeze(1)).item()
                if (completed & passed).any()
                else float("nan"),
                "bridged_opd/delta_mean_failed": masked_mean(delta, mask & (completed & ~passed).unsqueeze(1)).item()
                if (completed & ~passed).any()
                else float("nan"),
                "bridged_opd/censored_ratio": (~completed).float().mean().item(),
            }
        )

    def _apply_bridged_critic(self, batch: DataProto, bsd_config: dict[str, Any], metrics: dict[str, Any]) -> None:
        """Bridged critic: reweight the verifier advantage of failed rollouts by a privileged self-critic.

        The critic is the current policy (no snapshot, so it follows the policy at every step). It reads the problem,
        the external teacher's trace as a reference solution and the student's own response cut after N tokens, and
        its log-probability of the single response token "Yes" to "is everything so far correct and on a path to the
        correct final answer" is read in one forward pass without gradient. Only completed rows of groups with both
        outcomes are probed; compute_bridged_critic_weights turns the scores into per-token weights of the
        truncation-aware GRPO advantage (blame moves inside a failed row toward the segments where the critic's value
        drops; sign and row total unchanged). The trace enters only the critic prompt; the training prompt stays
        PI-free.
        """
        cfg = BridgedCriticConfig.from_config(bsd_config)
        start = time.perf_counter()
        response_mask = batch.batch["response_mask"]
        completed = self._completed_rows(batch)
        passed = _verifier_passed(batch).cpu()
        uids = np.asarray(batch.non_tensor_batch["uid"])
        groups: dict[Any, list[int]] = defaultdict(list)
        for row in torch.nonzero(completed).flatten().tolist():
            groups[uids[row]].append(int(row))
        mixed_rows = [
            row for rows in groups.values() if 0 < sum(bool(passed[r]) for r in rows) < len(rows) for row in rows
        ]
        _, close_id = self._think_tag_token_ids()
        probes = plan_bridged_critic_probes(
            batch.batch["responses"].cpu(),
            response_mask.cpu(),
            rows=mixed_rows,
            cuts=cfg.cuts,
            min_tail=cfg.min_tail,
            close_id=close_id,
        )
        scores: dict[tuple[int, int], float] = {}
        if probes:
            build = build_bridged_critic_score_batch(
                source_batch=batch,
                tokenizer=self.tokenizer,
                probes=probes,
                pi_trace_field=cfg.pi_trace_field,
                max_prompt_length=cfg.max_prompt_length,
                pad_multiple=int(self.actor_rollout_wg.world_size),
            )
            metrics.update(build.metrics)
            if build.probes:
                scored = self._compute_actor_log_prob(build.batch)
                logp_yes = scored.batch["ref_log_prob"][: len(build.probes), 0].detach().float().cpu()
                if not torch.isfinite(logp_yes).all():
                    raise ValueError("The bridged critic returned a non-finite log P(Yes)")
                scores = dict(zip(build.probes, logp_yes.tolist()))
                metrics["bridged_critic/logp_yes_mean"] = float(logp_yes.mean())
        weights, weight_metrics = compute_bridged_critic_weights(
            lengths=response_mask.sum(dim=-1).cpu().tolist(),
            uids=uids,
            completed=completed.tolist(),
            passed=passed.tolist(),
            scores=scores,
            calibration=cfg.calibration,
            beta=cfg.beta,
            width=int(response_mask.shape[1]),
        )
        advantages = batch.batch["advantages"]
        reweighted = advantages * weights.to(device=advantages.device, dtype=advantages.dtype)
        batch.batch["advantages"] = reweighted
        batch.batch["returns"] = reweighted.clone()
        metrics.update(weight_metrics)
        metrics["bridged_critic/advantage_change"] = (
            (reweighted - advantages).abs().sum() / advantages.abs().sum().clamp_min(1e-12)
        ).item()
        metrics["bridged_critic/mixed_group_rows"] = float(len(mixed_rows))
        metrics["bridged_critic/probes"] = float(len(scores))
        metrics["bridged_critic/beta"] = cfg.beta
        metrics["bridged_critic/seconds"] = time.perf_counter() - start

    def _attach_pi_teacher_log_probs(
        self, batch: DataProto, bsd_config: dict[str, Any], metrics: dict[str, Any]
    ) -> None:
        """Score the original response ids under the privileged (PI) self-teacher prompt.

        Writes ``teacher_pi_log_probs`` (masked to the response) into the batch. Shared by
        RLSD credit and Bridged OPD; the training prompt stays PI-free.
        """
        # The launchers always pass teacher_update_mode (BSD: periodic_snapshot). The fallback for a
        # missing key stays the adapter-free mode, because periodic_snapshot is only valid together
        # with actor_rollout_ref.model.bsd_teacher_adapter=True, which is off by default.
        teacher_update_mode = str(bsd_config.get("teacher_update_mode", "base_no_adapter")).lower()

        max_prompt_length = int(bsd_config.get("teacher_max_prompt_length", self.config.data.max_prompt_length))
        allow_truncation = bool(bsd_config.get("allow_teacher_prompt_truncation", False))
        teacher_thinking = bool(bsd_config.get("online_teacher_thinking", False))
        student_thinking = bsd_config.get("student_thinking")
        validate_student_prefix = student_thinking is not None and not getattr(
            self, "_bsd_prompt_serialization_checked", False
        )
        pi_source = normalize_pi_source(bsd_config.get("pi_source", PI_SOURCE_TRACE_FIELD))
        pi_trace_field = str(bsd_config.get("pi_trace_field", DEFAULT_PI_TRACE_FIELD))
        build = build_pi_teacher_score_batch(
            source_batch=batch,
            tokenizer=self.tokenizer,
            max_prompt_length=max_prompt_length,
            allow_prompt_truncation=allow_truncation,
            teacher_thinking=teacher_thinking,
            student_thinking=bool(student_thinking) if validate_student_prefix else None,
            emit_think=optional_bool(bsd_config.get("emit_think")),
            pi_trace_field=pi_trace_field,
            pi_source=pi_source,
            pi_teacher_template=str(bsd_config.get("pi_teacher_template") or "") or None,
        )
        if validate_student_prefix:
            self._bsd_prompt_serialization_checked = True
            metrics["bsd/student_prompt_prefix_token_match"] = 1.0
        teacher_ref = self._compute_pi_teacher_log_prob(
            build.batch,
            teacher_update_mode=teacher_update_mode,
            bsd_config=bsd_config,
            metrics=metrics,
        )
        teacher_log_probs = teacher_ref.batch["ref_log_prob"].detach().to(
            device=batch.batch["responses"].device,
            dtype=batch.batch["old_log_probs"].dtype,
        )
        if teacher_log_probs.shape != batch.batch["response_mask"].shape:
            raise ValueError(
                "teacher_pi_log_probs shape mismatch: "
                f"{tuple(teacher_log_probs.shape)} vs response_mask {tuple(batch.batch['response_mask'].shape)}"
            )
        batch.batch["teacher_pi_log_probs"] = teacher_log_probs * batch.batch["response_mask"].to(teacher_log_probs)

        mask = batch.batch["response_mask"].bool()
        metrics.update(build.metrics)
        metrics["bsd/pi_trace_field_is_qwen32b"] = float(
            pi_source == PI_SOURCE_TRACE_FIELD and pi_trace_field == DEFAULT_PI_TRACE_FIELD
        )
        metrics["bsd/pi_teacher_prompt_tokens_total"] = metrics["bsd/pi_teacher_prompt_tokens_mean"] * float(
            batch.batch["response_mask"].shape[0]
        )
        metrics["train/step_teacher_tokens"] = (
            metrics["bsd/pi_teacher_prompt_tokens_total"] + metrics["bsd/pi_teacher_response_tokens"]
        )
        metrics["bsd/teacher_pi_logp_mean"] = masked_mean(teacher_log_probs, mask).detach().item()
        metrics["bsd/teacher_pi_delta_mean"] = masked_mean(
            teacher_log_probs - batch.batch["old_log_probs"], mask
        ).detach().item()
