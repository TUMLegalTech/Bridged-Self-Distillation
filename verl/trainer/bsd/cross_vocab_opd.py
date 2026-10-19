# Copyright 2025 Individual Contributor: furunding
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-tokenizer OPD alignment, teacher RPC, and semantic-prior credit assignment.

The alignment and credit-assignment rules are adapted from
ivanniu/On-Policy-Distill at commit 927a8264f2e303b7f82c2d331a58fd4240c8805a.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import msgpack
import torch
import zmq

PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class CrossVocabAlignment:
    """Teacher log probabilities aligned to student response tokens."""

    log_probs: torch.Tensor
    chunk_ids: torch.LongTensor
    chunk_count: int
    invalid_token_count: int
    large_chunk_count: int
    teacher_truncated: bool


@dataclass(frozen=True)
class CrossVocabScoreBatch:
    """Padded tensors and diagnostics produced by cross-vocabulary scoring."""

    teacher_log_probs: torch.Tensor
    chunk_ids: torch.LongTensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class _RetokenizedRow:
    student_content_ids: list[int]
    teacher_content_ids: list[int]
    teacher_prefix_length: int
    student_special_count: int
    teacher_terminal_position: int | None
    skip_alignment: bool


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash vocabulary and prompt-critical tokenizer metadata."""

    payload = {
        "vocab": sorted((str(token), int(token_id)) for token, token_id in tokenizer.get_vocab().items()),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "chat_template": getattr(tokenizer, "chat_template", None),
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    text = tokenizer.decode(list(token_ids), skip_special_tokens=False)
    return unicodedata.normalize("NFC", text)


def _decode_serialized_sequence(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode the full serialized sequence exactly as the source retokenization path does."""

    return tokenizer.decode(list(token_ids), skip_special_tokens=False)


def _detect_tokenizer_family(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()
    if "<|begin_of_text|>" in vocab:
        return "llama"
    if "<|im_start|>" in vocab:
        return "qwen"
    if "<｜begin▁of▁sentence｜>" in vocab:
        return "deepseek"
    return "unknown"


def build_chat_template_mapping(student_tokenizer: Any, teacher_tokenizer: Any) -> list[tuple[str, str]]:
    """Return the serialized-chat marker replacements used by the source OPD implementation."""

    student_family = _detect_tokenizer_family(student_tokenizer)
    teacher_family = _detect_tokenizer_family(teacher_tokenizer)
    if student_family == "llama" and teacher_family == "qwen":
        return [
            ("<|begin_of_text|><|start_header_id|>", "<|im_start|>"),
            ("<|start_header_id|>", "<|im_start|>"),
            ("<|end_header_id|>\n\n", "\n"),
            ("<|eot_id|>", "<|im_end|>\n"),
            ("<|begin_of_text|>", ""),
            ("<|end_of_text|>", "<|im_end|>\n"),
        ]
    if student_family == "llama" and teacher_family == "deepseek":
        return [
            ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", "<｜begin▁of▁sentence｜>"),
            ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n", "<｜begin▁of▁sentence｜><｜User｜>"),
            ("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", "<｜User｜>"),
            ("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "<｜Assistant｜>"),
            ("<|eot_id|>", "<｜end▁of▁sentence｜>"),
            ("<|begin_of_text|>", "<｜begin▁of▁sentence｜>"),
            ("<|end_of_text|>", "<｜end▁of▁sentence｜>"),
        ]
    if student_family == "qwen" and teacher_family == "deepseek":
        return [
            ("<|im_start|>system\n", "<｜begin▁of▁sentence｜>"),
            ("<|im_end|>\n<|im_start|>user\n", "<｜User｜>"),
            ("<|im_start|>user\n", "<｜begin▁of▁sentence｜><｜User｜>"),
            ("<|im_end|>\n<|im_start|>assistant\n", "<｜Assistant｜>"),
            ("<|im_end|>\n", "<｜end▁of▁sentence｜>"),
            ("<|im_end|>", "<｜end▁of▁sentence｜>"),
            ("<|endoftext|>", "<｜end▁of▁sentence｜>"),
        ]
    if (student_family, teacher_family) in {("qwen", "qwen"), ("deepseek", "deepseek")}:
        return []
    raise NotImplementedError(
        f"Unsupported cross-tokenizer pair: student={student_family!r}, teacher={teacher_family!r}"
    )


def _apply_chat_template_mapping(text: str, mapping: Sequence[tuple[str, str]]) -> str:
    for old, new in mapping:
        text = text.replace(old, new)
    return text


def _family_response_marker(tokenizer: Any) -> str:
    family = _detect_tokenizer_family(tokenizer)
    if family == "llama":
        return "<|end_header_id|>\n\n"
    if family == "qwen":
        return "<|im_start|>assistant\n"
    if family == "deepseek":
        return "<｜Assistant｜>"
    raise NotImplementedError(f"Unsupported teacher tokenizer family: {family!r}")


def generation_prefix_tail(tokenizer: Any, enable_thinking: bool | None = None) -> str:
    """Return the text a chat template appends after the assistant header.

    The source implementation locates the response with one hardcoded marker per tokenizer
    family, which assumes the response starts immediately after that header. Templates that
    pre-open the reasoning block append text there -- ``<think>\\n`` for a thinking-on
    Nemotron or a prefilled template, ``<think>\\n\\n</think>\\n\\n`` for a thinking-off
    Qwen3. Those tokens belong to the prompt, so they must be part of the marker; otherwise
    the response boundary lands that many tokens early and every teacher log probability is
    shifted. Templates that append nothing return ``""``, which keeps the source behaviour.

    ``enable_thinking`` must be the mode the run renders its prompts with, because for
    Qwen3 the appended text depends on it. Leaving it ``None`` takes the template default.
    """

    messages = [{"role": "user", "content": "x"}]
    kwargs = {} if enable_thinking is None else {"enable_thinking": bool(enable_thinking)}
    try:
        with_generation = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        without_generation = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **kwargs
        )
    except Exception:
        return ""
    if not isinstance(with_generation, str) or not isinstance(without_generation, str):
        return ""
    if not with_generation.startswith(without_generation):
        return ""
    generation_prefix = with_generation[len(without_generation) :]
    marker = _family_response_marker(tokenizer)
    marker_position = generation_prefix.rfind(marker)
    if marker_position < 0:
        return ""
    return generation_prefix[marker_position + len(marker) :]


def _response_marker(tokenizer: Any, generation_tail: str = "") -> str:
    return _family_response_marker(tokenizer) + generation_tail


def _find_response_start(
    tokenizer: Any,
    token_ids: Sequence[int],
    generation_tail: str = "",
    prompt_length: int | None = None,
) -> int | None:
    """Locate the first response token in a serialized prompt+response sequence.

    ``prompt_length`` bounds the marker search to the first that many tokens. The
    generation prefix is by definition part of the prompt, so a caller that knows where
    the prompt ends should say so: otherwise a model that emits the marker text inside
    its own response moves ``rfind`` past the real boundary.
    """

    marker = _response_marker(tokenizer, generation_tail)
    search_ids = token_ids if prompt_length is None else token_ids[:prompt_length]
    response_start_char = _decode_serialized_sequence(tokenizer, search_ids).rfind(marker)
    if response_start_char < 0:
        return None
    response_start_char += len(marker)
    for token_index in range(len(token_ids)):
        prefix_text = _decode_serialized_sequence(tokenizer, token_ids[: token_index + 1])
        if len(prefix_text) >= response_start_char:
            return token_index + 1
    return None


def align_cross_vocab_response(
    *,
    student_ids: Sequence[int],
    teacher_ids: Sequence[int],
    teacher_log_probs: torch.Tensor,
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    large_chunk_threshold: int = 6,
) -> CrossVocabAlignment:
    """Greedily synchronize equal decoded text chunks across two tokenizers."""

    student_ids = [int(token_id) for token_id in student_ids]
    teacher_ids = [int(token_id) for token_id in teacher_ids]
    teacher_log_probs = torch.as_tensor(teacher_log_probs, dtype=torch.float32).detach().cpu()
    if teacher_log_probs.ndim != 1 or teacher_log_probs.numel() != len(teacher_ids):
        raise ValueError(
            "teacher_log_probs must have one value per teacher response token: "
            f"shape={tuple(teacher_log_probs.shape)}, teacher_tokens={len(teacher_ids)}"
        )
    if large_chunk_threshold <= 0:
        raise ValueError("large_chunk_threshold must be positive")
    if teacher_log_probs.numel() and not torch.isfinite(teacher_log_probs).all():
        raise ValueError("Cross-vocabulary OPD received non-finite teacher log probabilities")

    aligned = torch.zeros(len(student_ids), dtype=torch.float32)
    chunk_ids = torch.full((len(student_ids),), -1, dtype=torch.long)
    if not student_ids or not teacher_ids:
        return CrossVocabAlignment(aligned, chunk_ids, 0, len(student_ids), 0, False)

    student_pos = 0
    teacher_pos = 0
    chunk_count = 0
    large_chunk_count = 0
    while student_pos < len(student_ids) and teacher_pos < len(teacher_ids):
        student_end = student_pos + 1
        teacher_end = teacher_pos + 1
        matched = False
        while student_end <= len(student_ids) and teacher_end <= len(teacher_ids):
            student_chunk = _decode(student_tokenizer, student_ids[student_pos:student_end])
            teacher_chunk = _decode(teacher_tokenizer, teacher_ids[teacher_pos:teacher_end])
            if student_chunk == teacher_chunk and not student_chunk.endswith("\ufffd"):
                student_width = student_end - student_pos
                teacher_width = teacher_end - teacher_pos
                if student_width > large_chunk_threshold or teacher_width > large_chunk_threshold:
                    aligned[student_pos:student_end] = float("inf")
                    large_chunk_count += 1
                else:
                    chunk_log_prob = teacher_log_probs[teacher_pos:teacher_end].sum()
                    aligned[student_pos:student_end] = chunk_log_prob / student_width
                    chunk_ids[student_pos:student_end] = chunk_count
                chunk_count += 1
                student_pos = student_end
                teacher_pos = teacher_end
                matched = True
                break

            previous = (student_end, teacher_end)
            if len(student_chunk) < len(teacher_chunk):
                if student_end < len(student_ids):
                    student_end += 1
                elif teacher_end < len(teacher_ids):
                    teacher_end += 1
            elif len(student_chunk) > len(teacher_chunk):
                if teacher_end < len(teacher_ids):
                    teacher_end += 1
                elif student_end < len(student_ids):
                    student_end += 1
            else:
                student_incomplete = student_chunk.endswith("\ufffd")
                teacher_incomplete = teacher_chunk.endswith("\ufffd")
                if student_incomplete and not teacher_incomplete and student_end < len(student_ids):
                    student_end += 1
                elif teacher_incomplete and not student_incomplete and teacher_end < len(teacher_ids):
                    teacher_end += 1
                else:
                    if student_end < len(student_ids):
                        student_end += 1
                    if teacher_end < len(teacher_ids):
                        teacher_end += 1
            if previous == (student_end, teacher_end):
                break

        if not matched:
            aligned[student_pos:] = float("inf")
            student_pos = len(student_ids)
            teacher_pos = len(teacher_ids)
            break

    teacher_truncated = student_pos < len(student_ids)
    if teacher_truncated:
        aligned[student_pos:] = float("inf")
    invalid_token_count = int((chunk_ids < 0).sum().item())
    return CrossVocabAlignment(
        log_probs=aligned,
        chunk_ids=chunk_ids,
        chunk_count=chunk_count,
        invalid_token_count=invalid_token_count,
        large_chunk_count=large_chunk_count,
        teacher_truncated=teacher_truncated,
    )


def compute_cross_vocab_opd_advantages(
    *,
    teacher_log_probs: torch.Tensor,
    student_old_log_probs: torch.Tensor,
    chunk_ids: torch.Tensor,
    response_mask: torch.Tensor,
    loss_max_clamp: float | None = 10.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply source-compatible OPD semantic-prior credit from old-policy log probabilities."""

    if not (
        teacher_log_probs.shape
        == student_old_log_probs.shape
        == chunk_ids.shape
        == response_mask.shape
    ):
        raise ValueError("Cross-vocabulary OPD tensors must have identical shapes")
    if loss_max_clamp is not None and loss_max_clamp <= 0:
        raise ValueError("cross_vocab_opd_loss_max_clamp must be positive or null")

    with torch.no_grad():
        response_mask = response_mask.bool()
        sentinel_mask = torch.isinf(teacher_log_probs) | (chunk_ids < 0)
        valid = response_mask & (~sentinel_mask)
        for name, tensor in {
            "teacher_log_probs": teacher_log_probs,
            "student_old_log_probs": student_old_log_probs,
        }.items():
            if not torch.isfinite(tensor[valid]).all():
                bad = int((~torch.isfinite(tensor[valid])).sum().item())
                raise ValueError(f"Cross-vocabulary OPD found non-finite {name} on {bad} aligned tokens")

        advantages = torch.zeros_like(student_old_log_probs)
        chunk_total = 0
        for row_idx in range(student_old_log_probs.shape[0]):
            row_valid = valid[row_idx]
            for chunk_id in torch.unique(chunk_ids[row_idx][row_valid]):
                chunk_mask = row_valid & (chunk_ids[row_idx] == chunk_id)
                teacher_budget = teacher_log_probs[row_idx][chunk_mask].sum()
                student_budget = student_old_log_probs[row_idx][chunk_mask].sum()
                if student_budget.abs() < 1e-8:
                    target = torch.full_like(
                        student_old_log_probs[row_idx][chunk_mask],
                        teacher_budget / chunk_mask.sum().clamp_min(1),
                    )
                else:
                    target = (teacher_budget / student_budget) * student_old_log_probs[row_idx][chunk_mask]
                advantages[row_idx][chunk_mask] = target - student_old_log_probs[row_idx][chunk_mask]
                chunk_total += 1

        unclipped = advantages.clone()
        if loss_max_clamp is not None:
            advantages = advantages.clamp(min=-loss_max_clamp, max=loss_max_clamp)
        advantages = advantages * valid.to(advantages.dtype)

        valid_advantages = advantages[valid]
        response_tokens = int(response_mask.sum().item())
        invalid_tokens = int((sentinel_mask & response_mask).sum().item())
        if valid_advantages.numel():
            advantage_mean = float(valid_advantages.mean().item())
            advantage_std = float(valid_advantages.std(unbiased=False).item())
            advantage_abs_max = float(valid_advantages.abs().max().item())
            clamp_ratio = float(((unclipped[valid] != advantages[valid]).to(torch.float32)).mean().item())
        else:
            advantage_mean = advantage_std = advantage_abs_max = clamp_ratio = 0.0
        metrics = {
            "cross_vocab_opd/aligned_token_ratio": float(valid.sum().item() / max(response_tokens, 1)),
            "cross_vocab_opd/chunk_count": float(chunk_total),
            "cross_vocab_opd/advantage_mean": advantage_mean,
            "cross_vocab_opd/advantage_std": advantage_std,
            "cross_vocab_opd/advantage_abs_max": advantage_abs_max,
            "cross_vocab_opd/clamp_ratio": clamp_ratio,
            "actor/opd_inf_tokens": float(invalid_tokens),
            "actor/opd_inf_ratio": float(invalid_tokens / max(response_tokens, 1)),
        }
    return advantages, metrics


def pack_rpc_message(payload: dict[str, Any]) -> bytes:
    """Serialize one versioned teacher RPC message."""

    return msgpack.packb(payload, use_bin_type=True)


def unpack_rpc_message(payload: bytes) -> dict[str, Any]:
    """Deserialize one teacher RPC message and validate its top-level type."""

    value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    if not isinstance(value, dict):
        raise ValueError("Cross-vocabulary teacher RPC payload must be a mapping")
    return value


class CrossVocabTeacherClient:
    """Synchronous fail-closed ZeroMQ client for the external teacher server."""

    def __init__(self, endpoint: str, *, timeout_seconds: float = 600.0) -> None:
        if not endpoint.startswith("tcp://"):
            raise ValueError(f"Teacher endpoint must use tcp://, got {endpoint!r}")
        if timeout_seconds <= 0:
            raise ValueError("Teacher RPC timeout must be positive")
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_seconds * 1000)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        context = zmq.Context.instance()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.connect(self.endpoint)
        try:
            socket.send(pack_rpc_message(payload))
            response = unpack_rpc_message(socket.recv())
        except zmq.Again as exc:
            raise TimeoutError(f"Timed out waiting for cross-vocabulary teacher at {self.endpoint}") from exc
        finally:
            socket.close()
        if response.get("status") != "ok":
            raise RuntimeError(f"Cross-vocabulary teacher failed: {response.get('error', 'unknown error')}")
        return response

    def health(self, *, expected_tokenizer_fingerprint: str | None = None) -> dict[str, Any]:
        response = self._request({"protocol_version": PROTOCOL_VERSION, "action": "health"})
        if response.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError("Cross-vocabulary teacher protocol version mismatch")
        actual = response.get("tokenizer_fingerprint")
        if expected_tokenizer_fingerprint is not None and actual != expected_tokenizer_fingerprint:
            raise RuntimeError(
                "Cross-vocabulary teacher tokenizer mismatch: "
                f"expected={expected_tokenizer_fingerprint}, server={actual}"
            )
        return response

    def score(self, input_ids: Sequence[Sequence[int]]) -> list[list[float]]:
        rows = [[int(token_id) for token_id in row] for row in input_ids]
        if not rows or any(len(row) < 2 for row in rows):
            raise ValueError("Teacher scoring requires non-empty sequences with at least two tokens")
        response = self._request(
            {
                "protocol_version": PROTOCOL_VERSION,
                "action": "score",
                "input_ids": rows,
            }
        )
        log_probs = response.get("log_probs")
        if not isinstance(log_probs, list) or len(log_probs) != len(rows):
            raise RuntimeError("Cross-vocabulary teacher returned an invalid batch size")
        output: list[list[float]] = []
        for row_idx, (ids, values) in enumerate(zip(rows, log_probs, strict=True)):
            if not isinstance(values, list) or len(values) != len(ids):
                raise RuntimeError(f"Cross-vocabulary teacher length mismatch for row {row_idx}")
            cast_values = [float(value) for value in values]
            if not all(math.isfinite(value) for value in cast_values[1:]):
                raise RuntimeError(f"Cross-vocabulary teacher returned non-finite logprobs for row {row_idx}")
            output.append(cast_values)
        return output


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def retokenize_student_sequence(
    *,
    student_full_ids: Sequence[int],
    student_prefix_ids: Sequence[int],
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    max_sequence_length: int,
    generation_tail: str | None = None,
) -> tuple[list[int], int, bool]:
    """Retokenize the serialized student sequence with the source OPD marker mapping."""

    student_full_ids = [int(token_id) for token_id in student_full_ids]
    student_prefix_ids = [int(token_id) for token_id in student_prefix_ids]
    if student_full_ids[: len(student_prefix_ids)] != student_prefix_ids:
        raise ValueError("Student input_ids do not start with the exact serialized prompt token IDs")
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    mapping = build_chat_template_mapping(student_tokenizer, teacher_tokenizer)
    if generation_tail is None:
        generation_tail = generation_prefix_tail(student_tokenizer)
    student_full_text = _decode_serialized_sequence(student_tokenizer, student_full_ids)
    if "<reference_solution>" in student_full_text:
        raise ValueError("Cross-vocabulary OPD student sequence must not contain privileged PI traces")

    teacher_full_text = _apply_chat_template_mapping(student_full_text, mapping)
    if _detect_tokenizer_family(student_tokenizer) == "llama" and student_full_text.endswith(
        ("<|eot_id|>", "<|end_of_text|>")
    ):
        teacher_full_text = teacher_full_text.rstrip("\n")
    teacher_full_ids = _encode_text(teacher_tokenizer, teacher_full_text)
    was_truncated = len(teacher_full_ids) > max_sequence_length
    if was_truncated:
        teacher_full_ids = teacher_full_ids[:max_sequence_length]
    # The teacher scores the mapped student text, so its marker is the mapped student one.
    response_start = _find_response_start(
        teacher_tokenizer, teacher_full_ids, _apply_chat_template_mapping(generation_tail, mapping)
    )
    if response_start is None:
        response_start = len(teacher_full_ids)
    return teacher_full_ids, response_start, was_truncated


def _terminal_token_ids(tokenizer: Any) -> set[int]:
    terminal_ids = {
        token_id
        for token_id in (getattr(tokenizer, "eos_token_id", None),)
        if isinstance(token_id, int) and token_id >= 0
    }
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    for token in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id != unknown_id:
            terminal_ids.add(token_id)
    return terminal_ids


class CrossVocabOPDScorer:
    """Retokenize full student sequences, request scores, and align response tokens."""

    def __init__(
        self,
        *,
        student_tokenizer: Any,
        teacher_model_path: str,
        endpoint: str,
        max_sequence_length: int,
        large_chunk_threshold: int = 6,
        timeout_seconds: float = 600.0,
        teacher_tokenizer: Any | None = None,
        rpc_client: Any | None = None,
        student_thinking: bool | None = None,
    ) -> None:
        if max_sequence_length <= 0:
            raise ValueError("cross_vocab_opd_teacher_max_sequence_length must be positive")
        if teacher_tokenizer is None:
            from transformers import AutoTokenizer

            teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, trust_remote_code=True)
        if student_tokenizer.get_vocab() == teacher_tokenizer.get_vocab():
            raise ValueError("Cross-vocabulary OPD requires different student and teacher tokenizers")

        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.generation_tail = generation_prefix_tail(student_tokenizer, student_thinking)
        self.teacher_generation_tail = _apply_chat_template_mapping(
            self.generation_tail, build_chat_template_mapping(student_tokenizer, teacher_tokenizer)
        )
        self.max_sequence_length = int(max_sequence_length)
        self.large_chunk_threshold = int(large_chunk_threshold)
        self.rpc_client = rpc_client or CrossVocabTeacherClient(endpoint, timeout_seconds=timeout_seconds)
        self.teacher_tokenizer_fingerprint = tokenizer_fingerprint(teacher_tokenizer)
        self.rpc_client.health(expected_tokenizer_fingerprint=self.teacher_tokenizer_fingerprint)

    def score_batch(self, source_batch: Any) -> CrossVocabScoreBatch:
        for key in ("prompts", "responses", "input_ids", "attention_mask", "response_mask"):
            if key not in source_batch.batch:
                raise ValueError(f"Cross-vocabulary OPD source batch is missing {key!r}")

        prompts = source_batch.batch["prompts"].detach().cpu().long()
        responses = source_batch.batch["responses"].detach().cpu().long()
        input_ids = source_batch.batch["input_ids"].detach().cpu().long()
        attention_mask = source_batch.batch["attention_mask"].detach().cpu().bool()
        response_mask = source_batch.batch["response_mask"].detach().cpu().bool()
        prompt_width = prompts.shape[1]
        if attention_mask.shape != input_ids.shape or attention_mask.shape[1] < prompt_width:
            raise ValueError("Cross-vocabulary OPD received inconsistent input_ids and attention_mask shapes")

        teacher_terminal_ids = _terminal_token_ids(self.teacher_tokenizer)
        request_rows: list[list[int]] = []
        row_metadata: list[_RetokenizedRow] = []
        prompt_tokens_total = 0
        teacher_response_tokens_total = 0
        truncated_sequences = 0
        exact_boundary_rows = 0
        for row_idx in range(responses.shape[0]):
            valid_positions = response_mask[row_idx].nonzero(as_tuple=True)[0]
            expected_positions = torch.arange(valid_positions.numel())
            if not torch.equal(valid_positions, expected_positions):
                raise ValueError("Cross-vocabulary OPD requires right-padded contiguous response masks")
            student_response_ids = responses[row_idx][response_mask[row_idx]].tolist()

            prefix_mask = attention_mask[row_idx, :prompt_width]
            student_prefix_ids = prompts[row_idx][prefix_mask].tolist()
            student_full_ids = input_ids[row_idx][attention_mask[row_idx]].tolist()
            expected_full_ids = student_prefix_ids + student_response_ids
            if student_full_ids != expected_full_ids:
                raise ValueError(
                    "Cross-vocabulary OPD requires input_ids to contain the exact prompt and response IDs"
                )
            student_response_start = _find_response_start(
                self.student_tokenizer,
                student_full_ids,
                self.generation_tail,
                prompt_length=len(student_prefix_ids),
            )
            if student_response_start == len(student_prefix_ids):
                exact_boundary_rows += 1
            elif student_response_start is not None:
                # In the source implementation "marker found" and "response starts here" are
                # the same thing, so this is an invariant violation, not a tolerated anomaly:
                # the marker did not cover the whole generation prefix and every teacher log
                # probability for this row would be written at the wrong response offset.
                # Failing here costs one step; the shifted objective would cost a whole run.
                raise ValueError(
                    f"Cross-vocabulary OPD located the student response at token {student_response_start}, "
                    f"but the prompt is {len(student_prefix_ids)} tokens. Marker "
                    f"{_response_marker(self.student_tokenizer, self.generation_tail)!r} does not cover the "
                    "whole generation prefix. Check the chat template and the thinking mode the run renders "
                    "prompts with."
                )
            if student_response_start is None:
                warnings.warn(
                    f"Cross-vocabulary OPD cannot find the student response marker for row {row_idx}; "
                    "the source implementation assigns zero gradient to this row",
                    stacklevel=2,
                )
                student_content_ids = []
                student_special_count = 0
            else:
                student_content_end = len(student_full_ids)
                student_terminal_ids = _terminal_token_ids(self.student_tokenizer)
                while (
                    student_content_end > student_response_start
                    and student_full_ids[student_content_end - 1] in student_terminal_ids
                ):
                    student_content_end -= 1
                student_content_ids = student_full_ids[student_response_start:student_content_end]
                student_special_count = len(student_full_ids) - student_content_end
            teacher_full_ids, teacher_prefix_length, was_truncated = retokenize_student_sequence(
                student_full_ids=student_full_ids,
                student_prefix_ids=student_prefix_ids,
                student_tokenizer=self.student_tokenizer,
                teacher_tokenizer=self.teacher_tokenizer,
                max_sequence_length=self.max_sequence_length,
                generation_tail=self.generation_tail,
            )
            teacher_marker_found = (
                _find_response_start(self.teacher_tokenizer, teacher_full_ids, self.teacher_generation_tail)
                is not None
            )
            if student_response_start is not None and not teacher_marker_found:
                warnings.warn(
                    f"Cross-vocabulary OPD cannot find the teacher response marker for row {row_idx}; "
                    "the source implementation assigns zero gradient to this row",
                    stacklevel=2,
                )
            if len(teacher_full_ids) < 2:
                raise ValueError("Cross-vocabulary teacher sequence is too short to score")
            teacher_response_ids = teacher_full_ids[teacher_prefix_length:]
            teacher_content_end = len(teacher_response_ids)
            while teacher_content_end > 0 and teacher_response_ids[teacher_content_end - 1] in teacher_terminal_ids:
                teacher_content_end -= 1
            teacher_content_ids = teacher_response_ids[:teacher_content_end]
            teacher_terminal_position = (
                teacher_prefix_length + teacher_content_end
                if teacher_content_end < len(teacher_response_ids)
                else None
            )
            request_rows.append(teacher_full_ids)
            row_metadata.append(
                _RetokenizedRow(
                    student_content_ids=student_content_ids,
                    teacher_content_ids=teacher_content_ids,
                    teacher_prefix_length=teacher_prefix_length,
                    student_special_count=student_special_count,
                    teacher_terminal_position=teacher_terminal_position,
                    skip_alignment=student_response_start is None or not teacher_marker_found,
                )
            )
            prompt_tokens_total += teacher_prefix_length
            teacher_response_tokens_total += len(teacher_response_ids)
            truncated_sequences += int(was_truncated)

        score_start = time.perf_counter()
        scored_rows = self.rpc_client.score(request_rows)
        score_seconds = time.perf_counter() - score_start

        aligned_log_probs = torch.zeros_like(responses, dtype=torch.float32)
        aligned_chunk_ids = torch.full_like(responses, -1, dtype=torch.long)
        chunk_count = 0
        large_chunks = 0
        skipped_sequences = 0
        for row_idx, (full_log_probs, metadata) in enumerate(zip(scored_rows, row_metadata, strict=True)):
            if metadata.skip_alignment:
                continue
            response_end = metadata.teacher_prefix_length + len(metadata.teacher_content_ids)
            teacher_response_log_probs = torch.tensor(
                full_log_probs[metadata.teacher_prefix_length:response_end], dtype=torch.float32
            )
            alignment = align_cross_vocab_response(
                student_ids=metadata.student_content_ids,
                teacher_ids=metadata.teacher_content_ids,
                teacher_log_probs=teacher_response_log_probs,
                student_tokenizer=self.student_tokenizer,
                teacher_tokenizer=self.teacher_tokenizer,
                large_chunk_threshold=self.large_chunk_threshold,
            )
            if alignment.teacher_truncated:
                aligned_log_probs[row_idx, :] = float("inf")
                aligned_chunk_ids[row_idx, :] = -1
                skipped_sequences += 1
                large_chunks += alignment.large_chunk_count
                warnings.warn(
                    f"Cross-vocabulary OPD teacher tokens were exhausted for row {row_idx}; "
                    "the source implementation assigns zero gradient to the whole row",
                    stacklevel=2,
                )
                continue
            content_length = len(metadata.student_content_ids)
            aligned_log_probs[row_idx, :content_length] = alignment.log_probs
            aligned_chunk_ids[row_idx, :content_length] = alignment.chunk_ids
            if metadata.student_special_count > 0:
                first_special_position = content_length
                if metadata.teacher_terminal_position is None:
                    aligned_log_probs[row_idx, first_special_position] = float("inf")
                else:
                    aligned_log_probs[row_idx, first_special_position] = float(
                        full_log_probs[metadata.teacher_terminal_position]
                    )
                    aligned_chunk_ids[row_idx, first_special_position] = (
                        int(alignment.chunk_ids.max().item()) + 1 if len(alignment.chunk_ids) else 0
                    )
                if metadata.student_special_count > 1:
                    special_end = first_special_position + metadata.student_special_count
                    aligned_log_probs[row_idx, first_special_position + 1 : special_end] = float("inf")
                    aligned_chunk_ids[row_idx, first_special_position + 1 : special_end] = -1
            chunk_count += alignment.chunk_count
            large_chunks += alignment.large_chunk_count

        response_tokens = int(response_mask.sum().item())
        matched_tokens = int(((aligned_chunk_ids >= 0) & response_mask).sum().item())
        invalid_tokens = int(((aligned_chunk_ids < 0) & response_mask).sum().item())
        metrics = {
            "cross_vocab_opd/teacher_prompt_tokens": float(prompt_tokens_total),
            "cross_vocab_opd/teacher_response_tokens": float(teacher_response_tokens_total),
            "cross_vocab_opd/aligned_tokens": float(matched_tokens),
            "cross_vocab_opd/invalid_tokens": float(invalid_tokens),
            "cross_vocab_opd/aligned_token_ratio": float(matched_tokens / max(response_tokens, 1)),
            "cross_vocab_opd/chunk_count": float(chunk_count),
            "cross_vocab_opd/large_chunk_count": float(large_chunks),
            "cross_vocab_opd/teacher_sequence_truncated_count": float(truncated_sequences),
            "cross_vocab_opd/teacher_sequence_skipped_count": float(skipped_sequences),
            "cross_vocab_opd/teacher_score_seconds": float(score_seconds),
            # Below 1.0 only for rows whose marker was not found at all, which the source
            # implementation zeroes rather than raising. A marker found in the wrong place
            # raises above, because it shifts every aligned teacher log probability while
            # leaving aligned_token_ratio at ~1.0 -- both sides shift together.
            "cross_vocab_opd/student_boundary_exact_ratio": float(
                exact_boundary_rows / max(responses.shape[0], 1)
            ),
        }
        return CrossVocabScoreBatch(aligned_log_probs, aligned_chunk_ids, metrics)
