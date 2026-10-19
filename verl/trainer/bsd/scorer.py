"""Exact-token teacher scoring helpers for Bridged Self-Distillation."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Protocol

import torch

from verl.utils.attention_utils import default_attn_implementation


@dataclass
class TeacherScoreRequest:
    """Request to score exact student response token IDs under a teacher prefix."""

    prompt_input_ids: torch.LongTensor
    prompt_attention_mask: torch.Tensor
    response_ids: torch.LongTensor
    response_mask: torch.Tensor
    temperature: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TeacherScoreOutput:
    """Response-token log probabilities and scorer diagnostics."""

    response_log_probs: torch.Tensor
    response_mask: torch.Tensor
    metrics: dict[str, Any] = field(default_factory=dict)


class TeacherScorer(Protocol):
    """Teacher scoring protocol shared by external and self-PEFT backends."""

    def score(self, request: TeacherScoreRequest) -> TeacherScoreOutput:
        """Score response tokens from a request."""


def assert_left_padded_prefix_mask(prefix_attention_mask: torch.Tensor) -> None:
    """Fail unless every prefix row is left-padded with a real token in the last column."""

    if prefix_attention_mask.ndim != 2:
        raise ValueError("prefix_attention_mask must be a rank-2 tensor")
    mask = prefix_attention_mask.to(dtype=torch.bool)
    token_counts = mask.long().sum(dim=-1)
    if token_counts.eq(0).any():
        raise ValueError("Teacher scorer found an empty prompt prefix")
    if not mask[:, -1].all():
        raise ValueError(
            "Teacher scorer requires left-padded prompt prefixes with a real token at prefix_width-1. "
            "Found right-padded or non-contiguous prompt prefix."
        )

    width = mask.shape[1]
    col_idx = torch.arange(width, device=mask.device).view(1, -1)
    expected = col_idx >= (width - token_counts).view(-1, 1)
    if not torch.equal(mask, expected):
        raise ValueError(
            "Teacher scorer requires contiguous left-padded prompt prefixes. "
            "Found right-padded or non-contiguous prompt prefix."
        )


def build_causal_score_inputs(
    *,
    prefix_input_ids: torch.LongTensor,
    prefix_attention_mask: torch.Tensor,
    response_ids: torch.LongTensor,
    response_mask: torch.Tensor,
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Append original response IDs to a teacher prefix without retokenizing.

    The logits at position ``prefix_width - 1 + t`` predict response token ``t``.
    Prefix rows are expected to be left-padded, matching verl prompt handling.
    """

    if prefix_input_ids.ndim != 2 or response_ids.ndim != 2:
        raise ValueError("prefix_input_ids and response_ids must be rank-2 tensors")
    if prefix_input_ids.shape != prefix_attention_mask.shape:
        raise ValueError("prefix_input_ids and prefix_attention_mask shapes differ")
    if response_ids.shape != response_mask.shape:
        raise ValueError("response_ids and response_mask shapes differ")
    if prefix_input_ids.shape[0] != response_ids.shape[0]:
        raise ValueError("prefix and response batch sizes differ")
    assert_left_padded_prefix_mask(prefix_attention_mask)

    response_mask_bool = response_mask.to(device=response_ids.device, dtype=torch.bool)
    response_ids_for_input = response_ids.masked_fill(~response_mask_bool, pad_token_id)
    full_input_ids = torch.cat([prefix_input_ids, response_ids_for_input], dim=1)
    full_attention_mask = torch.cat([prefix_attention_mask.to(dtype=torch.bool), response_mask_bool], dim=1)
    position_ids = full_attention_mask.long().cumsum(dim=1) - 1
    position_ids = position_ids.clamp_min(0)

    prefix_width = prefix_input_ids.shape[1]
    response_len = response_ids.shape[1]
    response_logit_positions = torch.arange(
        prefix_width - 1, prefix_width + response_len - 1, device=response_ids.device, dtype=torch.long
    ).view(1, -1)
    response_logit_positions = response_logit_positions.expand(response_ids.shape[0], -1)
    return {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask,
        "position_ids": position_ids,
        "response_ids": response_ids,
        "response_mask": response_mask_bool,
        "response_logit_positions": response_logit_positions,
    }


def gather_response_log_probs(
    *,
    logits: torch.Tensor,
    response_ids: torch.LongTensor,
    response_logit_positions: torch.LongTensor,
    temperature: float,
) -> torch.Tensor:
    """Gather response-token logprobs from causal logits at explicit positions."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, seq, vocab]")
    if response_ids.shape != response_logit_positions.shape:
        raise ValueError("response_ids and response_logit_positions shapes differ")

    batch_idx = torch.arange(logits.shape[0], device=logits.device).view(-1, 1)
    selected_logits = logits[batch_idx, response_logit_positions.to(device=logits.device)]
    log_probs = torch.log_softmax(selected_logits / temperature, dim=-1)
    return log_probs.gather(dim=-1, index=response_ids.to(device=logits.device).unsqueeze(-1)).squeeze(-1)


def _hash_text(value: str | None) -> str | None:
    if value is None:
        return None
    return sha256(value.encode("utf-8")).hexdigest()


def assert_tokenizer_compatible(
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    sample_strings: list[str] | None = None,
) -> None:
    """Fail fast if tokenizers are unsafe for exact response-token reuse."""

    checks = [
        ("vocab_size", getattr(student_tokenizer, "vocab_size", None), getattr(teacher_tokenizer, "vocab_size", None)),
        ("bos_token_id", student_tokenizer.bos_token_id, teacher_tokenizer.bos_token_id),
        ("eos_token_id", student_tokenizer.eos_token_id, teacher_tokenizer.eos_token_id),
        ("pad_token_id", student_tokenizer.pad_token_id, teacher_tokenizer.pad_token_id),
        ("special_tokens_map", student_tokenizer.special_tokens_map, teacher_tokenizer.special_tokens_map),
        (
            "chat_template_hash",
            _hash_text(getattr(student_tokenizer, "chat_template", None)),
            _hash_text(getattr(teacher_tokenizer, "chat_template", None)),
        ),
    ]
    failures = [name for name, left, right in checks if left != right]
    if hasattr(student_tokenizer, "get_vocab") and hasattr(teacher_tokenizer, "get_vocab"):
        if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
            failures.append("full_vocab_mapping")
    for text in sample_strings or ["x", "\\boxed{1}", "Solve the problem.", "<|im_start|>user"]:
        if student_tokenizer.encode(text, add_special_tokens=False) != teacher_tokenizer.encode(
            text, add_special_tokens=False
        ):
            failures.append(f"token_mapping:{text!r}")
            break
    if failures:
        raise ValueError(f"Teacher tokenizer is not exact-token compatible: {', '.join(failures)}")


@contextmanager
def peft_adapter_disabled(model: Any):
    """Temporarily disable PEFT adapters when the model exposes a compatible API."""

    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
        return
    if hasattr(model, "disable_adapters") and hasattr(model, "enable_adapters"):
        model.disable_adapters()
        try:
            yield
        finally:
            model.enable_adapters()
        return
    yield


class InlineHFTeacherScorer:
    """No-grad Hugging Face teacher scorer for same-GPU sequential scoring."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        trust_remote_code: bool = True,
        attn_implementation: str | None = None,
        disable_adapter: bool = False,
    ) -> None:
        attn_implementation = attn_implementation or default_attn_implementation()
        if model is None:
            if model_path is None:
                raise ValueError("model_path is required when model is not provided")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
                use_cache=False,
            )
        self.model = model.to(device) if hasattr(model, "to") else model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.disable_adapter = disable_adapter
        self.model.eval()

    def score(self, request: TeacherScoreRequest) -> TeacherScoreOutput:
        tensors = build_causal_score_inputs(
            prefix_input_ids=request.prompt_input_ids.to(self.device),
            prefix_attention_mask=request.prompt_attention_mask.to(self.device),
            response_ids=request.response_ids.to(self.device),
            response_mask=request.response_mask.to(self.device),
            pad_token_id=getattr(self.tokenizer, "pad_token_id", 0) or 0,
        )
        adapter_ctx = peft_adapter_disabled(self.model) if self.disable_adapter else nullcontext()
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad(), adapter_ctx:
            output = self.model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
                position_ids=tensors["position_ids"],
                use_cache=False,
            )
            response_log_probs = gather_response_log_probs(
                logits=output.logits,
                response_ids=tensors["response_ids"],
                response_logit_positions=tensors["response_logit_positions"],
                temperature=request.temperature,
            )
        if was_training:
            self.model.train()
        return TeacherScoreOutput(
            response_log_probs=response_log_probs.detach(),
            response_mask=tensors["response_mask"],
            metrics={"teacher/tokens_scored": int(tensors["response_mask"].sum().detach().item())},
        )


class InlineSelfPeftScorer(InlineHFTeacherScorer):
    """Self-teacher scorer that disables LoRA adapters, so that the base model does the scoring."""

    def __init__(self, *, model: Any, tokenizer: Any, device: str | torch.device = "cuda") -> None:
        super().__init__(model=model, tokenizer=tokenizer, device=device, disable_adapter=True)
