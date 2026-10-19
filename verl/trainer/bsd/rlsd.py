"""Token-level Bridged Self-Distillation and sampled-token OPD-PG math."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from verl.utils.metric.teacher_student import compute_rollout_teacher_student_metrics


VALID_POSITIVE_MODES = ("off", "rlsd", "rlrt")
VALID_NEGATIVE_MODES = ("off", "rlsd")


@dataclass(frozen=True)
class RLSDConfig:
    """Configuration for RLSD advantage reweighting.

    ``positive_mode`` selects the weighting applied to positive-advantage
    sequences: ``off`` leaves them unweighted, ``rlsd`` follows the RLSD paper
    (w = P_T/P_S: concentrate credit on teacher-supported tokens), and
    ``rlrt`` reverses it per RLRT (w = P_S/P_T: amplify self-driven tokens
    the teacher disfavored on correct rollouts). ``negative_mode`` selects
    the wrong-rollout branch: ``rlsd`` (w = P_S/P_T: heavier blame on
    teacher-disfavored tokens) or ``off``. ``negative_only=True`` forces
    ``positive_mode='off'``: successful rollouts (A > 0) keep their advantage
    and only unsuccessful rollouts (A < 0) are weighted. This is the BSD
    setting and the default; a variant that also weights successful rollouts
    must set ``negative_only=False`` explicitly.
    """

    lam: float = 0.5
    clip_range: float = 0.2
    negative_only: bool = True
    positive_mode: str = "rlsd"
    negative_mode: str = "rlsd"

    def validate(self) -> None:
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {self.lam}")
        if not 0.0 <= self.clip_range < 1.0:
            raise ValueError(f"clip_range must be in [0, 1), got {self.clip_range}")
        if self.positive_mode not in VALID_POSITIVE_MODES:
            raise ValueError(f"positive_mode must be one of {VALID_POSITIVE_MODES}, got {self.positive_mode!r}")
        if self.negative_mode not in VALID_NEGATIVE_MODES:
            raise ValueError(f"negative_mode must be one of {VALID_NEGATIVE_MODES}, got {self.negative_mode!r}")
        if self.positive_mode == "off" and self.negative_mode == "off":
            raise ValueError("positive_mode and negative_mode cannot both be 'off'")

    @property
    def effective_positive_mode(self) -> str:
        return "off" if self.negative_only else self.positive_mode


def _as_mask(response_mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return response_mask.to(device=like.device, dtype=like.dtype)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype, device=value.device)
    denom = mask.sum().clamp_min(1.0)
    masked_value = torch.where(mask.bool(), value, torch.zeros_like(value))
    return masked_value.sum() / denom


def _masked_std(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = _masked_mean(value, mask)
    return torch.sqrt(_masked_mean((value - mean) ** 2, mask).clamp_min(0.0))


def compute_rlsd_advantages(
    *,
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    rlsd_token_mask: torch.Tensor | None = None,
    config: RLSDConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reweight GRPO advantages using privileged teacher evidence.

    Formula under the default routing (``negative_only=True``, the BSD setting):
        delta_t = stopgrad(teacher_pi_logp_t - student_logp_t)
        unsuccessful rollout (A < 0):
            weight_t = clamp(exp(-delta_t), 1 - eps, 1 + eps)
            A'_t = A_t * ((1 - lambda) + lambda * weight_t)
        successful rollout (A > 0):
            A'_t = A_t  (the weight is 1)

    Optional two-sided variant (``negative_only=False``): successful rollouts
    are weighted in the same way with weight_t = clamp(exp(s * delta_t),
    1 - eps, 1 + eps), where s = +1 for ``positive_mode='rlsd'`` and s = -1
    for ``positive_mode='rlrt'``. Tokens outside ``rlsd_token_mask`` keep
    weight 1 in every case.

    The output preserves the sign of every non-zero valid advantage by
    construction because the multiplicative factor is strictly positive.
    """

    config.validate()
    if not (teacher_log_probs.shape == student_log_probs.shape == advantages.shape == response_mask.shape):
        raise ValueError(
            "teacher_log_probs, student_log_probs, advantages, and response_mask must share the same shape"
        )
    if rlsd_token_mask is not None and rlsd_token_mask.shape != response_mask.shape:
        raise ValueError("rlsd_token_mask must share the same shape as response_mask")

    mask = _as_mask(response_mask, advantages)
    credit_mask = mask if rlsd_token_mask is None else _as_mask(rlsd_token_mask, advantages) * mask
    delta = teacher_log_probs.detach() - student_log_probs.detach()
    # Branch signs are gated on the sequence-level advantage sign. For GRPO the
    # token advantage is uniform within a response, so this equals gating on
    # the per-token sign; sequence gating keeps the branch selection
    # well-defined if a token-level advantage variant is ever plugged in.
    seq_adv = (advantages.detach() * mask).sum(dim=-1, keepdim=True)
    pos_sign = {"off": 0.0, "rlsd": 1.0, "rlrt": -1.0}[config.effective_positive_mode]
    neg_sign = {"off": 0.0, "rlsd": -1.0}[config.negative_mode]
    branch_sign = pos_sign * (seq_adv > 0).to(advantages.dtype) + neg_sign * (seq_adv < 0).to(advantages.dtype)
    signed_delta = (branch_sign * delta).detach()
    valid = mask.bool()
    if valid.any():
        for name, tensor in {
            "teacher_log_probs": teacher_log_probs.detach(),
            "student_log_probs": student_log_probs.detach(),
            "advantages": advantages.detach(),
            "signed_delta": signed_delta,
        }.items():
            if not torch.isfinite(tensor[valid]).all():
                bad = (~torch.isfinite(tensor[valid])).sum().item()
                raise ValueError(f"RLSD found non-finite {name} on {bad} valid tokens")

    log_low = math.log(1.0 - config.clip_range)
    log_high = math.log(1.0 + config.clip_range)
    clipped_log_weight = signed_delta.clamp(min=log_low, max=log_high)
    clipped_weights = torch.exp(clipped_log_weight)
    rlsd_reweight = (1.0 - config.lam) + config.lam * clipped_weights
    reweight = torch.where(credit_mask.bool(), rlsd_reweight, torch.ones_like(rlsd_reweight)) * mask

    reweighted_advantages = advantages * reweight.detach() * mask
    valid_nonzero = (mask > 0) & (advantages != 0)
    sign_flips = ((torch.sign(reweighted_advantages) != torch.sign(advantages)) & valid_nonzero).sum()

    metrics = {
        "rlsd/delta_mean": _masked_mean(delta, mask).detach().item(),
        "rlsd/delta_std": _masked_std(delta, mask).detach().item(),
        "rlsd/log_weight_mean": _masked_mean(signed_delta, credit_mask).detach().item(),
        "rlsd/log_weight_std": _masked_std(signed_delta, credit_mask).detach().item(),
        "rlsd/weight_mean": _masked_mean(clipped_weights, credit_mask).detach().item(),
        "rlsd/weight_std": _masked_std(clipped_weights, credit_mask).detach().item(),
        "rlsd/clip_low_ratio": _masked_mean(
            (signed_delta < log_low).to(dtype=advantages.dtype), credit_mask
        ).detach().item(),
        "rlsd/clip_high_ratio": _masked_mean(
            (signed_delta > log_high).to(dtype=advantages.dtype), credit_mask
        ).detach().item(),
        "rlsd/token_mask_ratio": (credit_mask.sum() / mask.sum().clamp_min(1.0)).detach().item(),
        "rlsd/effective_lambda": config.lam,
        "rlsd/adv_abs_mean_before": _masked_mean(advantages.detach().abs(), mask).detach().item(),
        "rlsd/adv_abs_mean_after": _masked_mean(reweighted_advantages.detach().abs(), mask).detach().item(),
        # |A|-weighted mean modulation |m-1|: the actual credit-redistribution
        # strength (net |A| change cancels amplified vs protected tokens).
        "rlsd/modulation_strength": (
            (advantages.detach().abs() * (reweight.detach() - mask).abs() * mask).sum()
            / (advantages.detach().abs() * mask).sum().clamp_min(1e-8)
        ).detach().item(),
        "rlsd/modulation_strength_pos": (
            (advantages.detach().abs() * (reweight.detach() - mask).abs() * mask * (seq_adv > 0)).sum()
            / (advantages.detach().abs() * mask * (seq_adv > 0)).sum().clamp_min(1e-8)
        ).detach().item(),
        "rlsd/modulation_strength_neg": (
            (advantages.detach().abs() * (reweight.detach() - mask).abs() * mask * (seq_adv < 0)).sum()
            / (advantages.detach().abs() * mask * (seq_adv < 0)).sum().clamp_min(1e-8)
        ).detach().item(),
        "rlsd/sign_flip_count": int(sign_flips.detach().item()),
        **compute_rollout_teacher_student_metrics(
            teacher_log_probs=teacher_log_probs,
            student_log_probs=student_log_probs,
            response_mask=mask,
        ),
    }

    pos_mask = mask * (advantages > 0).to(dtype=advantages.dtype)
    neg_mask = mask * (advantages < 0).to(dtype=advantages.dtype)
    metrics["rlsd/delta_pos_adv_mean"] = _masked_mean(delta, pos_mask).detach().item()
    metrics["rlsd/delta_neg_adv_mean"] = _masked_mean(delta, neg_mask).detach().item()
    return reweighted_advantages, metrics


def compute_opd_pg_advantages(
    *,
    teacher_log_probs: torch.Tensor,
    student_old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    logratio_clip_abs: float = 5.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build sampled-token OPD-PG/K1 advantages from teacher and old-policy logprobs."""

    if teacher_log_probs.shape != student_old_log_probs.shape or teacher_log_probs.shape != response_mask.shape:
        raise ValueError("teacher_log_probs, student_old_log_probs, and response_mask must share the same shape")
    if logratio_clip_abs <= 0:
        raise ValueError("logratio_clip_abs must be positive")

    mask = _as_mask(response_mask, teacher_log_probs)
    raw_logratio = (teacher_log_probs.detach() - student_old_log_probs.detach()) * mask
    token_advantages = raw_logratio.clamp(min=-logratio_clip_abs, max=logratio_clip_abs) * mask
    metrics = {
        "opd/logratio_mean": _masked_mean(raw_logratio, mask).detach().item(),
        "opd/logratio_abs_mean": _masked_mean(raw_logratio.abs(), mask).detach().item(),
        "opd/logratio_p99": torch.quantile(raw_logratio[mask.bool()].abs(), 0.99).detach().item()
        if mask.bool().any()
        else 0.0,
        "opd/logratio_clamp_ratio": _masked_mean(
            (raw_logratio.abs() > logratio_clip_abs).to(dtype=teacher_log_probs.dtype), mask
        ).detach().item(),
        "opd/teacher_logp_mean": _masked_mean(teacher_log_probs.detach(), mask).detach().item(),
        "opd/student_old_logp_mean": _masked_mean(student_old_log_probs.detach(), mask).detach().item(),
        **compute_rollout_teacher_student_metrics(
            teacher_log_probs=teacher_log_probs,
            student_log_probs=student_old_log_probs,
            response_mask=mask,
        ),
    }
    return token_advantages, metrics


def compute_truncation_aware_grpo_advantages(
    *,
    scores: torch.Tensor,
    index: Any,
    completed: torch.Tensor,
    response_mask: torch.Tensor,
    norm_adv_by_std: bool = True,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """GRPO outcome advantages whose group statistics use completed responses only.

    Under a response-length cap, truncated responses score 0, so the standard group
    statistics turn the verifier advantage into a length signal. Here a truncated
    (censored) response gets advantage 0 and is excluded from its group's mean and std;
    a group with fewer than two completed responses gets 0 for every member. Completed
    rows follow upstream GRPO (``core_algos.compute_grpo_outcome_advantage``): unbiased
    std, ``(s - mean) / (std + epsilon)``.
    """

    if scores.dim() != 1 or completed.shape != scores.shape or response_mask.shape[0] != scores.shape[0]:
        raise ValueError("scores and completed must be 1-D with one entry per response_mask row")
    if len(index) != scores.shape[0]:
        raise ValueError("index must have one group id per response")
    if not torch.isfinite(scores).all():
        raise ValueError("truncation-aware GRPO received non-finite sequence scores")

    groups: dict[Any, list[int]] = {}
    for row, uid in enumerate(index):
        groups.setdefault(uid, []).append(row)
    seq_adv = torch.zeros_like(scores, dtype=torch.float32)
    active_groups = 0
    with torch.no_grad():
        for rows in groups.values():
            keep = [row for row in rows if bool(completed[row])]
            if len(keep) < 2:
                continue
            group_scores = scores[keep].float()
            centered = group_scores - group_scores.mean()
            if norm_adv_by_std:
                centered = centered / (group_scores.std() + epsilon)
            seq_adv[keep] = centered
            active_groups += int(bool((centered != 0).any()))
    mask = _as_mask(response_mask, seq_adv)
    advantages = seq_adv.unsqueeze(-1) * mask
    n_groups = max(len(groups), 1)
    metrics = {
        "grpo_ta/censored_ratio": float((~completed.bool()).float().mean().item()),
        "grpo_ta/groups_with_gradient_ratio": active_groups / n_groups,
        "grpo_ta/groups_under_two_completed_ratio": sum(
            sum(bool(completed[row]) for row in rows) < 2 for rows in groups.values()
        )
        / n_groups,
    }
    return advantages, metrics
