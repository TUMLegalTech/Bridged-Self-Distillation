"""Teacher/student diagnostics on sampled rollout tokens."""

from __future__ import annotations

import math

import torch


def compute_rollout_teacher_student_metrics(
    *,
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Compare teacher and student probabilities on valid rollout tokens.

    The signed gap is always teacher log-probability minus student
    log-probability. The geometric mean ratio is therefore teacher/student.
    """

    if not (teacher_log_probs.shape == student_log_probs.shape == response_mask.shape):
        raise ValueError("teacher_log_probs, student_log_probs, and response_mask must share the same shape")

    valid = response_mask.to(device=teacher_log_probs.device, dtype=torch.bool)
    if not valid.any():
        return {
            "teacher_student/rollout_token_logprob_gap_mean": 0.0,
            "teacher_student/rollout_token_logprob_gap_abs_mean": 0.0,
            "teacher_student/rollout_token_logprob_gap_std": 0.0,
            "teacher_student/rollout_token_probability_gap_abs_mean": 0.0,
            "teacher_student/rollout_token_probability_ratio_geomean": 1.0,
            "teacher_student/rollout_token_teacher_higher_probability_fraction": 0.0,
        }

    gap = teacher_log_probs.detach() - student_log_probs.detach()
    valid_gap = gap[valid].float()
    teacher_probabilities = teacher_log_probs.detach()[valid].float().exp()
    student_probabilities = student_log_probs.detach()[valid].float().exp()
    gap_mean = valid_gap.mean()
    probability_ratio_geomean = math.exp(max(min(gap_mean.double().item(), 700.0), -700.0))

    return {
        "teacher_student/rollout_token_logprob_gap_mean": gap_mean.item(),
        "teacher_student/rollout_token_logprob_gap_abs_mean": valid_gap.abs().mean().item(),
        "teacher_student/rollout_token_logprob_gap_std": valid_gap.std(unbiased=False).item(),
        "teacher_student/rollout_token_probability_gap_abs_mean": (teacher_probabilities - student_probabilities)
        .abs()
        .mean()
        .item(),
        "teacher_student/rollout_token_probability_ratio_geomean": probability_ratio_geomean,
        "teacher_student/rollout_token_teacher_higher_probability_fraction": (valid_gap > 0).float().mean().item(),
    }
