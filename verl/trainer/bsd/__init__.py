"""Bridged Self-Distillation helpers.

The ``bsd`` module path is retained for checkpoint and config compatibility.

The weighting math, prompt assembly, exact-token scoring helpers, and manifest
collection live in separate modules of this package so that each can be used
and checked without the trainer.
"""

from .rlsd import RLSDConfig, compute_opd_pg_advantages, compute_rlsd_advantages
from .scorer import (
    TeacherScoreOutput,
    TeacherScorer,
    TeacherScoreRequest,
    assert_left_padded_prefix_mask,
    assert_tokenizer_compatible,
    build_causal_score_inputs,
    gather_response_log_probs,
)

__all__ = [
    "RLSDConfig",
    "TeacherScorer",
    "TeacherScoreRequest",
    "TeacherScoreOutput",
    "assert_left_padded_prefix_mask",
    "assert_tokenizer_compatible",
    "build_causal_score_inputs",
    "compute_opd_pg_advantages",
    "compute_rlsd_advantages",
    "gather_response_log_probs",
]
