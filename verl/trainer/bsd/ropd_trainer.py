"""Group-level trainer adapter for the vendored ROPD (rubric-based OPD) baseline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.trainer.bsd.judge_usage import UsageRecorder, attach_usage_recorder
from verl.trainer.bsd.ropd_vendor import activate_ropd_vendor
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def stable_uids_from_extra_info(extra_infos: np.ndarray) -> np.ndarray:
    """Extract stable dataset identifiers required for grouping and index lookup."""

    uids: list[str] = []
    for row_index, value in enumerate(extra_infos):
        if hasattr(value, "as_py"):
            value = value.as_py()
        if not isinstance(value, Mapping):
            raise ValueError(f"ROPD extra_info at row {row_index} must be a mapping.")
        uid = value.get("uid") or value.get("id")
        if uid is None or not str(uid).strip():
            raise ValueError(f"ROPD extra_info at row {row_index} must contain a non-empty uid or id.")
        uids.append(str(uid))
    if len(set(uids)) != len(uids):
        raise ValueError("ROPD requires one unique stable uid per prompt before rollout repetition.")
    return np.asarray(uids, dtype=object)


def ropd_result_to_dataproto(result: Mapping[str, Any]) -> DataProto:
    """Convert the official reward-manager result to the current verl protocol."""

    reward_tensor = result.get("reward_tensor")
    if not isinstance(reward_tensor, torch.Tensor):
        raise TypeError("ROPD reward manager must return reward_tensor as a torch.Tensor.")
    reward_extra_info = result.get("reward_extra_info", {})
    if not isinstance(reward_extra_info, Mapping):
        raise TypeError("ROPD reward_extra_info must be a mapping.")
    extra_keys = list(reward_extra_info)
    non_tensors = {
        key: np.asarray(values, dtype=object) for key, values in reward_extra_info.items()
    }
    return DataProto.from_dict(
        tensors={"rm_scores": reward_tensor},
        non_tensors=non_tensors,
        meta_info={"reward_extra_keys": extra_keys},
    )


class RopdRayPPOTrainer(RayPPOTrainer):
    """Run official ROPD scoring after all rollouts for a prompt are available."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        activate_ropd_vendor()
        from algo.ropd.reward_manager import RopdRewardManager

        ropd_config = OmegaConf.to_container(self.config.ropd, resolve=True)
        self.ropd_reward_manager = RopdRewardManager(
            tokenizer=self.tokenizer,
            num_examine=0,
            compute_score=None,
            ropd=ropd_config,
        )
        self._ropd_reward_control: Mapping[str, Any] = {}

        # The judges are the only part of ROPD that costs money, so their token usage
        # is recorded: without it a finished run leaves no way to tell what it spent, or
        # which steps were expensive. The reward manager builds its own clients, so attach
        # after construction. The teacher is deliberately skipped -- under the train
        # entrypoint it is an offline index and never calls out.
        self._ropd_usage = UsageRecorder()
        for stage, attribute in (("rubricator", "rubric_client"), ("verifier", "verifier_client")):
            client = getattr(self.ropd_reward_manager, attribute, None)
            for one in client if isinstance(client, (list, tuple)) else [client]:
                if one is not None:
                    attach_usage_recorder(one, self._ropd_usage, stage)
        self._ropd_usage_prev: dict[str, float] = {}

    def _ensure_batch_uids(self, batch: DataProto) -> None:
        if "uid" in batch.non_tensor_batch:
            return
        extra_infos = batch.non_tensor_batch.get("extra_info")
        if extra_infos is None:
            raise ValueError("ROPD requires dataset extra_info with a stable uid or id.")
        batch.non_tensor_batch["uid"] = stable_uids_from_extra_info(extra_infos)

    def _should_stream_reward_during_rollout(self) -> bool:
        return False

    def _compute_batch_reward(self, batch: DataProto, *, is_validation: bool = False) -> DataProto:
        if is_validation:
            return self.reward_loop_manager.compute_rm_score(batch)
        result = self.ropd_reward_manager(batch, return_dict=True)
        self._ropd_reward_control = result.get("reward_control", {})
        return ropd_result_to_dataproto(result)

    def _before_actor_update(
        self,
        batch: DataProto,
        metrics: dict[str, Any],
        timing_raw: dict[str, Any],
    ) -> DataProto:
        del timing_raw
        for source_key, metric_key in (
            ("student_score", "ropd/student_score_mean"),
            ("reward_score", "ropd/reward_score_mean"),
            ("judge_error", "ropd/judge_error_rate"),
            ("fallback_used", "ropd/fallback_rate"),
            ("group_size", "ropd/group_size_mean"),
        ):
            values = batch.non_tensor_batch.get(source_key)
            if values is not None and len(values):
                metrics[metric_key] = float(np.asarray(values, dtype=np.float64).mean())

        control_metrics = (
            ("total_group_count", "ropd/total_group_count"),
            ("effective_uid_count", "ropd/effective_group_count"),
            ("effective_group_rate", "ropd/effective_group_rate"),
            ("teacher_below_student_group_rate", "ropd/teacher_below_student_group_rate"),
        )
        for source_key, metric_key in control_metrics:
            value = self._ropd_reward_control.get(source_key)
            if value is not None:
                metrics[metric_key] = float(value)

        # Both the running total and this step's delta. The delta is the actionable one:
        # it shows a step that suddenly costs more, which is how an untrimmed judge input
        # or a retry storm announces itself. cost is only populated by gateways that
        # price the call for us; on OpenAI itself the token counters carry the answer.
        usage = self._ropd_usage.snapshot().get("total", {})
        for field, metric_key in (
            ("calls", "ropd/judge_calls"),
            ("prompt_tokens", "ropd/judge_prompt_tokens"),
            ("completion_tokens", "ropd/judge_completion_tokens"),
            ("reasoning_tokens", "ropd/judge_reasoning_tokens"),
            ("cached_tokens", "ropd/judge_cached_tokens"),
            ("cost", "ropd/judge_cost_usd"),
        ):
            total = usage.get(field)
            if total is None:
                continue
            metrics[f"{metric_key}_total"] = float(total)
            metrics[f"{metric_key}_step"] = float(total - self._ropd_usage_prev.get(field, 0.0))
        self._ropd_usage_prev = dict(usage)
        return batch
