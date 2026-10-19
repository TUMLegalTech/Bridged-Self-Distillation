# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
from tensordict import TensorDict

from verl.trainer.bsd.cross_vocab_opd import compute_cross_vocab_opd_advantages
from verl.trainer.bsd.rlsd import RLSDConfig, compute_opd_pg_advantages, compute_rlsd_advantages
from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    rlsd_enabled = bool(config.policy_loss.get("rlsd_enabled", False))
    opd_pg_enabled = bool(config.policy_loss.get("opd_pg_enabled", False))
    cross_vocab_opd_enabled = bool(config.policy_loss.get("cross_vocab_opd_enabled", False))
    if sum((rlsd_enabled, opd_pg_enabled, cross_vocab_opd_enabled)) > 1:
        raise ValueError("RLSD, sampled-token OPD-PG, and cross-vocabulary OPD are mutually exclusive")
    if rlsd_enabled:
        if "teacher_pi_log_probs" not in data:
            raise ValueError("actor.policy_loss.rlsd_enabled=True requires teacher_pi_log_probs in the update batch")
        fields.append("teacher_pi_log_probs")
        if "rlsd_token_mask" in data:
            fields.append("rlsd_token_mask")
    if opd_pg_enabled:
        if "teacher_opd_log_probs" not in data:
            raise ValueError("actor.policy_loss.opd_pg_enabled=True requires teacher_opd_log_probs in the update batch")
        fields.append("teacher_opd_log_probs")
    if cross_vocab_opd_enabled:
        loss_mode = config.policy_loss.get("loss_mode", "vanilla")
        if loss_mode != "opd":
            raise ValueError(
                "actor.policy_loss.cross_vocab_opd_enabled=True requires "
                "actor.policy_loss.loss_mode=opd"
            )
        required = ("cross_vocab_teacher_log_probs", "cross_vocab_chunk_ids")
        missing = [field for field in required if field not in data]
        if missing:
            raise ValueError(
                "actor.policy_loss.cross_vocab_opd_enabled=True requires aligned teacher fields: "
                + ", ".join(missing)
            )
        fields.extend(required)
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    rlsd_effective_lambda = tu.get_non_tensor_data(
        data=data,
        key="rlsd_effective_lambda",
        default=config.policy_loss.get("rlsd_lambda", 0.5),
    )
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    if rlsd_enabled:
        rlsd_delta_source = str(config.policy_loss.get("rlsd_delta_student_logp_source", "current")).lower()
        if rlsd_delta_source == "current":
            rlsd_student_log_probs = log_prob.detach()
        elif rlsd_delta_source == "old":
            rlsd_student_log_probs = old_log_prob
        else:
            raise ValueError(
                "actor.policy_loss.rlsd_delta_student_logp_source must be 'current' or 'old', "
                f"got {rlsd_delta_source!r}"
            )
        advantages, rlsd_metrics = compute_rlsd_advantages(
            teacher_log_probs=data["teacher_pi_log_probs"],
            student_log_probs=rlsd_student_log_probs,
            advantages=advantages,
            response_mask=response_mask,
            rlsd_token_mask=data.get("rlsd_token_mask", None),
            config=RLSDConfig(
                lam=float(rlsd_effective_lambda),
                clip_range=float(config.policy_loss.get("rlsd_reweight_clip_range", 0.2)),
                negative_only=bool(config.policy_loss.get("rlsd_negative_only", True)),
                positive_mode=str(config.policy_loss.get("rlsd_positive_mode", "rlsd")),
                negative_mode=str(config.policy_loss.get("rlsd_negative_mode", "rlsd")),
            ),
        )
        metrics.update(Metric.from_dict(rlsd_metrics, aggregation=AggregationType.MEAN))
    if opd_pg_enabled:
        advantages, opd_metrics = compute_opd_pg_advantages(
            teacher_log_probs=data["teacher_opd_log_probs"],
            student_old_log_probs=old_log_prob,
            response_mask=response_mask,
            logratio_clip_abs=float(config.policy_loss.get("opd_logratio_clip_abs", 5.0)),
        )
        metrics.update(Metric.from_dict(opd_metrics, aggregation=AggregationType.MEAN))
    if cross_vocab_opd_enabled:
        advantages, cross_vocab_metrics = compute_cross_vocab_opd_advantages(
            teacher_log_probs=data["cross_vocab_teacher_log_probs"],
            student_old_log_probs=old_log_prob,
            chunk_ids=data["cross_vocab_chunk_ids"],
            response_mask=response_mask,
            loss_max_clamp=config.policy_loss.get("cross_vocab_opd_loss_max_clamp", 10.0),
        )
        metrics.update(Metric.from_dict(cross_vocab_metrics, aggregation=AggregationType.MEAN))
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
