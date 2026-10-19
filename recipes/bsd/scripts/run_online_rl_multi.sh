#!/usr/bin/env bash
# Shared online-RL launcher. It assembles the full hydra command -- dataset class,
# chat-template arguments, rollout, FSDP/LoRA actor, verifier reward, checkpointing,
# manifest, logging -- and ends with "$@" so a method wrapper can append its own
# overrides (hydra keeps the last one). Its defaults are GRPO (verifier reward only, no
# privileged context, no self-teacher) with the BSD budget: 16 problems x 8 rollouts per
# update, 150 updates, learning rate 1e-6, LoRA rank 32 / alpha 64.
#
# Consumers (multi-GPU):
#   run_xopd_multi.sh      xOPD (cross-tokenizer OPD)
#   run_opd_multi.sh                  vanilla OPD (shared tokenizer, teacher pool)
#   run_ropd_multi.sh       ROPD (rubric-based OPD)
#
# Do not add method-specific behaviour here. Put it in the method's own wrapper.
set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env_multi.sh"

TRAIN_FILE="${TRAIN_FILE:?TRAIN_FILE is required: path of the training parquet; see the README section \"Data preparation\"}"
VAL_FILE="${VAL_FILE:?VAL_FILE is required: path of the development parquet; see the README section \"Data preparation\"}"
MODEL_PATH="${MODEL_PATH:-$(bsd_model_path Qwen/Qwen3-4B)}"
PROJECT_NAME="${PROJECT_NAME:-bridged_self_distillation}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME is required: a unique run name; it names the checkpoint and output directories}"
METHOD_NAME="${METHOD_NAME:-grpo}"
# Optional JSON manifest of the data split; when set, its path and SHA-256 are recorded
# in the run manifest.
SPLIT_MANIFEST="${SPLIT_MANIFEST:-}"
PPO_ENTRYPOINT="${PPO_ENTRYPOINT:-verl.trainer.main_ppo}"
SCALE_MODE="${SCALE_MODE:-fixed}"
STUDENT_THINKING="${STUDENT_THINKING:-True}"
# Tri-state. Empty follows STUDENT_THINKING: with thinking on, the default, it selects the
# plain student prompt of the Qwen3 students. Qwen3-4B-Instruct sets
# EMIT_THINK_PROMPTS=True and keeps STUDENT_THINKING=True: its chat template opens the
# reasoning block, and the last line of its prompt is the instruction to reason inside
# the <think> tags (student_emit_think.jinja).
EMIT_THINK_PROMPTS="${EMIT_THINK_PROMPTS:-}"
SAVE_LORA_ONLY="${SAVE_LORA_ONLY:-True}"
# Score of a boxed-but-wrong answer. 0 (default) keeps the verifier reward binary,
# r in {0, 1}; a positive value adds a format term to the group advantages (see
# recipes/bsd/rewards/math_verify_reward.py).
export BSD_FORMAT_BONUS="${BSD_FORMAT_BONUS:-0}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-12288}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
SAVE_FREQ="${SAVE_FREQ:-25}"
# Periodic in-training validation on VAL_FILE, every TEST_FREQ updates and after the last
# one; TEST_FREQ=-1 disables it. It only monitors the run: checkpoints are selected
# afterwards on the development set with the evaluation protocol. Independently of
# TEST_FREQ, verl validates once before the first update unless
# trainer.val_before_train=False is appended as a hydra override.
TEST_FREQ="${TEST_FREQ:-25}"
SEED="${SEED:-43}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"
RESUME_MODE="${RESUME_MODE:-disable}"
# Read from variables so that the run manifest and the hydra command below always carry the
# same value: a method wrapper that needs another value (ROPD does) exports it instead of
# appending a hydra override, which would leave the manifest key at its default (the
# manifest records appended overrides only as one string, hydra_overrides).
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"
POLICY_CLIP_RATIO_HIGH="${POLICY_CLIP_RATIO_HIGH:-0.28}"
NORM_ADV_BY_STD_IN_GRPO="${NORM_ADV_BY_STD_IN_GRPO:-True}"
# Samples per validation problem. The default 4 samples at temperature 1.0 and reports
# mean@4, which is less noisy than a single greedy decode; VAL_ROLLOUT_N=1 validates
# greedily (mean@1, which is not the same quantity as mean@4).
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-4}"
# Sampled validation draws from the same vLLM random stream as the training rollouts
# (rollout.full_determinism is off, so no per-request seed is injected), whereas greedy
# decoding consumes no randomness. A run that validates therefore follows a different
# rollout sequence than the same configuration without validation. The draws come from the
# same distribution, so this is a re-randomisation and not a bias, but training trajectories
# are only comparable between runs that share the validation settings.
case "${VAL_ROLLOUT_N}" in
  ''|*[!0-9]*) echo "VAL_ROLLOUT_N must be a positive integer; got ${VAL_ROLLOUT_N}" >&2; exit 1 ;;
  0) echo "VAL_ROLLOUT_N must be >= 1" >&2; exit 1 ;;
esac
# n=1 decodes greedily; anything above it has to sample, or the n draws are identical and
# mean@N is just a slower mean@1.
if [ "${VAL_ROLLOUT_N}" -eq 1 ]; then
  VAL_DO_SAMPLE=False
  VAL_TEMPERATURE=0
else
  VAL_DO_SAMPLE=True
  VAL_TEMPERATURE=1.0
fi

for bool_var in STUDENT_THINKING SAVE_LORA_ONLY; do
  case "${!bool_var}" in
    1|true|True|TRUE|yes|Yes|YES) printf -v "${bool_var}" '%s' True ;;
    0|false|False|FALSE|no|No|NO) printf -v "${bool_var}" '%s' False ;;
    *) echo "${bool_var} must be a boolean; got ${!bool_var}" >&2; exit 1 ;;
  esac
done
case "${EMIT_THINK_PROMPTS}" in
  "") ;;
  1|true|True|TRUE|yes|Yes|YES) EMIT_THINK_PROMPTS=True ;;
  0|false|False|FALSE|no|No|NO) EMIT_THINK_PROMPTS=False ;;
  *) echo "EMIT_THINK_PROMPTS must be a boolean or empty; got ${EMIT_THINK_PROMPTS}" >&2; exit 1 ;;
esac
# Hydra needs an explicit null for "not specified"; an empty override value is ambiguous.
EMIT_THINK_OVERRIDE="${EMIT_THINK_PROMPTS:-null}"

if [ "${SCALE_MODE}" = "linear" ]; then
  DEFAULT_TRAIN_BATCH_SIZE=$((8 * NGPUS_PER_NODE * NNODES))
elif [ "${SCALE_MODE}" = "fixed" ]; then
  DEFAULT_TRAIN_BATCH_SIZE=16
else
  echo "SCALE_MODE must be fixed or linear; got ${SCALE_MODE}" >&2
  exit 1
fi
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
# The trainer deals the TRAIN_BATCH_SIZE x ROLLOUT_N rollouts of an update evenly over the
# student GPUs, and it requires the rollouts of one optimizer step,
# PPO_MINI_BATCH_SIZE x ROLLOUT_N, to be divisible by their number as well. A count that
# does not divide would stop the run only after the model is loaded and the first rollouts
# are generated, so both are checked here. Values that are appended as hydra overrides
# instead of being set through these variables are not seen by this check.
for batch_var in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE; do
  if [ $(( ${!batch_var} * ${ROLLOUT_N:-8} % (NGPUS_PER_NODE * NNODES) )) -ne 0 ]; then
    echo "${batch_var} x ROLLOUT_N ($(( ${!batch_var} * ${ROLLOUT_N:-8} ))) must be divisible by the number of student GPUs ($(( NGPUS_PER_NODE * NNODES )))." >&2
    exit 1
  fi
done

manifest_model_args=(--model "student=${MODEL_PATH}")
if [ -n "${TEACHER_MODEL:-}" ]; then
  manifest_model_args+=(--model "teacher=${TEACHER_MODEL}")
fi
# Empty for GRPO and vanilla OPD, hence expanded as ${name[@]+"${name[@]}"} below: bash
# older than 4.4 treats a plain "${name[@]}" of an empty array as unbound under set -u.
manifest_method_args=()
if [ -n "${CROSS_VOCAB_OPD_TEACHER_ENDPOINT:-}" ]; then
  manifest_method_args+=(
    --config_kv "cross_vocab_opd_source_commit=${CROSS_VOCAB_OPD_SOURCE_COMMIT:-927a8264f2e303b7f82c2d331a58fd4240c8805a}"
    --config_kv "cross_vocab_opd_teacher_endpoint=${CROSS_VOCAB_OPD_TEACHER_ENDPOINT}"
    --config_kv "cross_vocab_opd_teacher_max_sequence_length=${CROSS_VOCAB_OPD_TEACHER_MAX_SEQUENCE_LENGTH:-30720}"
    --config_kv "cross_vocab_opd_large_chunk_threshold=${CROSS_VOCAB_OPD_LARGE_CHUNK_THRESHOLD:-6}"
    --config_kv "cross_vocab_opd_loss_max_clamp=${CROSS_VOCAB_OPD_LOSS_MAX_CLAMP:-10.0}"
    --config_kv "cross_vocab_opd_policy_loss_mode=opd"
    --config_kv "cross_vocab_opd_clip_ratio_low=0.2"
    --config_kv "cross_vocab_opd_clip_ratio_high=0.28"
    --config_kv "cross_vocab_opd_clip_ratio_c=10.0"
    --config_kv "cross_vocab_opd_loss_agg_mode=token-mean"
  )
fi
if [ "${METHOD_NAME}" = "black_box_ropd" ]; then
  ropd_index_sha256="$(sha256sum "${ROPD_TEACHER_INDEX_PATH}" | awk '{print $1}')"
  manifest_method_args+=(
    --config_kv "ropd_source_commit=6b8d364de4e2cd02672c5d225295fda3bed82128"
    --config_kv "ropd_teacher_index=${ROPD_TEACHER_INDEX_PATH}"
    --config_kv "ropd_teacher_index_sha256=${ropd_index_sha256}"
    --config_kv "ropd_provider_spec=${ROPD_PROVIDER_SPEC}"
    --config_kv "ropd_teacher_model=${ROPD_TEACHER_MODEL}"
    --config_kv "ropd_rubricator_model=${ROPD_RUBRICATOR_MODEL}"
    --config_kv "ropd_verifier_model=${ROPD_VERIFIER_MODEL}"
    --config_kv "ropd_teacher_answer_count=${ROPD_TEACHER_ANSWER_COUNT}"
    --config_kv "ropd_reward=student_score_over_rubric_maximum"
    # The pinned ROPD source parses the reward_quality_gate fallback-rate fields that
    # the ROPD launcher passes but does not enforce them. Recorded so a manifest does
    # not imply a fallback-rate circuit breaker.
    --config_kv "ropd_quality_gate_enforced=false"
    --config_kv "lora_rank=32"
    --config_kv "lora_alpha=64"
  )
fi

# Actor master weights (LoRA parameters and Adam moments) default to fp32, verl's training default: an Adam step at
# learning rate 1e-6 is below the bf16 resolution of the LoRA weights and would be lost to rounding if they were stored
# in bf16. Compute still runs under bf16 mixed precision.
export ACTOR_FSDP_MODEL_DTYPE="${ACTOR_FSDP_MODEL_DTYPE:-fp32}"
write_bsd_run_manifest "${EXPERIMENT_NAME}" \
  "${manifest_model_args[@]}" \
  ${manifest_method_args[@]+"${manifest_method_args[@]}"} \
  --config_kv "method=${METHOD_NAME}" \
  --config_kv "project_name=${PROJECT_NAME}" \
  --config_kv "model_path=${MODEL_PATH}" \
  --config_kv "student_thinking=${STUDENT_THINKING}" \
  --config_kv "emit_think=${EMIT_THINK_PROMPTS}" \
  --config_kv "train_file=${TRAIN_FILE}" \
  --config_kv "val_file=${VAL_FILE}" \
  --config_kv "scale_mode=${SCALE_MODE}" \
  --config_kv "train_batch_size=${TRAIN_BATCH_SIZE}" \
  --config_kv "ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" \
  --config_kv "ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  --config_kv "actor_lr=${ACTOR_LR}" \
  --config_kv "optimizer=AdamW" \
  --config_kv "lr_scheduler_type=constant" \
  --config_kv "lr_warmup_steps=${LR_WARMUP_STEPS}" \
  --config_kv "weight_decay=0.01" \
  --config_kv "clip_grad_norm=1.0" \
  --config_kv "policy_clip_ratio_low=0.2" \
  --config_kv "policy_clip_ratio_high=${POLICY_CLIP_RATIO_HIGH}" \
  --config_kv "loss_aggregation=token-mean" \
  --config_kv "norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO}" \
  --config_kv "rollout_importance_sampling=token" \
  --config_kv "rollout_importance_sampling_clip=2.0" \
  --config_kv "rollout_is_batch_normalize=False" \
  --config_kv "kl_in_reward=False" \
  --config_kv "kl_coef=0.0" \
  --config_kv "kl_loss=False" \
  --config_kv "kl_loss_coef=0.0" \
  --config_kv "rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}" \
  --config_kv "resume_mode=${RESUME_MODE}" \
  --config_kv "data_shuffle=True" \
  --config_kv "model_use_fused_kernels=True" \
  --config_kv "actor_use_dynamic_bsz=True" \
  --config_kv "actor_ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-81920}" \
  --config_kv "actor_fsdp_model_dtype=${ACTOR_FSDP_MODEL_DTYPE}" \
  --config_kv "actor_fsdp_forward_prefetch=True" \
  --config_kv "actor_fsdp_param_offload=False" \
  --config_kv "actor_fsdp_optimizer_offload=False" \
  --config_kv "rollout_log_prob_use_dynamic_bsz=True" \
  --config_kv "rollout_log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-131072}" \
  --config_kv "rollout_max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-16384}" \
  --config_kv "lora_rank=32" \
  --config_kv "lora_alpha=64" \
  --config_kv "save_lora_only=${SAVE_LORA_ONLY}" \
  --config_kv "max_prompt_length=${MAX_PROMPT_LENGTH:-2048}" \
  --config_kv "max_response_length=${MAX_RESPONSE_LENGTH}" \
  --config_kv "total_training_steps=${TOTAL_TRAINING_STEPS}" \
  --config_kv "rollout_n=${ROLLOUT_N:-8}" \
  --config_kv "save_freq=${SAVE_FREQ}" \
  --config_kv "test_freq=${TEST_FREQ}" \
  --config_kv "val_rollout_n=${VAL_ROLLOUT_N}" \
  --config_kv "val_do_sample=${VAL_DO_SAMPLE}" \
  --config_kv "val_temperature=${VAL_TEMPERATURE}" \
  --config_kv "format_bonus=${BSD_FORMAT_BONUS}" \
  --config_kv "hydra_overrides=$*" \
  --config_kv "seed=${SEED}"

"${BSD_TRAIN_PYTHON}" -m "${PPO_ENTRYPOINT}" \
  trainer.use_v1=False \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.custom_cls.path=pkg://verl.trainer.bsd.prompt_dataset \
  data.custom_cls.name=PromptTemplateRLHFDataset \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-2048}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  +data.apply_chat_template_kwargs.enable_thinking="${STUDENT_THINKING}" \
  +data.emit_think=${EMIT_THINK_OVERRIDE} \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=True \
  custom_reward_function.path="${BSD_REPO_ROOT}/recipes/bsd/rewards/math_verify_reward.py" \
  custom_reward_function.name=compute_score \
  reward.custom_reward_function.path="${BSD_REPO_ROOT}/recipes/bsd/rewards/math_verify_reward.py" \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.model.bsd_teacher_adapter=False \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.optim.optimizer=AdamW \
  actor_rollout_ref.actor.optim.optimizer_impl=torch.optim \
  actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.optim.clip_grad=1.0 \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  +actor_rollout_ref.actor.checkpoint.save_lora_only="${SAVE_LORA_ONLY}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-81920}" \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_FSDP_MODEL_DTYPE}" \
  actor_rollout_ref.actor.fsdp_config.seed="${SEED}" \
  actor_rollout_ref.actor.fsdp_config.full_determinism=True \
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high="${POLICY_CLIP_RATIO_HIGH}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.data_loader_seed="${SEED}" \
  actor_rollout_ref.actor.policy_loss.rlsd_enabled=False \
  actor_rollout_ref.actor.policy_loss.opd_pg_enabled=False \
  actor_rollout_ref.actor.policy_loss.cross_vocab_opd_enabled=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE:-1}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.5}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.seed="${SEED}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-131072}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-16384}" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_ROLLOUT_N}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD_IN_GRPO}" \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  algorithm.rollout_correction.rollout_is_batch_normalize=False \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.rollout_correction.bypass_mode=False \
  algorithm.rollout_correction.loss_type=ppo_clip \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.seed="${SEED}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.default_local_dir="${BSD_DATA_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
  trainer.logger="${TRAINER_LOGGER}" \
  "$@"
