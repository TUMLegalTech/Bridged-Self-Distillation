#!/usr/bin/env bash
# Vanilla OPD baseline for a shared-tokenizer external teacher. The teacher runs
# in its own resource pool and scores the student's rollouts token by token.
# By default two GPUs train the student (NGPUS_PER_NODE) and one serves the teacher
# (TEACHER_NGPUS_PER_NODE); the rollouts of an update, TRAIN_BATCH_SIZE x ROLLOUT_N,
# must be divisible by the number of student GPUs.
# Defaults follow the paper's training configuration: learning rate 1e-6,
# 16 problems per update with 1 rollout each, 150 updates, LoRA 32/64, a
# checkpoint every 25 updates, and 2,048 / 12,288 prompt / response tokens.
set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${NGPUS_PER_NODE}}"
source "${SCRIPT_DIR}/_env_multi.sh"

TRAIN_FILE="${TRAIN_FILE:?TRAIN_FILE is required: the training parquet of the teacher-student pair; see the README section \"Data preparation\"}"
VAL_FILE="${VAL_FILE:?VAL_FILE is required: the development-set parquet; see the README section \"Data preparation\"}"
# Optional: the dataset manifest of the training parquet; when set, the run manifest records it.
SPLIT_MANIFEST="${SPLIT_MANIFEST:-}"
TEACHER_MODEL="${TEACHER_MODEL:-$(bsd_model_path Qwen/Qwen3-32B)}"
MODEL_PATH="${MODEL_PATH:-$(bsd_model_path Qwen/Qwen3-4B)}"
PROJECT_NAME="${PROJECT_NAME:-bridged_self_distillation}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME is required: a unique run name; it names the checkpoint and output directories}"
METHOD_NAME="${METHOD_NAME:-opd_pg_qwen32b_multi_teacher_pool}"
SCALE_MODE="${SCALE_MODE:-fixed}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
OPD_LOGRATIO_CLIP_ABS="${OPD_LOGRATIO_CLIP_ABS:-2.0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-12288}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
ROLLOUT_N="${ROLLOUT_N:-1}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
SAVE_FREQ="${SAVE_FREQ:-25}"
TEST_FREQ="${TEST_FREQ:-25}"
STUDENT_THINKING="${STUDENT_THINKING:-True}"
export TRAIN_FILE VAL_FILE SPLIT_MANIFEST TEACHER_MODEL MODEL_PATH PROJECT_NAME
export EXPERIMENT_NAME METHOD_NAME SCALE_MODE TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE
export PPO_MICRO_BATCH_SIZE_PER_GPU ACTOR_LR OPD_LOGRATIO_CLIP_ABS
export MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_NUM_BATCHED_TOKENS ROLLOUT_N
export ROLLOUT_TP_SIZE ROLLOUT_GPU_MEM_UTIL ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU
export TOTAL_TRAINING_STEPS SAVE_FREQ TEST_FREQ

TEACHER_NGPUS_PER_NODE="${TEACHER_NGPUS_PER_NODE:-1}"
if [ "${TEACHER_NGPUS_PER_NODE}" -lt 1 ]; then
  echo "TEACHER_NGPUS_PER_NODE must be >=1 for multi teacher-pool OPD." >&2
  exit 1
fi
if [ $((NGPUS_PER_NODE + TEACHER_NGPUS_PER_NODE)) -gt "${BSD_VISIBLE_GPU_COUNT}" ]; then
  echo "Requested actor GPUs (${NGPUS_PER_NODE}) + teacher GPUs (${TEACHER_NGPUS_PER_NODE}) exceeds visible GPUs (${BSD_VISIBLE_GPU_COUNT})." >&2
  echo "Set NGPUS_PER_NODE to the actor pool size, for example 2 with CUDA_VISIBLE_DEVICES=0,1,2 and TEACHER_NGPUS_PER_NODE=1." >&2
  exit 1
fi

"${BSD_TRAIN_PYTHON}" "${BSD_REPO_ROOT}/tools/bsd/check_tokenizer_compat.py" \
  --student_model "${MODEL_PATH}" \
  --teacher_model "${TEACHER_MODEL}"

bash "${SCRIPT_DIR}/run_online_rl_multi.sh" \
  distillation.enabled=True \
  distillation.n_gpus_per_node="${TEACHER_NGPUS_PER_NODE}" \
  distillation.nnodes="${TEACHER_NNODES:-${NNODES}}" \
  distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="${TEACHER_TP_SIZE:-1}" \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TEACHER_GPU_MEM_UTIL:-0.5}" \
  distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-8192}" \
  distillation.distillation_loss.loss_mode=k1 \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient=True \
  distillation.distillation_loss.loss_max_clamp="${OPD_LOGRATIO_CLIP_ABS}" \
  distillation.distillation_loss.log_prob_min_clamp="-${OPD_LOGRATIO_CLIP_ABS}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  "$@"
