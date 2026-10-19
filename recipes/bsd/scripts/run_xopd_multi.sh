#!/usr/bin/env bash
# xOPD baseline (cross-tokenizer OPD) for an external teacher whose tokenizer
# differs from the student's. The teacher is served by a separate vLLM process on
# its own GPU(s); its token log-probabilities are aligned to the student's tokens.
# By default two GPUs train the student (ACTOR_GPUS) and one serves the teacher
# (TEACHER_GPUS); the rollouts of an update, TRAIN_BATCH_SIZE x ROLLOUT_N, and of one
# optimizer step, PPO_MINI_BATCH_SIZE x ROLLOUT_N, must be divisible by the number of
# student GPUs.
# Defaults follow the paper's training configuration: learning rate 1e-6,
# 16 problems per update with 8 rollouts each, 150 updates, LoRA 32/64, a
# checkpoint every 25 updates, and 2,048 / 12,288 prompt / response tokens.
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CROSS_VOCAB_OPD_SOURCE_COMMIT="927a8264f2e303b7f82c2d331a58fd4240c8805a"
ACTOR_GPUS="${ACTOR_GPUS:-0,1}"
TEACHER_GPUS="${TEACHER_GPUS:-2}"

IFS=',' read -r -a actor_gpu_array <<< "${ACTOR_GPUS}"
IFS=',' read -r -a teacher_gpu_array <<< "${TEACHER_GPUS}"
for actor_gpu in "${actor_gpu_array[@]}"; do
  for teacher_gpu in "${teacher_gpu_array[@]}"; do
    if [ "${actor_gpu}" = "${teacher_gpu}" ]; then
      echo "ACTOR_GPUS and TEACHER_GPUS must be disjoint; both contain ${actor_gpu}" >&2
      exit 1
    fi
  done
done

export CUDA_VISIBLE_DEVICES="${ACTOR_GPUS}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-${#actor_gpu_array[@]}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${NGPUS_PER_NODE}}"
source "${SCRIPT_DIR}/_env_multi.sh"
if [ "${NGPUS_PER_NODE}" -ne "${#actor_gpu_array[@]}" ]; then
  echo "NGPUS_PER_NODE must match ACTOR_GPUS count (${#actor_gpu_array[@]}); got ${NGPUS_PER_NODE}" >&2
  exit 1
fi

TEACHER_MODEL="${TEACHER_MODEL:-$(bsd_model_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)}"
MODEL_PATH="${MODEL_PATH:-$(bsd_model_path Qwen/Qwen3-4B)}"
PROJECT_NAME="${PROJECT_NAME:-bridged_self_distillation}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME is required: a unique run name; it names the checkpoint and output directories}"
METHOD_NAME="${METHOD_NAME:-cross_vocab_opd_nemotron}"
TRAIN_FILE="${TRAIN_FILE:?TRAIN_FILE is required: the training parquet of the teacher-student pair; see the README section \"Data preparation\"}"
VAL_FILE="${VAL_FILE:?VAL_FILE is required: the development-set parquet; see the README section \"Data preparation\"}"
# Optional: the dataset manifest of the training parquet; when set, the run manifest records it.
SPLIT_MANIFEST="${SPLIT_MANIFEST:-}"
STUDENT_THINKING="${STUDENT_THINKING:-True}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-12288}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
SAVE_FREQ="${SAVE_FREQ:-25}"
TEST_FREQ="${TEST_FREQ:-25}"

CROSS_VOCAB_OPD_TEACHER_ENDPOINT="${CROSS_VOCAB_OPD_TEACHER_ENDPOINT:-tcp://127.0.0.1:15555}"
CROSS_VOCAB_OPD_TEACHER_MAX_SEQUENCE_LENGTH="${CROSS_VOCAB_OPD_TEACHER_MAX_SEQUENCE_LENGTH:-30720}"
CROSS_VOCAB_OPD_TEACHER_TIMEOUT_SECONDS="${CROSS_VOCAB_OPD_TEACHER_TIMEOUT_SECONDS:-600}"
CROSS_VOCAB_OPD_LARGE_CHUNK_THRESHOLD="${CROSS_VOCAB_OPD_LARGE_CHUNK_THRESHOLD:-6}"
CROSS_VOCAB_OPD_LOSS_MAX_CLAMP="${CROSS_VOCAB_OPD_LOSS_MAX_CLAMP:-10.0}"
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-${#teacher_gpu_array[@]}}"
TEACHER_GPU_MEM_UTIL="${TEACHER_GPU_MEM_UTIL:-0.85}"
TEACHER_MAX_NUM_BATCHED_TOKENS="${TEACHER_MAX_NUM_BATCHED_TOKENS:-8192}"
START_TEACHER_SERVER="${START_TEACHER_SERVER:-1}"
TEACHER_READY_TIMEOUT_SECONDS="${TEACHER_READY_TIMEOUT_SECONDS:-900}"
TEACHER_SERVER_LOG="${TEACHER_SERVER_LOG:-${BSD_DATA_ROOT}/outputs/${EXPERIMENT_NAME}/teacher_server.log}"
if [ "${TEACHER_TP_SIZE}" -ne "${#teacher_gpu_array[@]}" ]; then
  echo "TEACHER_TP_SIZE must match TEACHER_GPUS count (${#teacher_gpu_array[@]}); got ${TEACHER_TP_SIZE}" >&2
  exit 1
fi
# run_online_rl_multi.sh makes the same check, but it is reached only after the teacher
# server below has started and answered, so a count that does not divide is refused here.
for batch_var in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE; do
  if [ $(( ${!batch_var} * ROLLOUT_N % (NGPUS_PER_NODE * NNODES) )) -ne 0 ]; then
    echo "${batch_var} x ROLLOUT_N ($(( ${!batch_var} * ROLLOUT_N ))) must be divisible by the number of student GPUs ($(( NGPUS_PER_NODE * NNODES )))." >&2
    exit 1
  fi
done

export TEACHER_MODEL MODEL_PATH PROJECT_NAME EXPERIMENT_NAME METHOD_NAME
export TRAIN_FILE VAL_FILE SPLIT_MANIFEST STUDENT_THINKING MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH
export TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU ROLLOUT_N ACTOR_LR
export TOTAL_TRAINING_STEPS SAVE_FREQ TEST_FREQ
export CROSS_VOCAB_OPD_TEACHER_ENDPOINT CROSS_VOCAB_OPD_TEACHER_MAX_SEQUENCE_LENGTH
export CROSS_VOCAB_OPD_LARGE_CHUNK_THRESHOLD CROSS_VOCAB_OPD_LOSS_MAX_CLAMP
export CROSS_VOCAB_OPD_SOURCE_COMMIT

teacher_pid=""
cleanup_teacher() {
  if [ -n "${teacher_pid}" ] && kill -0 "${teacher_pid}" 2>/dev/null; then
    kill -- "-${teacher_pid}" 2>/dev/null || true
    wait "${teacher_pid}" 2>/dev/null || true
  fi
}
trap cleanup_teacher EXIT INT TERM

case "${START_TEACHER_SERVER}" in
  1|true|TRUE|yes|YES)
    GENERATION_ENV_PREFIX="${BSD_GENERATION_ENV_PREFIX:-${BSD_ENV_ROOT}/BSD-generation}"
    GENERATION_PYTHON="${BSD_GENERATION_PYTHON:-${GENERATION_ENV_PREFIX}/bin/python}"
    if [ ! -x "${GENERATION_PYTHON}" ]; then
      echo "Missing generation Python: ${GENERATION_PYTHON}" >&2
      exit 1
    fi
    "${GENERATION_PYTHON}" "${REPO_ROOT}/tools/bsd/runtime.py" \
      --role generation \
      --expected-prefix "${GENERATION_ENV_PREFIX}" >/dev/null
    mkdir -p "$(dirname "${TEACHER_SERVER_LOG}")"
    # A server that already answers at the endpoint would pass the readiness check below in
    # place of the one started here, whose bind then fails. One short probe rules that out.
    if "${BSD_TRAIN_PYTHON}" "${REPO_ROOT}/tools/bsd/wait_for_cross_vocab_teacher.py" \
      --endpoint "${CROSS_VOCAB_OPD_TEACHER_ENDPOINT}" \
      --wait_seconds 1 \
      --request_timeout_seconds 3 >/dev/null 2>&1; then
      echo "A teacher server already answers at ${CROSS_VOCAB_OPD_TEACHER_ENDPOINT}. Stop it, or set START_TEACHER_SERVER=0 to use it." >&2
      exit 1
    fi
    setsid env \
      CUDA_VISIBLE_DEVICES="${TEACHER_GPUS}" \
      PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      "${GENERATION_PYTHON}" "${REPO_ROOT}/tools/bsd/cross_vocab_teacher_server.py" \
      --model "${TEACHER_MODEL}" \
      --bind "${CROSS_VOCAB_OPD_TEACHER_ENDPOINT}" \
      --tensor_parallel_size "${TEACHER_TP_SIZE}" \
      --gpu_memory_utilization "${TEACHER_GPU_MEM_UTIL}" \
      --max_model_len "${CROSS_VOCAB_OPD_TEACHER_MAX_SEQUENCE_LENGTH}" \
      --max_num_batched_tokens "${TEACHER_MAX_NUM_BATCHED_TOKENS}" \
      >"${TEACHER_SERVER_LOG}" 2>&1 &
    teacher_pid=$!
    "${BSD_TRAIN_PYTHON}" "${REPO_ROOT}/tools/bsd/wait_for_cross_vocab_teacher.py" \
      --endpoint "${CROSS_VOCAB_OPD_TEACHER_ENDPOINT}" \
      --wait_seconds "${TEACHER_READY_TIMEOUT_SECONDS}" \
      --pid "${teacher_pid}" \
      --expected_model "${TEACHER_MODEL}"
    ;;
  0|false|FALSE|no|NO)
    "${BSD_TRAIN_PYTHON}" "${REPO_ROOT}/tools/bsd/wait_for_cross_vocab_teacher.py" \
      --endpoint "${CROSS_VOCAB_OPD_TEACHER_ENDPOINT}" \
      --wait_seconds "${TEACHER_READY_TIMEOUT_SECONDS}"
    ;;
  *)
    echo "START_TEACHER_SERVER must be a boolean; got ${START_TEACHER_SERVER}" >&2
    exit 1
    ;;
esac

PPO_ENTRYPOINT=verl.trainer.bsd.main_bsd \
bash "${SCRIPT_DIR}/run_online_rl_multi.sh" \
  actor_rollout_ref.actor.policy_loss.rlsd_enabled=False \
  actor_rollout_ref.actor.policy_loss.opd_pg_enabled=False \
  actor_rollout_ref.actor.policy_loss.cross_vocab_opd_enabled=True \
  actor_rollout_ref.actor.policy_loss.cross_vocab_opd_loss_max_clamp="${CROSS_VOCAB_OPD_LOSS_MAX_CLAMP}" \
  actor_rollout_ref.actor.policy_loss.loss_mode=opd \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  +bsd.cross_vocab_opd_teacher_backend=zmq_vllm \
  +bsd.cross_vocab_opd_teacher_model_path="${TEACHER_MODEL}" \
  +bsd.cross_vocab_opd_teacher_endpoint="${CROSS_VOCAB_OPD_TEACHER_ENDPOINT}" \
  +bsd.cross_vocab_opd_teacher_max_sequence_length="${CROSS_VOCAB_OPD_TEACHER_MAX_SEQUENCE_LENGTH}" \
  +bsd.cross_vocab_opd_teacher_timeout_seconds="${CROSS_VOCAB_OPD_TEACHER_TIMEOUT_SECONDS}" \
  +bsd.cross_vocab_opd_large_chunk_threshold="${CROSS_VOCAB_OPD_LARGE_CHUNK_THRESHOLD}" \
  "$@"
