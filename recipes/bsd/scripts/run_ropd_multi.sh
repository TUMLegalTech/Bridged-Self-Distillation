#!/usr/bin/env bash
# ROPD baseline (rubric-based OPD) for a closed API external teacher, built on the
# released ROPD code vendored under third_party/ropd_official: the teacher derives
# a rubric for each problem and scores the student's rollouts against it.
# Defaults follow the paper's training configuration: learning rate 1e-6,
# 16 problems per update with 8 rollouts each, 150 updates, LoRA 32/64, a
# checkpoint every 25 updates, and 2,048 / 8,192 prompt / response tokens (the
# response limit of the ROPD release).
# API credentials are read from the environment or from an untracked .env file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DOTENV_PATH="${BSD_DOTENV_PATH:-${REPO_ROOT}/.env}"
if [ -f "${DOTENV_PATH}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${DOTENV_PATH}"
  set +a
fi
source "${SCRIPT_DIR}/_env_multi.sh"

export TRAIN_FILE="${TRAIN_FILE:?TRAIN_FILE is required: the training parquet of the teacher-student pair; see the README section \"Data preparation\"}"
export VAL_FILE="${VAL_FILE:?VAL_FILE is required: the development-set parquet; see the README section \"Data preparation\"}"
# Optional: the dataset manifest of the training parquet; when set, the run manifest records it.
export SPLIT_MANIFEST="${SPLIT_MANIFEST:-}"
export MODEL_PATH="${MODEL_PATH:-$(bsd_model_path Qwen/Qwen3-4B)}"
export PROJECT_NAME="${PROJECT_NAME:-bridged_self_distillation}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME is required: a unique run name; it names the checkpoint and output directories}"
export METHOD_NAME="black_box_ropd"
export PPO_ENTRYPOINT="verl.trainer.bsd.main_bsd"
export SCALE_MODE="fixed"
export STUDENT_THINKING="${STUDENT_THINKING:-true}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
export SAVE_FREQ="${SAVE_FREQ:-25}"
export TEST_FREQ="${TEST_FREQ:-25}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.60}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-131072}"
# Three fixed settings of this baseline that differ from the defaults of the shared launcher:
# no warm-up, upper policy clipping threshold 0.2, and advantages that are not divided by
# the group standard deviation. They are exported instead of appended as hydra overrides, so
# that run_online_rl_multi.sh writes the values in effect into the run manifest.
export LR_WARMUP_STEPS=-1
export POLICY_CLIP_RATIO_HIGH=0.2
export NORM_ADV_BY_STD_IN_GRPO=False

export ROPD_TEACHER_INDEX_PATH="${ROPD_TEACHER_INDEX_PATH:-${BSD_DATA_ROOT}/datasets/ropd/teacher_index.jsonl}"
export ROPD_PROVIDER_SPEC="${ROPD_PROVIDER_SPEC:-${BSD_REPO_ROOT}/recipes/bsd/configs/ropd_judge_providers.yaml}"
export ROPD_TEACHER_ANSWER_COUNT="${ROPD_TEACHER_ANSWER_COUNT:-4}"
export ROPD_MAX_CONCURRENCY="${ROPD_MAX_CONCURRENCY:-4}"
# No silent default: the teacher / rubricator / verifier model defines the baseline
# and its API cost, so it has to be named explicitly. Each role falls back to
# CLOSED_TEACHER_MODEL and is checked on its own.
for ropd_role in TEACHER RUBRICATOR VERIFIER; do
  ropd_model_var="ROPD_${ropd_role}_MODEL"
  ropd_model="${!ropd_model_var:-${CLOSED_TEACHER_MODEL:-}}"
  if [ -z "${ropd_model}" ]; then
    echo "Set CLOSED_TEACHER_MODEL or ${ropd_model_var} before launching ROPD." >&2
    exit 1
  fi
  export "${ropd_model_var}=${ropd_model}"
done
export OPENAI_API_KEY="${OPENAI_API_KEY:-${CLOSED_TEACHER_API_KEY:-}}"
# Export the base URL only when one is configured: an empty OPENAI_BASE_URL is not
# "unset" for the OpenAI client and would replace its default endpoint.
ropd_base_url="${OPENAI_BASE_URL:-${CLOSED_TEACHER_BASE_URL:-}}"
if [ -n "${ropd_base_url}" ]; then
  export OPENAI_BASE_URL="${ropd_base_url}"
else
  unset OPENAI_BASE_URL
fi

# Hydra takes the last assignment, and "$@" lands after this recipe's own overrides inside
# run_online_rl_multi.sh. A caller-supplied data.emit_think or
# data.apply_chat_template_kwargs.enable_thinking would therefore pick a different student
# template at rollout time while the preflight below still keyed the teacher index off the
# environment variables: the index would validate clean and then miss on every group at step 1.
# Same reasoning as the VAL_ROLLOUT_N refusal further down -- set the prompt protocol through
# the environment or not at all.
for hydra_override in "$@"; do
  case "${hydra_override}" in
    data.emit_think=*|+data.emit_think=*|++data.emit_think=*|\
    data.apply_chat_template_kwargs.enable_thinking=*|\
    +data.apply_chat_template_kwargs.enable_thinking=*|\
    ++data.apply_chat_template_kwargs.enable_thinking=*)
      echo "Set the student prompt protocol through STUDENT_THINKING/EMIT_THINK_PROMPTS, not ${hydra_override}: the ROPD teacher index is keyed on the resolved student prompt and the preflight cannot see a hydra override." >&2
      exit 1
      ;;
  esac
done

if [ -z "${OPENAI_API_KEY}" ]; then
  echo "Set OPENAI_API_KEY or CLOSED_TEACHER_API_KEY for the ROPD rubricator and verifier." >&2
  exit 1
fi
if [ ! -f "${ROPD_TEACHER_INDEX_PATH}" ]; then
  echo "Missing ROPD teacher index: ${ROPD_TEACHER_INDEX_PATH}" >&2
  echo "Build it with recipes/bsd/scripts/04_build_ropd_teacher_index.sh." >&2
  exit 1
fi
if [ $((TRAIN_BATCH_SIZE % (NGPUS_PER_NODE * NNODES))) -ne 0 ]; then
  echo "TRAIN_BATCH_SIZE must be divisible by world size $((NGPUS_PER_NODE * NNODES))." >&2
  exit 1
fi

# The offline teacher index is keyed on a hash of the student prompt, so the preflight has to
# render that prompt the way this run will. PromptTemplateRLHFDataset resolves the variant
# through resolve_emit_think(enable_thinking, emit_think); forward both inputs verbatim rather
# than recomputing the tri-state here, so the check and the run cannot drift apart.
"${BSD_TRAIN_PYTHON}" "${BSD_REPO_ROOT}/tools/bsd/preflight_ropd.py" \
  --train_parquet "${TRAIN_FILE}" \
  --teacher_index "${ROPD_TEACHER_INDEX_PATH}" \
  --provider_spec "${ROPD_PROVIDER_SPEC}" \
  --teacher_answer_count "${ROPD_TEACHER_ANSWER_COUNT}" \
  --enable_thinking "${STUDENT_THINKING}" \
  --emit_think "${EMIT_THINK_PROMPTS:-}"

export RUN_MANIFEST="${RUN_MANIFEST:-${BSD_DATA_ROOT}/outputs/${EXPERIMENT_NAME}/run_manifest.json}"

# This recipe pins its own val_kwargs below, and those land after the shared launcher's, so
# hydra's last-wins rule makes them authoritative. Honour VAL_ROLLOUT_N only when it agrees;
# silently ignoring it would let the manifest record a validation protocol the run did not
# use. The pinned value comes from this baseline's source, so change it there, not here.
if [ "${VAL_ROLLOUT_N:-4}" != "4" ]; then
  echo "black-box ROPD pins val_kwargs.n=4 from its source recipe; VAL_ROLLOUT_N=${VAL_ROLLOUT_N} cannot take effect" >&2
  exit 1
fi
export VAL_ROLLOUT_N=4

# WARNING: of the eight reward_quality_gate fields, the pinned source reads only
# max_step_judge_retry_attempts and the three backoff values. enabled,
# max_fallback_rate, max_retry_rounds and retry_pair_concurrency are parsed into
# RopdRewardQualityGateConfig (reward_manager.py:31-39) and never read anywhere in
# third_party/ropd_official; quality_gate_stop is hard-coded False at
# reward_manager.py:736 and :810. So there is NO fallback-rate circuit breaker: if
# the judge API fails persistently (for example an exhausted key), every reward
# falls back and the run keeps going at exactly zero gradient. They are passed anyway
# because pinned source behaviour is authoritative for this baseline and
# upstream may wire them up later -- but do not read them as a safety guarantee.
# The manifest records ropd_quality_gate_enforced=false so a reader cannot either.
exec bash "${SCRIPT_DIR}/run_online_rl_multi.sh" \
  ++bsd.trainer_class=verl.trainer.bsd.ropd_trainer.RopdRayPPOTrainer \
  ++ropd.teacher_answer_count="${ROPD_TEACHER_ANSWER_COUNT}" \
  ++ropd.max_concurrency="${ROPD_MAX_CONCURRENCY}" \
  ++ropd.provider_resolution.spec_path="${ROPD_PROVIDER_SPEC}" \
  ++ropd.provider_resolution.entrypoint=train \
  ++ropd.provider_resolution.overrides.global={} \
  ++ropd.reward_quality_gate.enabled=true \
  ++ropd.reward_quality_gate.max_fallback_rate=0.4 \
  ++ropd.reward_quality_gate.max_retry_rounds=1 \
  ++ropd.reward_quality_gate.retry_pair_concurrency=1 \
  ++ropd.reward_quality_gate.max_step_judge_retry_attempts=2 \
  ++ropd.reward_quality_gate.step_judge_retry_initial_backoff_seconds=1.0 \
  ++ropd.reward_quality_gate.step_judge_retry_backoff_multiplier=2.0 \
  ++ropd.reward_quality_gate.step_judge_retry_max_backoff_seconds=4.0 \
  ++ropd.debug.output_dir="${BSD_DATA_ROOT}/outputs/${EXPERIMENT_NAME}/ropd_debug" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=64 \
  ++actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.rollout.max_model_len=10240 \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=true \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  data.trust_remote_code=true \
  data.validation_shuffle=false \
  trainer.val_before_train=false \
  "$@"
