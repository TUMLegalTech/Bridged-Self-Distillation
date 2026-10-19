#!/usr/bin/env bash
# SFT baseline: the student is fine-tuned (LoRA) to imitate the external teacher's
# responses, the same responses that serve as traces for BSD.
# Defaults follow the paper's training configuration: learning rate 5e-6,
# 16 problems per update, 150 updates, LoRA 32/64, a checkpoint every 25 updates.
set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env_multi.sh"

TRAIN_FILE="${TRAIN_FILE:?TRAIN_FILE is required: the SFT parquet (messages column) of the teacher-student pair; see the README section \"Data preparation\"}"
# Optional: SFT runs without in-training validation unless VAL_FILE and TEST_FREQ are set.
VAL_FILE="${VAL_FILE:-}"
MODEL_PATH="${MODEL_PATH:-$(bsd_model_path Qwen/Qwen3-4B)}"
PROJECT_NAME="${PROJECT_NAME:-bridged_self_distillation}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME is required: a unique run name; it names the checkpoint and output directories}"
# Optional: the dataset manifest of the training parquet; when set, the run manifest records it.
SPLIT_MANIFEST="${SPLIT_MANIFEST:-}"
SCALE_MODE="${SCALE_MODE:-fixed}"
SFT_MICRO_BATCH_SIZE_PER_GPU="${SFT_MICRO_BATCH_SIZE_PER_GPU:-${SFT_MICRO_BATCH_SIZE:-1}}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
SFT_TOTAL_TRAINING_STEPS="${SFT_TOTAL_TRAINING_STEPS:-150}"
SFT_LR="${SFT_LR:-5e-6}"
# One limit on the whole tokenized conversation (prompt and teacher response together).
# Nothing is truncated: a longer row fails the preflight below and the dataset
# (truncation=error).
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-10240}"
SFT_MAX_TOKEN_LEN_PER_GPU="${SFT_MAX_TOKEN_LEN_PER_GPU:-${MAX_SEQUENCE_LENGTH}}"
SFT_LORA_RANK="${SFT_LORA_RANK:-32}"
SFT_LORA_ALPHA="${SFT_LORA_ALPHA:-64}"
SAVE_FREQ="${SAVE_FREQ:-25}"
TEST_FREQ="${TEST_FREQ:--1}"
RESUME_MODE="${RESUME_MODE:-disable}"
SFT_PREFLIGHT="${SFT_PREFLIGHT:-1}"
SFT_THINKING="${SFT_THINKING:-True}"
# Tri-state, same contract as the online launchers. It selects the student template variant
# through resolve_emit_think, which the SFT dataset applies to the user turn.
#
# Empty follows SFT_THINKING: with thinking on, the default, it selects the plain student
# prompt of the Qwen3 students. Qwen3-4B-Instruct sets EMIT_THINK_PROMPTS=True and keeps
# SFT_THINKING=True: its chat template opens the reasoning block, and the last line of
# its prompt is the instruction to reason inside the <think> tags
# (student_emit_think.jinja). The explicit value keeps the SFT baseline on the same
# student prompt as the online methods and records the choice in the run manifest.
EMIT_THINK_PROMPTS="${EMIT_THINK_PROMPTS:-}"
SFT_IGNORE_INPUT_IDS_MISMATCH="${SFT_IGNORE_INPUT_IDS_MISMATCH:-False}"
SAVE_LORA_ONLY="${SAVE_LORA_ONLY:-True}"
SEED="${SEED:-43}"

for bool_var in SFT_THINKING SFT_IGNORE_INPUT_IDS_MISMATCH SAVE_LORA_ONLY; do
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

if [ "${SFT_IGNORE_INPUT_IDS_MISMATCH}" != "False" ]; then
  echo "SFT_IGNORE_INPUT_IDS_MISMATCH=True is not supported; whole-conversation prefix alignment is mandatory" >&2
  exit 1
fi

if [ "${SCALE_MODE}" = "linear" ]; then
  DEFAULT_SFT_TRAIN_BATCH_SIZE=$((8 * NGPUS_PER_NODE * NNODES))
elif [ "${SCALE_MODE}" = "fixed" ]; then
  DEFAULT_SFT_TRAIN_BATCH_SIZE=16
else
  echo "SCALE_MODE must be fixed or linear; got ${SCALE_MODE}" >&2
  exit 1
fi
SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-${DEFAULT_SFT_TRAIN_BATCH_SIZE}}"
WORLD_SIZE=$((NGPUS_PER_NODE * NNODES))
if [ $((SFT_TRAIN_BATCH_SIZE % WORLD_SIZE)) -ne 0 ]; then
  echo "SFT_TRAIN_BATCH_SIZE must be divisible by WORLD_SIZE=${WORLD_SIZE}; got ${SFT_TRAIN_BATCH_SIZE}" >&2
  exit 1
fi

val_args=(data.val_files=null)
if [ -n "${VAL_FILE}" ]; then
  val_args=(data.val_files="${VAL_FILE}")
fi

if [ "${SFT_PREFLIGHT}" != "0" ]; then
  "${BSD_TRAIN_PYTHON}" "${BSD_REPO_ROOT}/tools/bsd/inspect_sft_parquet.py" \
    --parquet "${TRAIN_FILE}" \
    --model "${MODEL_PATH}" \
    --max_length "${MAX_SEQUENCE_LENGTH}" \
    --enable_thinking_key __force_enable_thinking_missing__ \
    --enable_thinking_default "${SFT_THINKING}" \
    --fail_on_overlong
fi

write_bsd_run_manifest "${EXPERIMENT_NAME}" \
  --model "student=${MODEL_PATH}" \
  --config_kv "method=off_policy_sft" \
  --config_kv "project_name=${PROJECT_NAME}" \
  --config_kv "model_path=${MODEL_PATH}" \
  --config_kv "train_file=${TRAIN_FILE}" \
  --config_kv "val_file=${VAL_FILE}" \
  --config_kv "scale_mode=${SCALE_MODE}" \
  --config_kv "sft_train_batch_size=${SFT_TRAIN_BATCH_SIZE}" \
  --config_kv "data_shuffle=true" \
  --config_kv "micro_batch_size_per_gpu=${SFT_MICRO_BATCH_SIZE_PER_GPU}" \
  --config_kv "sft_epochs=${SFT_EPOCHS}" \
  --config_kv "total_training_steps=${SFT_TOTAL_TRAINING_STEPS}" \
  --config_kv "sft_lr=${SFT_LR}" \
  --config_kv "max_sequence_length=${MAX_SEQUENCE_LENGTH}" \
  --config_kv "max_token_len_per_gpu=${SFT_MAX_TOKEN_LEN_PER_GPU}" \
  --config_kv "lora_rank=${SFT_LORA_RANK}" \
  --config_kv "lora_alpha=${SFT_LORA_ALPHA}" \
  --config_kv "save_freq=${SAVE_FREQ}" \
  --config_kv "test_freq=${TEST_FREQ}" \
  --config_kv "seed=${SEED}" \
  --config_kv "sft_preflight=${SFT_PREFLIGHT}" \
  --config_kv "student_thinking=${SFT_THINKING}" \
  --config_kv "emit_think=${EMIT_THINK_PROMPTS}" \
  --config_kv "ignore_input_ids_mismatch=${SFT_IGNORE_INPUT_IDS_MISMATCH}" \
  --config_kv "save_lora_only=${SAVE_LORA_ONLY}" \
  --config_kv "hydra_overrides=$*" \
  --config_kv "multi_gpu=true"

"${BSD_TRAIN_PYTHON}" -m torch.distributed.run --standalone \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${TRAIN_FILE}" \
  "${val_args[@]}" \
  data.custom_cls.path=pkg://verl.trainer.bsd.prompt_dataset \
  data.custom_cls.name=PromptTemplateSFTDataset \
  data.messages_key=messages \
  data.enable_thinking_key=__force_enable_thinking_missing__ \
  data.enable_thinking_default="${SFT_THINKING}" \
  +data.apply_chat_template_kwargs.enable_thinking="${SFT_THINKING}" \
  +data.emit_think=${EMIT_THINK_OVERRIDE} \
  data.ignore_input_ids_mismatch="${SFT_IGNORE_INPUT_IDS_MISMATCH}" \
  data.train_batch_size="${SFT_TRAIN_BATCH_SIZE}" \
  data.shuffle=True \
  data.micro_batch_size_per_gpu="${SFT_MICRO_BATCH_SIZE_PER_GPU}" \
  data.max_length="${MAX_SEQUENCE_LENGTH}" \
  data.max_token_len_per_gpu="${SFT_MAX_TOKEN_LEN_PER_GPU}" \
  data.truncation=error \
  data.num_workers="${SFT_NUM_WORKERS:-0}" \
  optim.lr="${SFT_LR}" \
  model.path="${MODEL_PATH}" \
  model.enable_gradient_checkpointing=True \
  model.lora_rank="${SFT_LORA_RANK}" \
  model.lora_alpha="${SFT_LORA_ALPHA}" \
  engine.seed="${SEED}" \
  engine.full_determinism=True \
  +checkpoint.save_lora_only="${SAVE_LORA_ONLY}" \
  trainer.default_local_dir="${BSD_DATA_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
  trainer.total_epochs="${SFT_EPOCHS}" \
  trainer.total_training_steps="${SFT_TOTAL_TRAINING_STEPS}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.seed="${SEED}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.logger="${TRAINER_LOGGER}" \
  "$@"
