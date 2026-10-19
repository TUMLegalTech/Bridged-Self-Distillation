#!/usr/bin/env bash
# Evaluate one model or checkpoint on AIME 2024, AIME 2025 and HMMT 2025 with the
# evaluation parameters of the paper: 4 samples per problem, seeds 0-3, thinking enabled,
# temperature 1.0, top-p 1.0, at most 32,768 new tokens. In every result JSON "avg" is
# Avg@4 and "pass" is Pass@4.
#
# Run 00_prepare_eval_benchmarks.sh first. MODEL_PATH is the base model,
# CHECKPOINT_PATH a training checkpoint whose LoRA adapter is applied to it. Results go to
# ${BSD_DATA_ROOT}/outputs/eval_official/<RUN_NAME>/<dataset>_<CHECKPOINT_NAME>.json,
# with summary.csv and summary.md next to them (OUTPUT_ROOT overrides the directory).
# For a checkpoint .../<run>/global_step_<N>[/actor] the two names default to <run> and
# global_step_<N>; without a checkpoint both default to the basename of MODEL_PATH.
# EVAL_BACKEND selects the generation backend, vllm (default) or hf. The benchmark files
# are read from EVAL_DATA_DIR, which defaults to the directory that
# 00_prepare_eval_benchmarks.sh writes (OFFICIAL_EVAL_DATA_DIR, if set).
# EVAL_DATASETS (default "aime24 aime25 hmmt25") lists the sets to evaluate, each a file
# <dataset>.parquet in EVAL_DATA_DIR.
# Only the sampled passes run. EVAL_GREEDY_PASS=1 (default 0) adds one temperature-0 pass,
# reported in the separate greedy_* fields, which are null without it.
# Qwen3-4B-Instruct is evaluated with thinking enabled and EMIT_THINK_PROMPTS=True.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

EVAL_DATA_DIR="${EVAL_DATA_DIR:-${OFFICIAL_EVAL_DATA_DIR:-${BSD_DATA_ROOT}/datasets/eval/official_math}}"
DATASETS="${EVAL_DATASETS:-aime24 aime25 hmmt25}"
MODEL_PATH="${MODEL_PATH:-$(bsd_model_path Qwen/Qwen3-4B)}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
if [ -n "${CHECKPOINT_PATH}" ]; then
  # Name the results after the run and the step. The checkpoint directory of every online
  # run is called "actor", so its basename alone would give all checkpoints one result file.
  checkpoint_dir="${CHECKPOINT_PATH%/}"
  if [ "$(basename "${checkpoint_dir}")" = "actor" ]; then
    checkpoint_dir="$(dirname "${checkpoint_dir}")"
  fi
  RUN_NAME="${RUN_NAME:-$(basename "$(dirname "${checkpoint_dir}")")}"
  CHECKPOINT_NAME="${CHECKPOINT_NAME:-$(basename "${checkpoint_dir}")}"
else
  RUN_NAME="${RUN_NAME:-$(basename "${MODEL_PATH}")}"
  CHECKPOINT_NAME="${CHECKPOINT_NAME:-${RUN_NAME}}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${BSD_DATA_ROOT}/outputs/eval_official/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

outputs=()
for dataset in ${DATASETS}; do
  eval_file="${EVAL_DATA_DIR}/${dataset}.parquet"
  if [ ! -f "${eval_file}" ]; then
    echo "Missing eval file: ${eval_file}" >&2
    echo "Run recipes/bsd/scripts/00_prepare_eval_benchmarks.sh first." >&2
    exit 1
  fi

  output_json="${OUTPUT_ROOT}/${dataset}_${CHECKPOINT_NAME}.json"
  log_file="${LOG_DIR}/${dataset}_${CHECKPOINT_NAME}.log"
  args=(
    --eval_file "${eval_file}"
    --dataset_name "${dataset}"
    --model "${MODEL_PATH}"
    --output_json "${output_json}"
    --run "${RUN_NAME}"
    --checkpoint_name "${CHECKPOINT_NAME}"
    --backend "${EVAL_BACKEND:-vllm}"
    --num_samples "${NUM_SAMPLES:-4}"
    --seeds "${EVAL_SEEDS:-0,1,2,3}"
    --max_new_tokens "${MAX_NEW_TOKENS:-32768}"
    --temperature "${EVAL_TEMPERATURE:-1.0}"
    --top_p "${EVAL_TOP_P:-1.0}"
    --pi_trace_field "${PI_TRACE_FIELD:-qwen32b_pi_trace}"
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-1}"
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.9}"
  )
  if [ -n "${CHECKPOINT_PATH}" ]; then
    args+=(--checkpoint "${CHECKPOINT_PATH}")
  fi
  if [ -n "${ADAPTER_PATH}" ]; then
    args+=(--adapter_path "${ADAPTER_PATH}")
  fi
  if [ -n "${MAX_ROWS:-}" ]; then
    args+=(--max_rows "${MAX_ROWS}")
  fi
  case "${EVAL_ENABLE_THINKING:-1}" in
    1|true|True|TRUE|yes|Yes|YES)
      args+=(--enable_thinking)
      ;;
    0|false|False|FALSE|no|No|NO)
      args+=(--no-enable_thinking)
      ;;
    *)
      echo "EVAL_ENABLE_THINKING must be a boolean; got ${EVAL_ENABLE_THINKING}" >&2
      exit 1
      ;;
  esac
  # Named as in the training launchers, so that one exported value keeps training and
  # evaluation on the same student prompt. Unset follows the thinking mode: with thinking
  # enabled, the default, it is the plain student prompt of the Qwen3 students.
  # Qwen3-4B-Instruct sets EMIT_THINK_PROMPTS=True and keeps thinking enabled: its chat
  # template opens the reasoning block, and the last line of its prompt is the instruction
  # to reason inside the <think> tags.
  if [ -n "${EMIT_THINK_PROMPTS:-}" ]; then
    args+=(--emit_think "${EMIT_THINK_PROMPTS}")
  fi
  # The evaluation protocol is the sampled passes only; the temperature-0 pass is opt-in.
  case "${EVAL_GREEDY_PASS:-0}" in
    1|true|True|TRUE|yes|Yes|YES)
      ;;
    0|false|False|FALSE|no|No|NO)
      args+=(--skip_greedy)
      ;;
    *)
      echo "EVAL_GREEDY_PASS must be a boolean; got ${EVAL_GREEDY_PASS}" >&2
      exit 1
      ;;
  esac

  echo "Running official eval: dataset=${dataset}, output=${output_json}" >&2
  (
    set -o pipefail
    "${BSD_TRAIN_PYTHON}" "${BSD_REPO_ROOT}/tools/bsd/evaluate_checkpoints.py" "${args[@]}" "$@" 2>&1 | tee "${log_file}"
  )
  outputs+=("${output_json}")
done

"${BSD_TRAIN_PYTHON}" "${BSD_REPO_ROOT}/tools/bsd/aggregate_results.py" \
  --input "${outputs[@]}" \
  --output_csv "${OUTPUT_ROOT}/summary.csv" \
  --output_md "${OUTPUT_ROOT}/summary.md"

echo "Wrote ${OUTPUT_ROOT}/summary.csv" >&2
echo "Wrote ${OUTPUT_ROOT}/summary.md" >&2
