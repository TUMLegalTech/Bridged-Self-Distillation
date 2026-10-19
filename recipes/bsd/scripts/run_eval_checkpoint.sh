#!/usr/bin/env bash
# Evaluate one model or checkpoint on one evaluation file, for example the development set
# that selects the checkpoint. The defaults are the evaluation parameters of the paper:
# 4 samples per problem, seeds 0-3, thinking enabled, temperature 1.0, top-p 1.0, at most
# 32,768 new tokens.
#
# EVAL_FILE is required (parquet or JSONL; every row needs the problem text and the gold
# answer). MODEL_PATH is the base model, CHECKPOINT_PATH a training checkpoint whose LoRA
# adapter is applied to it. The result is written to
# ${BSD_DATA_ROOT}/outputs/eval/<RUN_NAME>/<CHECKPOINT_NAME>.json (OUTPUT_JSON overrides it).
# For a checkpoint .../<run>/global_step_<N>[/actor] the two names default to <run> and
# global_step_<N>; without a checkpoint both default to the basename of MODEL_PATH.
# EVAL_BACKEND selects the generation backend, vllm (default) or hf.
# Only the sampled passes run. EVAL_GREEDY_PASS=1 (default 0) adds one temperature-0 pass,
# reported in the separate greedy_* fields, which are null without it.
# Qwen3-4B-Instruct is evaluated with thinking enabled and EMIT_THINK_PROMPTS=True.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

EVAL_FILE="${EVAL_FILE:?Set EVAL_FILE to the evaluation file (parquet or JSONL), for example the development set}"
MODEL_PATH="${MODEL_PATH:-$(bsd_model_path Qwen/Qwen3-4B)}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
if [ -n "${CHECKPOINT_PATH}" ]; then
  # Name the result after the run and the step. The checkpoint directory of every online
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
OUTPUT_JSON="${OUTPUT_JSON:-${BSD_DATA_ROOT}/outputs/eval/${RUN_NAME}/${CHECKPOINT_NAME}.json}"

args=(
  --eval_file "${EVAL_FILE}"
  --model "${MODEL_PATH}"
  --output_json "${OUTPUT_JSON}"
  --run "${RUN_NAME}"
  --checkpoint_name "${CHECKPOINT_NAME}"
  --backend "${EVAL_BACKEND:-vllm}"
  --num_samples "${NUM_SAMPLES:-4}"
  --seeds "${EVAL_SEEDS:-0,1,2,3}"
  --max_new_tokens "${MAX_NEW_TOKENS:-32768}"
  --temperature "${EVAL_TEMPERATURE:-1.0}"
  --top_p "${EVAL_TOP_P:-1.0}"
  --pi_trace_field "${PI_TRACE_FIELD:-qwen32b_pi_trace}"
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

"${BSD_TRAIN_PYTHON}" "${BSD_REPO_ROOT}/tools/bsd/evaluate_checkpoints.py" "${args[@]}" "$@"
