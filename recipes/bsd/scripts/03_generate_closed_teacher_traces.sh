#!/usr/bin/env bash
# Generate the traces of the closed-API external teacher through an OpenAI-compatible
# chat-completions endpoint: one generation per problem from the same plain prompt as the
# open teachers, reasoning effort medium, at most 8,192 completion tokens, and the
# provider's default sampling. The response is stored as the endpoint returns it; a
# response with a wrong final answer is kept and flagged. The run is resumable.
# The teacher hides its reasoning, so the trace is built from this response in a second
# step: jsonl_to_verl_parquet.py and build_teacher_sft_parquet.py enclose the derivation
# in a reasoning block (<think> and </think>) and place the boxed final answer after the
# block, by default for a response that they read from closed_teacher_output
# (tools/bsd/closed_teacher_trace.py). --no-cot_structure turns it off in both tools.
#
# CLOSED_TEACHER_BASE_URL, CLOSED_TEACHER_API_KEY and CLOSED_TEACHER_MODEL come from the
# environment or from the untracked .env file in the repository root (BSD_DOTENV_PATH
# selects another file).
#
# INPUT_JSONL is required. It is the problem file: one JSON object per row with the problem
# text and the gold answer; rows are keyed by their uid, or by a hash of the problem text
# when they carry none (README section "Data preparation"). OUTPUT_JSONL defaults to
# ${BSD_DATA_ROOT}/datasets/closed_teacher_luna_2400/luna_trace_2400_generated.jsonl.
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
source "${SCRIPT_DIR}/_env.sh"

INPUT_JSONL="${INPUT_JSONL:?INPUT_JSONL is required: the problem file (JSONL with the problem text and the gold answer per row); see the README section \"Data preparation\"}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${BSD_DATA_ROOT}/datasets/closed_teacher_luna_2400/luna_trace_2400_generated.jsonl}"

args=(
  --input_jsonl "${INPUT_JSONL}"
  --output_jsonl "${OUTPUT_JSONL}"
  # Reasoning effort medium unless CLOSED_TEACHER_REASONING_EFFORT overrides it. An empty
  # value omits the parameter, for endpoints that do not accept it.
  --reasoning_effort "${CLOSED_TEACHER_REASONING_EFFORT-medium}"
)
if [ -n "${MAX_ROWS:-}" ]; then
  args+=(--max_rows "${MAX_ROWS}")
fi

"${BSD_TRAIN_PYTHON}" "${REPO_ROOT}/tools/bsd/generate_closed_teacher_data.py" \
  "${args[@]}" \
  "$@"
