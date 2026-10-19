#!/usr/bin/env bash
# Build the fixed multi-answer teacher index that the ROPD baseline reads during
# training (run_ropd_multi.sh). It calls the closed API teacher once per
# answer; credentials are read from the environment or from an untracked .env file.
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

TRAIN_FILE="${TRAIN_FILE:?TRAIN_FILE is required: the training parquet of the teacher-student pair; see the README section \"Data preparation\"}"
ROPD_TEACHER_INDEX_PATH="${ROPD_TEACHER_INDEX_PATH:-${BSD_DATA_ROOT}/datasets/ropd/teacher_index.jsonl}"
ROPD_PROVIDER_SPEC="${ROPD_PROVIDER_SPEC:-${BSD_REPO_ROOT}/recipes/bsd/configs/ropd_judge_providers.yaml}"
ROPD_TEACHER_ANSWER_COUNT="${ROPD_TEACHER_ANSWER_COUNT:-4}"
ROPD_INDEX_MAX_CONCURRENCY="${ROPD_INDEX_MAX_CONCURRENCY:-4}"
# The index key is a hash of the student prompt, so the build has to render it the way the
# consuming run will. The defaults match run_ropd_multi.sh and select the plain
# student prompt of the Qwen3 students. For Qwen3-4B-Instruct pass EMIT_THINK_PROMPTS=True
# to this script and to the training run; STUDENT_THINKING keeps its default.
STUDENT_THINKING="${STUDENT_THINKING:-true}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-${CLOSED_TEACHER_API_KEY:-}}"
# Export the base URL only when one is configured: an empty OPENAI_BASE_URL is not
# "unset" for the OpenAI client and would replace its default endpoint.
ropd_base_url="${OPENAI_BASE_URL:-${CLOSED_TEACHER_BASE_URL:-}}"
if [ -n "${ropd_base_url}" ]; then
  export OPENAI_BASE_URL="${ropd_base_url}"
else
  unset OPENAI_BASE_URL
fi
# No silent default: the teacher model defines the index and its API cost.
if [ -z "${ROPD_TEACHER_MODEL:-${CLOSED_TEACHER_MODEL:-}}" ]; then
  echo "Set CLOSED_TEACHER_MODEL (or ROPD_TEACHER_MODEL) before building the ROPD teacher index." >&2
  exit 1
fi
export ROPD_TEACHER_MODEL="${ROPD_TEACHER_MODEL:-${CLOSED_TEACHER_MODEL}}"
# The provider specification resolves the model id of all three roles when it is loaded,
# so the rubricator and the verifier need one here although this build queries only the
# teacher. They do not enter the teacher fingerprint that is stored in the index.
export ROPD_RUBRICATOR_MODEL="${ROPD_RUBRICATOR_MODEL:-${CLOSED_TEACHER_MODEL:-${ROPD_TEACHER_MODEL}}}"
export ROPD_VERIFIER_MODEL="${ROPD_VERIFIER_MODEL:-${CLOSED_TEACHER_MODEL:-${ROPD_TEACHER_MODEL}}}"
if [ -z "${OPENAI_API_KEY}" ]; then
  echo "Set OPENAI_API_KEY or CLOSED_TEACHER_API_KEY before building the ROPD teacher index." >&2
  exit 1
fi

args=(
  --input_parquet "${TRAIN_FILE}"
  --output_jsonl "${ROPD_TEACHER_INDEX_PATH}"
  --provider_spec "${ROPD_PROVIDER_SPEC}"
  --teacher_answer_count "${ROPD_TEACHER_ANSWER_COUNT}"
  --max_concurrency "${ROPD_INDEX_MAX_CONCURRENCY}"
  --enable_thinking "${STUDENT_THINKING}"
  --emit_think "${EMIT_THINK_PROMPTS:-}"
)
if [ "${ROPD_INDEX_RESUME:-1}" = "1" ]; then
  args+=(--resume)
fi
if [ -n "${MAX_ROWS:-}" ]; then
  args+=(--max_rows "${MAX_ROWS}")
fi

"${BSD_TRAIN_PYTHON}" "${BSD_REPO_ROOT}/tools/bsd/build_ropd_teacher_index.py" "${args[@]}"
