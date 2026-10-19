#!/usr/bin/env bash
# Download AIME 2024, AIME 2025 and HMMT 2025 from the Hugging Face Hub and write
# <dataset>.jsonl, <dataset>.parquet and manifest.json to
# ${BSD_DATA_ROOT}/datasets/eval/official_math (OFFICIAL_EVAL_DATA_DIR overrides it).
# The JSONL files are also the input of the contamination filter in 01_prepare_training_problems.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

OUTPUT_DIR="${OFFICIAL_EVAL_DATA_DIR:-${BSD_DATA_ROOT}/datasets/eval/official_math}"
PROMPT_STYLE="${OFFICIAL_EVAL_PROMPT_STYLE:-opsd}"

"${BSD_TRAIN_PYTHON}" "${BSD_REPO_ROOT}/tools/bsd/prepare_official_math_eval.py" \
  --output_dir "${OUTPUT_DIR}" \
  --prompt_style "${PROMPT_STYLE}" \
  "$@"
