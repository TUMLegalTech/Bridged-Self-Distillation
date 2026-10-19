#!/usr/bin/env bash
# Normalise the mathematical-reasoning subset of OpenThoughts into the training pool
# ${BSD_DATA_ROOT}/datasets/opsd_clean/all_clean.jsonl. Problems that overlap an
# evaluation problem are moved to contamination_quarantine.jsonl, so the evaluation files
# must exist first: run 00_prepare_eval_benchmarks.sh and pass its JSONL files through
# EVAL_JSONL (colon-separated) or through repeated --eval_jsonl arguments.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

args=("$@")
if [ -n "${EVAL_JSONL:-}" ]; then
  IFS=':' read -r -a eval_jsonl_paths <<< "${EVAL_JSONL}"
  for eval_jsonl_path in "${eval_jsonl_paths[@]}"; do
    args+=(--eval_jsonl "${eval_jsonl_path}")
  done
fi

has_eval_jsonl=0
# ${args[@]+...}: bash older than 4.4 rejects "${args[@]}" of an empty array under set -u.
for arg in ${args[@]+"${args[@]}"}; do
  if [[ "${arg}" == "--eval_jsonl"* ]]; then
    has_eval_jsonl=1
    break
  fi
done
if [ "${has_eval_jsonl}" -ne 1 ] && [ "${ALLOW_MISSING_EVAL_CONTAMINATION:-0}" != "1" ]; then
  echo "Refusing to prepare the training pool without the evaluation files of the contamination filter." >&2
  echo "Pass --eval_jsonl /path/to/eval.jsonl, set EVAL_JSONL=path1:path2, or set ALLOW_MISSING_EVAL_CONTAMINATION=1 for local development only." >&2
  exit 1
fi

"${BSD_TRAIN_PYTHON}" "${SCRIPT_DIR}/../../../tools/bsd/prepare_opsd_data.py" \
  --output_dir "${BSD_DATA_ROOT}/datasets/opsd_clean" \
  ${args[@]+"${args[@]}"}
