#!/usr/bin/env bash
# Aggregate the result JSON files written by run_eval_checkpoint.sh
# (${BSD_DATA_ROOT}/outputs/eval/<run>/<checkpoint>.json; RESULTS_DIR overrides the
# directory) into one CSV and one Markdown table under ${BSD_DATA_ROOT}/outputs/reports
# (OUTPUT_DIR overrides it).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

RESULTS_DIR="${RESULTS_DIR:-${BSD_DATA_ROOT}/outputs/eval}"
OUTPUT_DIR="${OUTPUT_DIR:-${BSD_DATA_ROOT}/outputs/reports}"
mkdir -p "${OUTPUT_DIR}"

mapfile -t RESULT_FILES < <(find "${RESULTS_DIR}" -maxdepth 2 -type f -name '*.json' | sort)
if [ "${#RESULT_FILES[@]}" -eq 0 ]; then
  echo "No evaluation JSON files found under ${RESULTS_DIR}." >&2
  echo "Run checkpoint evaluation first, then re-run this aggregation script." >&2
  exit 1
fi

"${BSD_TRAIN_PYTHON}" "${BSD_REPO_ROOT}/tools/bsd/aggregate_results.py" \
  --input "${RESULT_FILES[@]}" \
  --output_csv "${OUTPUT_DIR}/bridged_self_distillation_eval_summary.csv" \
  --output_md "${OUTPUT_DIR}/bridged_self_distillation_eval_summary.md"

echo "Wrote ${OUTPUT_DIR}/bridged_self_distillation_eval_summary.csv" >&2
echo "Wrote ${OUTPUT_DIR}/bridged_self_distillation_eval_summary.md" >&2
