#!/usr/bin/env bash
# Generate the traces of a local external teacher with one isolated vLLM worker per GPU in
# CUDA_VISIBLE_DEVICES (default 0,1,2,3). One trace is generated per problem. A trace whose
# final answer the verifier rejects is kept and flagged (<teacher>_pi_verified is false);
# worker_<i>/raw.jsonl holds every row and worker_<i>/verified.jsonl the verified ones.
#
# The defaults are the Nemotron-3-30B-A3B settings: thinking enabled, the sampling preset
# REASONING of tools/bsd/generate_local_traces.py (temperature 1.0, top-p 1.0, top-k
# disabled), at most 8,192 new tokens, and the plain prompt
# recipes/bsd/prompts/pi_generation.jinja.
#
#   INPUT_JSONL=/path/to/problems.jsonl \
#     bash recipes/bsd/scripts/02_generate_teacher_traces.sh
#
# INPUT_JSONL is required. It holds one JSON object per problem with an identifier (uid or
# id), the problem text (problem) and the gold answer (gold_answer), for example a subset
# of the all_clean.jsonl written by 01_prepare_training_problems.sh.
#
# The Qwen3-32B traces are generated with the same script and that teacher's sampling
# preset A (temperature 0.6, top-p 0.95):
#
#   INPUT_JSONL=/path/to/problems.jsonl \
#   MODEL_PATH=Qwen/Qwen3-32B \
#   TRACE_FIELD=qwen32b_pi_trace \
#   OUTPUT_DIR=/path/to/qwen32b_generation \
#   SAMPLING_CONFIG_NAME=A \
#     bash recipes/bsd/scripts/02_generate_teacher_traces.sh
#
# MODEL_PATH is a hub id or a local directory. TRACE_FIELD is the output key; it names the
# teacher and must end in _pi_trace. Every teacher needs its own OUTPUT_DIR. Rerunning the
# same command resumes an interrupted job. SAMPLING_CONFIG_NAME selects a named sampling
# preset of generate_local_traces.py (--config_name): REASONING (default) or A. The launcher
# passes the preset and no explicit sampling flag; the tool rejects --temperature, --top_p
# or --top_k next to a preset. TRACE_MAX_NEW_TOKENS (default 8192), MAX_MODEL_LEN,
# BATCH_SIZE, MAX_NUM_SEQS, GPU_MEMORY_UTILIZATION and SEED override the remaining
# settings.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BSD_GENERATION_MULTI=1
source "${SCRIPT_DIR}/_env_generation.sh"

DEFAULT_MODEL_PATH="$(bsd_model_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)"
DEFAULT_TRACE_FIELD=nemotron_pi_trace
DEFAULT_OUTPUT_DIR="${BSD_DATA_ROOT}/datasets/teacher_traces/nemotron_generation"

INPUT_JSONL="${INPUT_JSONL:?Set INPUT_JSONL to the problem file (JSONL with uid or id, problem and gold_answer per row)}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
# The output key must name the teacher that produced the trace, so TRACE_FIELD travels with
# MODEL_PATH and is never assumed for a MODEL_PATH other than the default one: it then has
# to be given explicitly (the default key is still accepted when it is stated, as for a
# local copy of the default teacher). The two further checks keep a second teacher out of
# the first one's output directory and reject an output key passed as --trace_field.
if [ "${MODEL_PATH}" != "${DEFAULT_MODEL_PATH}" ] && [ -z "${TRACE_FIELD:-}" ]; then
  echo "MODEL_PATH is not the default teacher; set TRACE_FIELD to name the teacher that produced the traces (for example qwen32b_pi_trace)." >&2
  exit 1
fi
TRACE_FIELD="${TRACE_FIELD:-${DEFAULT_TRACE_FIELD}}"
if [ "${TRACE_FIELD}" != "${DEFAULT_TRACE_FIELD}" ] && [ "${OUTPUT_DIR}" = "${DEFAULT_OUTPUT_DIR}" ]; then
  # generate_local_traces.py resumes on uid alone and rewrites the manifest with the new
  # model, so a second teacher aimed at the first teacher's directory would skip every
  # row as done and label the old traces with the new model.
  echo "TRACE_FIELD=${TRACE_FIELD} must not write into the default Nemotron OUTPUT_DIR; set OUTPUT_DIR." >&2
  exit 1
fi
for bsd_arg in "$@"; do
  if [ "${bsd_arg}" = "--trace_field" ]; then
    # argparse keeps the last occurrence, so this would win over TRACE_FIELD silently.
    echo "Pass the output key through TRACE_FIELD, not --trace_field." >&2
    exit 1
  fi
done
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  # Reached only under SLURM, where _env_generation.sh leaves the variable to the scheduler.
  echo "CUDA_VISIBLE_DEVICES is not set; under SLURM request GPUs for the job so that the scheduler sets it." >&2
  exit 1
fi
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_WORKERS="${#GPUS[@]}"
mkdir -p "${OUTPUT_DIR}/logs"

pids=()
for worker_id in "${!GPUS[@]}"; do
  gpu="${GPUS[${worker_id}]}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${BSD_GENERATION_PYTHON}" "${BSD_REPO_ROOT}/tools/bsd/generate_local_traces.py" \
      --input_jsonl "${INPUT_JSONL}" \
      --output_dir "${OUTPUT_DIR}" \
      --model "${MODEL_PATH}" \
      --trace_field "${TRACE_FIELD}" \
      --prompt_template pi_generation \
      --thinking_mode on \
      --keep_unverified_trace \
      --mirror_extra_info \
      --worker_id "${worker_id}" \
      --num_workers "${NUM_WORKERS}" \
      --config_name "${SAMPLING_CONFIG_NAME:-REASONING}" \
      --batch_size "${BATCH_SIZE:-4}" \
      --max_new_tokens "${TRACE_MAX_NEW_TOKENS:-8192}" \
      --max_model_len "${MAX_MODEL_LEN:-24576}" \
      --max_num_seqs "${MAX_NUM_SEQS:-4}" \
      --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
      --seed "${SEED:-0}" \
      "$@" \
      >"${OUTPUT_DIR}/logs/worker_${worker_id}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  echo "At least one generation worker failed; inspect ${OUTPUT_DIR}/logs." >&2
  exit 1
fi

echo "Trace generation completed: ${OUTPUT_DIR}"
