#!/usr/bin/env bash
# Shared environment defaults for multi-GPU Bridged Self-Distillation runs.
set -euo pipefail

export BSD_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export BSD_DATA_ROOT="${BSD_DATA_ROOT:-${BSD_REPO_ROOT}/workspace}"
export BSD_ENV_ROOT="${BSD_ENV_ROOT:-${BSD_REPO_ROOT}/workspace/conda_envs}"
export BSD_TRAIN_ENV_PREFIX="${BSD_TRAIN_ENV_PREFIX:-${BSD_ENV_ROOT}/BSD}"
export BSD_TRAIN_PYTHON="${BSD_TRAIN_PYTHON:-${BSD_TRAIN_ENV_PREFIX}/bin/python}"
if [ ! -x "${BSD_TRAIN_PYTHON}" ]; then
  echo "Missing Bridged Self-Distillation training Python: ${BSD_TRAIN_PYTHON}" >&2
  exit 1
fi
source "$(dirname "${BASH_SOURCE[0]}")/_hf_local.sh"
export PYTHONPATH="${BSD_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${BSD_TRAIN_PYTHON}" "${BSD_REPO_ROOT}/tools/bsd/runtime.py" \
  --role training \
  --expected-prefix "${BSD_TRAIN_ENV_PREFIX}" >/dev/null

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export BSD_VISIBLE_GPU_COUNT="${BSD_VISIBLE_GPU_COUNT:-$("${BSD_TRAIN_PYTHON}" - <<'PY'
import os

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(len([item for item in visible.split(",") if item.strip()]) if visible else 1)
PY
)}"
export NNODES="${NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-${BSD_VISIBLE_GPU_COUNT}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${NGPUS_PER_NODE}}"

if [ "${NGPUS_PER_NODE}" -lt 2 ]; then
  echo "Use single-card scripts for NGPUS_PER_NODE=1; multi scripts expect >=2 GPUs." >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [ -n "${BSD_FORCE_PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
  export PYTORCH_CUDA_ALLOC_CONF="${BSD_FORCE_PYTORCH_CUDA_ALLOC_CONF}"
else
  unset PYTORCH_CUDA_ALLOC_CONF
fi
export VERL_FILE_LOGGER_ROOT="${VERL_FILE_LOGGER_ROOT:-${BSD_DATA_ROOT}/outputs/file_logger}"
mkdir -p "${VERL_FILE_LOGGER_ROOT}"

# Multi-GPU resource pools should use Ray's normal CUDA isolation.
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES

if [ -z "${TRAINER_LOGGER:-}" ]; then
  # Weights & Biases is opt-in: only an explicit BSD_ENABLE_WANDB=1 adds it.
  case "${BSD_ENABLE_WANDB:-0}" in
    1|true|True|TRUE|yes|Yes|YES)
      export TRAINER_LOGGER='["console","wandb"]'
      ;;
    *)
      export TRAINER_LOGGER='["console"]'
      ;;
  esac
fi
if [[ "${TRAINER_LOGGER}" == *wandb* ]]; then
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_DIR="${WANDB_DIR:-${BSD_DATA_ROOT}/outputs/wandb}"
  mkdir -p "${WANDB_DIR}"
fi

write_bsd_run_manifest() {
  local experiment_name="$1"
  shift
  local manifest_path="${RUN_MANIFEST:-${BSD_DATA_ROOT}/outputs/${experiment_name}/run_manifest.json}"
  local manifest_args=(
    --output "${manifest_path}"
    --seed "${SEED:-42}"
    --config_kv "experiment_name=${experiment_name}"
    --config_kv "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    --config_kv "nnodes=${NNODES}"
    --config_kv "nproc_per_node=${NPROC_PER_NODE}"
    --config_kv "ngpus_per_node=${NGPUS_PER_NODE}"
    --config_kv "multi_gpu=true"
    --config_kv "trainer_logger=${TRAINER_LOGGER}"
    # The allocator settings decide whether a run pays the expandable-segment teardown
    # cost, so its step time is not interpretable without them; recorded raw.
    --config_kv "disable_expandable_segments=${VERL_DISABLE_EXPANDABLE_SEGMENTS:-}"
    --config_kv "pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF:-}"
  )
  if [ -n "${SPLIT_MANIFEST:-}" ]; then
    manifest_args+=(--dataset_manifest "${SPLIT_MANIFEST}")
  fi
  "${BSD_TRAIN_PYTHON}" "${BSD_REPO_ROOT}/tools/bsd/write_run_manifest.py" "${manifest_args[@]}" "$@"
  echo "Wrote run manifest: ${manifest_path}" >&2
}
