#!/usr/bin/env bash
# Shared environment defaults for isolated offline generation jobs.
set -euo pipefail

export BSD_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export BSD_DATA_ROOT="${BSD_DATA_ROOT:-${BSD_REPO_ROOT}/workspace}"
export BSD_ENV_ROOT="${BSD_ENV_ROOT:-${BSD_REPO_ROOT}/workspace/conda_envs}"
export BSD_GENERATION_ENV_PREFIX="${BSD_GENERATION_ENV_PREFIX:-${BSD_ENV_ROOT}/BSD-generation}"
export BSD_GENERATION_PYTHON="${BSD_GENERATION_PYTHON:-${BSD_GENERATION_ENV_PREFIX}/bin/python}"
if [ ! -x "${BSD_GENERATION_PYTHON}" ]; then
  echo "Missing Bridged Self-Distillation generation Python: ${BSD_GENERATION_PYTHON}" >&2
  exit 1
fi
source "$(dirname "${BASH_SOURCE[0]}")/_hf_local.sh"
export PYTHONPATH="${BSD_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${BSD_GENERATION_PYTHON}" "${BSD_REPO_ROOT}/tools/bsd/runtime.py" \
  --role generation \
  --expected-prefix "${BSD_GENERATION_ENV_PREFIX}" >/dev/null

if [ -n "${SLURM_JOB_ID:-}" ]; then
  # Under SLURM the scheduler controls GPU visibility (cgroup-scoped GPUs
  # appear as 0..N-1); never override or validate CUDA_VISIBLE_DEVICES here.
  :
else
  # A GPU set reaches this file as free text and is then split into one worker per
  # entry by the multi-GPU generation launcher, so a malformed set is not a typo -- it
  # is two vLLM engines on one card, or a worker pinned to a device that does not
  # exist. Validate shape and uniqueness here.
  bsd_validate_gpu_set() {
    local set_value="$1" seen="" gpu
    case "${set_value}" in
      *[!0-9,]*|,*|*,|*,,*|"") echo "GPU set must be comma-separated non-negative integers; got '${set_value}'" >&2; return 1 ;;
    esac
    for gpu in ${set_value//,/ }; do
      case " ${seen} " in *" ${gpu} "*) echo "GPU set repeats device ${gpu}: '${set_value}'" >&2; return 1 ;; esac
      seen="${seen} ${gpu}"
    done
    return 0
  }

  case "${BSD_GENERATION_MULTI:-0}" in
    1|true|TRUE|yes|YES)
      export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
      bsd_validate_gpu_set "${CUDA_VISIBLE_DEVICES}" || exit 1
      ;;
    *)
      export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
      ;;
  esac
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# vLLM engine cores must not fork after the parent touches CUDA; matches the
# spawn setting already used by the multi-GPU recipes and evaluate_checkpoints.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_HOME="${HF_HOME:-${BSD_DATA_ROOT}/cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${BSD_DATA_ROOT}/cache/vllm}"
mkdir -p "${HF_HOME}" "${VLLM_CACHE_ROOT}"
