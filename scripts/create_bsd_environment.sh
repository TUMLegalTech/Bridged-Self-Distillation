#!/usr/bin/env bash
# Create one isolated Bridged Self-Distillation runtime without modifying an existing environment.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${BSD_ENV_ROOT:-${PROJECT_DIR}/workspace/conda_envs}"
CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
ROLE="${1:-}"

case "${ROLE}" in
  generation)
    PREFIX="${BSD_GENERATION_ENV_PREFIX:-${ENV_ROOT}/BSD-generation}"
    REQUIREMENTS="${PROJECT_DIR}/recipes/bsd/environments/generation.txt"
    RUNTIME_ROLE="generation"
    ;;
  training)
    PREFIX="${BSD_TRAIN_ENV_PREFIX:-${ENV_ROOT}/BSD}"
    REQUIREMENTS="${PROJECT_DIR}/recipes/bsd/environments/training.txt"
    RUNTIME_ROLE="training"
    ;;
  *)
    echo "Usage: $0 {training|generation}" >&2
    exit 2
    ;;
esac

if [ -z "${CONDA_EXE}" ] || ! command -v "${CONDA_EXE}" >/dev/null 2>&1; then
  echo "conda was not found: put it on PATH or set CONDA_EXE to the conda executable." >&2
  exit 1
fi

if [ -e "${PREFIX}" ]; then
  echo "Refusing to replace existing environment: ${PREFIX}" >&2
  exit 1
fi

mkdir -p "$(dirname "${PREFIX}")"
"${CONDA_EXE}" create --yes --prefix "${PREFIX}" python=3.11 pip
"${PREFIX}/bin/python" -m pip install --requirement "${REQUIREMENTS}"
if [ "${RUNTIME_ROLE}" = "training" ]; then
  "${PREFIX}/bin/python" -m pip install --no-deps --editable "${PROJECT_DIR}"
fi
"${PREFIX}/bin/python" -m pip check
PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PREFIX}/bin/python" "${PROJECT_DIR}/tools/bsd/runtime.py" \
  --role "${RUNTIME_ROLE}" \
  --expected-prefix "${PREFIX}"

echo "Created Bridged Self-Distillation ${RUNTIME_ROLE} environment at ${PREFIX}"
