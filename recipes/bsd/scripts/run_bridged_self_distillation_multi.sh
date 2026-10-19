#!/usr/bin/env bash
# Bridged Self-Distillation (BSD), multi-GPU: the entry point of the method.
#
# The defaults of the base launcher run_bsd_multi.sh are the BSD configuration of
# the paper (16 problems x 8 rollouts per update, 150 updates, lambda 0.5 decayed linearly
# to 0 over training, clipping range 0.2, self-teacher synchronised every 10 updates, the
# weight on unsuccessful rollouts only), so this wrapper only names the method and
# delegates. EXPERIMENT_NAME, TRAIN_FILE and VAL_FILE are required; PI_TRACE_FIELD selects
# the trace field of the external teacher. See run_bsd_multi.sh for every variable.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export METHOD_NAME="${METHOD_NAME:-bridged_self_distillation}"

exec bash "${SCRIPT_DIR}/run_bsd_multi.sh" "$@"
