#!/usr/bin/env bash
# Shared helper: resolve a Hugging Face hub id to its local snapshot directory
# when running offline. transformers 4.57.3 crashes on hub-id model names under
# HF_HUB_OFFLINE=1 (_patch_mistral_regex queries the hub API unguarded), so
# offline clusters must receive local paths. Online runtimes return the hub id
# unchanged, as does an offline runtime with no local snapshot, so error
# messages stay meaningful. HF_HUB_OFFLINE is read as huggingface_hub reads it:
# 1, true, yes and on in any letter case mean offline. A missing snapshot never
# makes the function fail, so it is safe to call directly (not only inside
# $(...)) under `set -e` and `set -o pipefail`.

bsd_model_path() {
  local hub_id="$1"
  case "${HF_HUB_OFFLINE:-0}" in
    1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn]) ;;
    *)
      echo "${hub_id}"
      return 0
      ;;
  esac
  local cache_dir="${HF_HOME:-${HOME}/.cache/huggingface}/hub/models--${hub_id//\//--}"
  local snap="" ref=""
  if [ -f "${cache_dir}/refs/main" ]; then
    ref="$(cat "${cache_dir}/refs/main" 2>/dev/null || true)"
    if [ -n "${ref}" ]; then
      snap="${cache_dir}/snapshots/${ref}"
    fi
  fi
  if [ ! -d "${snap}" ]; then
    # Without a snapshot directory the listing fails; that must not abort the caller.
    snap="$(ls -d "${cache_dir}"/snapshots/*/ 2>/dev/null | head -1 || true)"
    snap="${snap%/}"
  fi
  if [ -n "${snap}" ] && [ -d "${snap}" ]; then
    echo "${snap}"
  else
    echo "${hub_id}"
  fi
}
