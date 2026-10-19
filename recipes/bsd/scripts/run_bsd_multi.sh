#!/usr/bin/env bash
# Base launcher of the BSD family, multi-GPU: Bridged Self-Distillation (BSD), RLSD, the
# BSD ablation variants and OPSD. Every knob is an environment variable; the defaults are
# the BSD configuration of the paper: 16 problems x 8 rollouts per update, 150 updates,
# learning rate 1e-6, LoRA rank 32 / alpha 64, lambda 0.5 decayed linearly to 0 over the
# 150 updates, clipping range 0.2, self-teacher synchronised every 10 updates, the weight
# on every response token of unsuccessful rollouts only, and a binary verifier reward.
# Trailing arguments are appended as hydra overrides (hydra keeps the last one).
#
# Entry point: run_bridged_self_distillation_multi.sh (BSD).
set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env_multi.sh"

TRAIN_FILE="${TRAIN_FILE:?TRAIN_FILE is required: path of the training parquet; see the README section \"Data preparation\"}"
VAL_FILE="${VAL_FILE:?VAL_FILE is required: path of the development parquet; see the README section \"Data preparation\"}"
MODEL_PATH="${MODEL_PATH:-$(bsd_model_path Qwen/Qwen3-4B)}"
PROJECT_NAME="${PROJECT_NAME:-bridged_self_distillation}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME is required: a unique run name; it names the checkpoint and output directories}"
METHOD_NAME="${METHOD_NAME:-bridged_self_distillation}"
# Optional JSON manifest of the data split; when set, its path and SHA-256 are recorded
# in the run manifest.
SPLIT_MANIFEST="${SPLIT_MANIFEST:-}"
# RLSD_ENABLED=False turns the weight off: the self-teacher is not scored and the update
# uses the plain verifier advantage, unless one of the optional paths below (guided rescue,
# Bridged OPD, bridged critic) is enabled. The value is recorded in the manifest.
RLSD_ENABLED="${RLSD_ENABLED:-True}"
# Weight w_t = (1 - lambda) + lambda * clip(exp(-delta_t), 1 - eps, 1 + eps). lambda is
# RLSD_LAMBDA, decayed linearly to 0 over RLSD_LAMBDA_DECAY_STEPS updates (0 disables the
# decay) after an optional linear warm-up of RLSD_LAMBDA_WARMUP_STEPS updates; eps is
# RLSD_CLIP_RANGE. The decay length does not follow TOTAL_TRAINING_STEPS: set both when
# the run length changes, so that lambda still reaches 0 at the last update.
RLSD_LAMBDA="${RLSD_LAMBDA:-0.5}"
RLSD_LAMBDA_WARMUP_STEPS="${RLSD_LAMBDA_WARMUP_STEPS:-0}"
RLSD_LAMBDA_DECAY_STEPS="${RLSD_LAMBDA_DECAY_STEPS:-150}"
RLSD_CLIP_RANGE="${RLSD_CLIP_RANGE:-0.2}"
# Routing of the weight. RLSD_NEGATIVE_ONLY=True (BSD) applies it to unsuccessful rollouts
# (A < 0) only and forces the positive branch off. With RLSD_NEGATIVE_ONLY=False the branch
# modes apply (rlsd.py): positive off|rlsd|rlrt, negative off|rlsd. Both branches set to
# rlsd is the ablation variant "BSD (w/ succ.)".
RLSD_NEGATIVE_ONLY="${RLSD_NEGATIVE_ONLY:-True}"
RLSD_POSITIVE_MODE="${RLSD_POSITIVE_MODE:-rlsd}"
RLSD_NEGATIVE_MODE="${RLSD_NEGATIVE_MODE:-rlsd}"
# Score of a boxed-but-wrong answer. 0 (default) keeps the verifier reward binary,
# r in {0, 1}; a positive value adds a format term to the group advantages (see
# recipes/bsd/rewards/math_verify_reward.py).
export BSD_FORMAT_BONUS="${BSD_FORMAT_BONUS:-0}"
# Response tokens that receive the weight. full (default) is every response token of a
# routed rollout; the other masks restrict it, e.g. 4k+ans to the first 4,096 and the last
# 256 response tokens (trainer.py _build_rlsd_token_mask).
RLSD_TOKEN_MASK="${RLSD_TOKEN_MASK:-full}"
# Bridged OPD (trainer.py _apply_bridged_opd), off by default: a token-level reward from the
# trace-conditioned self-teacher, A_t = verifier_coef * A_grpo + coef * clip(log q_t - log p_t, +-clip_abs),
# where the second term is restricted to the rows in scope (failed_completed | all) and p_t is the
# old-policy log-probability. Requires RLSD_ENABLED=False and one optimizer update per step.
# The OPSD ablation of the paper is this path with BRIDGED_OPD_SCOPE=all (every rollout),
# BRIDGED_OPD_VERIFIER_COEF=0 (no verifier reward), BRIDGED_OPD_CLIP_ABS=2.0 (log-ratio clipped at +-2)
# and TEACHER_UPDATE_MODE=base_no_adapter (the frozen base model as self-teacher), together with
# RLSD_ENABLED=False and BRIDGED_OPD_ENABLED=True.
# GRPO_TRUNCATION_AWARE, off by default, drops rollouts truncated by the response budget from the
# verifier advantage and its group statistics (rlsd.py compute_truncation_aware_grpo_advantages);
# usable with or without Bridged OPD.
BRIDGED_OPD_ENABLED="${BRIDGED_OPD_ENABLED:-False}"
BRIDGED_OPD_SCOPE="${BRIDGED_OPD_SCOPE:-failed_completed}"
BRIDGED_OPD_COEF="${BRIDGED_OPD_COEF:-1.0}"
BRIDGED_OPD_VERIFIER_COEF="${BRIDGED_OPD_VERIFIER_COEF:-1.0}"
BRIDGED_OPD_CLIP_ABS="${BRIDGED_OPD_CLIP_ABS:-2.0}"
GRPO_TRUNCATION_AWARE="${GRPO_TRUNCATION_AWARE:-False}"
# Bridged critic (trainer.py _apply_bridged_critic), off by default: the current policy reads the trace as a
# reference solution and judges every completed rollout of a group with both outcomes at absolute token cuts
# (one-token log P(Yes)); the truncation-aware GRPO advantage of unsuccessful rollouts is reweighted toward the
# segments where the calibrated critic value drops (sign and row total unchanged). Requires
# GRPO_TRUNCATION_AWARE=True, with RLSD, Bridged OPD and guided rescue off. The calibration
# V = sigmoid(a + w z + u logit b) has no default: fit it per student and pass BRIDGED_CRITIC_CALIB_A, _W and _U.
BRIDGED_CRITIC_ENABLED="${BRIDGED_CRITIC_ENABLED:-False}"
BRIDGED_CRITIC_BETA="${BRIDGED_CRITIC_BETA:-0.5}"
BRIDGED_CRITIC_CUTS="${BRIDGED_CRITIC_CUTS:-[1000,2000,3000,4000,6000,8000]}"
BRIDGED_CRITIC_MIN_TAIL="${BRIDGED_CRITIC_MIN_TAIL:-64}"
BRIDGED_CRITIC_MAX_PROMPT_LENGTH="${BRIDGED_CRITIC_MAX_PROMPT_LENGTH:-20480}"
BRIDGED_CRITIC_CALIB_A="${BRIDGED_CRITIC_CALIB_A:-}"
BRIDGED_CRITIC_CALIB_W="${BRIDGED_CRITIC_CALIB_W:-}"
BRIDGED_CRITIC_CALIB_U="${BRIDGED_CRITIC_CALIB_U:-}"
# Student term p_t of the guidance: current (default) is the log-probability under the
# policy being updated, old the one recorded before the update.
RLSD_DELTA_STUDENT_LOGP_SOURCE="${RLSD_DELTA_STUDENT_LOGP_SOURCE:-current}"
TEACHER_MAX_PROMPT_LENGTH="${TEACHER_MAX_PROMPT_LENGTH:-12288}"
# Self-teacher: periodic_snapshot (default) is a frozen copy of the student that is
# synchronised every TEACHER_SYNC_INTERVAL updates; base_no_adapter is the frozen base
# model; current_no_grad is the current student without gradient.
TEACHER_UPDATE_MODE="${TEACHER_UPDATE_MODE:-periodic_snapshot}"
TEACHER_SYNC_INTERVAL="${TEACHER_SYNC_INTERVAL:-10}"
# Privileged context read by the self-teacher. PI_SOURCE=trace_field reads the text in
# extra_info.<PI_TRACE_FIELD>; the trace fields are qwen32b_pi_trace (default, the
# shared-tokenizer teacher Qwen3-32B), nemotron_pi_trace (Nemotron-3-30B-A3B) and
# luna_pi_trace (GPT-5.6 Luna). PI_SOURCE=ground_truth reads only
# reward_model.ground_truth (RLSD) and ignores PI_TRACE_FIELD.
PI_SOURCE="${PI_SOURCE:-trace_field}"
PI_TRACE_FIELD="${PI_TRACE_FIELD:-qwen32b_pi_trace}"
# Optional self-teacher prompt template for another form of privileged context: a file name under
# recipes/bsd/prompts that renders {{ problem }} and {{ pi_trace }}. Empty keeps bsd_self_teacher_*.jinja.
# BSD-hint and BSD-guide set it to bsd_self_teacher_hint_nonthink.jinja / bsd_self_teacher_guide_nonthink.jinja
# and point PI_TRACE_FIELD at the hint / guide field of the training parquet.
PI_TEACHER_TEMPLATE="${PI_TEACHER_TEMPLATE:-}"
STUDENT_THINKING="${STUDENT_THINKING:-True}"
TEACHER_THINKING="${TEACHER_THINKING:-True}"
# Tri-state. Empty follows the thinking variables: with thinking on, the default, it selects
# the plain student and self-teacher prompts of the Qwen3 students. Qwen3-4B-Instruct sets
# EMIT_THINK_PROMPTS=True and keeps STUDENT_THINKING=True and TEACHER_THINKING=True: its
# chat template opens the reasoning block, and the last line of both prompts is the
# instruction to reason inside the <think> tags (the *_emit_think.jinja templates).
EMIT_THINK_PROMPTS="${EMIT_THINK_PROMPTS:-}"
# False (default) stops the run when a self-teacher prompt exceeds
# TEACHER_MAX_PROMPT_LENGTH; True shortens the trace, and only the trace, until it fits.
ALLOW_TEACHER_PROMPT_TRUNCATION="${ALLOW_TEACHER_PROMPT_TRUNCATION:-False}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-12288}"
# Guided rescue (verl/trainer/bsd/trainer.py, GuidedRescueConfig), off by default:
# every prompt also gets GUIDED_SAMPLES rollouts sampled under a prompt that carries the
# answer-stripped trace; in a group whose regular rollouts all fail the verifier, guided
# rollouts that pass it (and the length and phrase filters) replace rows of that group.
# Replaced rows keep the trace-free training prompt, and their importance weight becomes
# P(y_t|x) / P(y_t|x, trace), truncated to [GUIDED_RHO_MIN, GUIDED_RHO_MAX]. The guided
# sampling prompt must fit data.max_prompt_length, otherwise the trainer fails instead of
# truncating.
GUIDED_RESCUE_ENABLED="${GUIDED_RESCUE_ENABLED:-False}"
GUIDED_SAMPLES="${GUIDED_SAMPLES:-2}"
GUIDED_RHO_MIN="${GUIDED_RHO_MIN:-0.2}"
GUIDED_RHO_MAX="${GUIDED_RHO_MAX:-1.0}"
GUIDED_MIN_LENGTH_RATIO="${GUIDED_MIN_LENGTH_RATIO:-0.08}"
# Guided reject regex: leave GUIDED_REJECT_REGEX unset for the trainer-side default
# phrase list (DEFAULT_GUIDED_REJECT_REGEX); set it to "" to disable the filter; any
# other value is passed through hydra single-quoted, so it must not contain single quotes.
# Arrays that can be empty are expanded as ${name[@]+"${name[@]}"} below: bash older than
# 4.4 treats a plain "${name[@]}" of an empty array as an unbound variable under set -u.
GUIDED_REGEX_ARGS=()
if [ -n "${GUIDED_REJECT_REGEX+x}" ]; then
  GUIDED_REGEX_ARGS+=("+bsd.guided_reject_regex='${GUIDED_REJECT_REGEX}'")
fi
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
SAVE_FREQ="${SAVE_FREQ:-25}"
# Periodic in-training validation on VAL_FILE, every TEST_FREQ updates and after the last
# one; TEST_FREQ=-1 disables it. It only monitors the run: checkpoints are selected
# afterwards on the development set with the evaluation protocol. Independently of
# TEST_FREQ, verl validates once before the first update unless
# trainer.val_before_train=False is appended as a hydra override.
TEST_FREQ="${TEST_FREQ:-25}"
SCALE_MODE="${SCALE_MODE:-fixed}"
SAVE_LORA_ONLY="${SAVE_LORA_ONLY:-True}"
SEED="${SEED:-43}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-True}"
ACTOR_FSDP_SIZE="${ACTOR_FSDP_SIZE:--1}"
RESUME_MODE="${RESUME_MODE:-disable}"
# Samples per validation problem. The default 4 samples at temperature 1.0 and reports
# mean@4, which is less noisy than a single greedy decode; VAL_ROLLOUT_N=1 validates
# greedily (mean@1, which is not the same quantity as mean@4).
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-4}"
# Sampled validation draws from the same vLLM random stream as the training rollouts
# (rollout.full_determinism is off, so no per-request seed is injected), whereas greedy
# decoding consumes no randomness. A run that validates therefore follows a different
# rollout sequence than the same configuration without validation. The draws come from the
# same distribution, so this is a re-randomisation and not a bias, but training trajectories
# are only comparable between runs that share the validation settings.
case "${VAL_ROLLOUT_N}" in
  ''|*[!0-9]*) echo "VAL_ROLLOUT_N must be a positive integer; got ${VAL_ROLLOUT_N}" >&2; exit 1 ;;
  0) echo "VAL_ROLLOUT_N must be >= 1" >&2; exit 1 ;;
esac
# n=1 decodes greedily; anything above it has to sample, or the n draws are identical and
# mean@N is just a slower mean@1.
if [ "${VAL_ROLLOUT_N}" -eq 1 ]; then
  VAL_DO_SAMPLE=False
  VAL_TEMPERATURE=0
else
  VAL_DO_SAMPLE=True
  VAL_TEMPERATURE=1.0
fi
BSD_TEACHER_ADAPTER=False

for bool_var in STUDENT_THINKING TEACHER_THINKING ALLOW_TEACHER_PROMPT_TRUNCATION RLSD_ENABLED RLSD_NEGATIVE_ONLY SAVE_LORA_ONLY ROLLOUT_FREE_CACHE_ENGINE GUIDED_RESCUE_ENABLED BRIDGED_OPD_ENABLED GRPO_TRUNCATION_AWARE BRIDGED_CRITIC_ENABLED; do
  case "${!bool_var}" in
    1|true|True|TRUE|yes|Yes|YES) printf -v "${bool_var}" '%s' True ;;
    0|false|False|FALSE|no|No|NO) printf -v "${bool_var}" '%s' False ;;
    *) echo "${bool_var} must be a boolean; got ${!bool_var}" >&2; exit 1 ;;
  esac
done
case "${EMIT_THINK_PROMPTS}" in
  "") ;;
  1|true|True|TRUE|yes|Yes|YES) EMIT_THINK_PROMPTS=True ;;
  0|false|False|FALSE|no|No|NO) EMIT_THINK_PROMPTS=False ;;
  *) echo "EMIT_THINK_PROMPTS must be a boolean or empty; got ${EMIT_THINK_PROMPTS}" >&2; exit 1 ;;
esac
# Hydra needs an explicit null for "not specified"; an empty override value is ambiguous.
EMIT_THINK_OVERRIDE="${EMIT_THINK_PROMPTS:-null}"
if ! [[ "${ACTOR_FSDP_SIZE}" =~ ^-?[0-9]+$ ]] || \
  [ "${ACTOR_FSDP_SIZE}" -eq 0 ] || [ "${ACTOR_FSDP_SIZE}" -lt -1 ]; then
  echo "ACTOR_FSDP_SIZE must be -1 or a positive integer; got ${ACTOR_FSDP_SIZE}" >&2
  exit 1
fi

case "${TEACHER_UPDATE_MODE}" in
  base_no_adapter|current_no_grad)
    ;;
  periodic_snapshot)
    BSD_TEACHER_ADAPTER=True
    ;;
  *)
    echo "TEACHER_UPDATE_MODE must be base_no_adapter, current_no_grad, or periodic_snapshot; got ${TEACHER_UPDATE_MODE}" >&2
    exit 1
    ;;
esac
case "${PI_SOURCE}" in
  trace_field)
    PI_FIELD_PATH="extra_info.${PI_TRACE_FIELD}"
    PI_MANIFEST_ARGS=(--config_kv "pi_trace_field=${PI_TRACE_FIELD}")
    PI_HYDRA_ARGS=(+bsd.pi_source="${PI_SOURCE}" +bsd.pi_trace_field="${PI_TRACE_FIELD}")
    if [ -n "${PI_TEACHER_TEMPLATE}" ]; then
      if [ ! -f "${SCRIPT_DIR}/../prompts/${PI_TEACHER_TEMPLATE}" ]; then
        echo "PI_TEACHER_TEMPLATE=${PI_TEACHER_TEMPLATE} is not a file under recipes/bsd/prompts" >&2
        exit 1
      fi
      PI_MANIFEST_ARGS+=(--config_kv "pi_teacher_template=${PI_TEACHER_TEMPLATE}")
      PI_HYDRA_ARGS+=(+bsd.pi_teacher_template="${PI_TEACHER_TEMPLATE}")
    fi
    ;;
  ground_truth)
    PI_FIELD_PATH="reward_model.ground_truth"
    PI_MANIFEST_ARGS=()
    PI_HYDRA_ARGS=(+bsd.pi_source="${PI_SOURCE}")
    ;;
  *)
    echo "PI_SOURCE must be trace_field or ground_truth; got ${PI_SOURCE}" >&2
    exit 1
    ;;
esac
if [ "${SCALE_MODE}" = "linear" ]; then
  DEFAULT_TRAIN_BATCH_SIZE=$((8 * NGPUS_PER_NODE * NNODES))
elif [ "${SCALE_MODE}" = "fixed" ]; then
  DEFAULT_TRAIN_BATCH_SIZE=16
else
  echo "SCALE_MODE must be fixed or linear; got ${SCALE_MODE}" >&2
  exit 1
fi
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
# The rollouts of one update and of one optimizer step must be divisible by the number of GPUs
# that train the student; otherwise the run stops only after the first rollouts are generated.
for batch_var in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE; do
  if [ $(( ${!batch_var} * ${ROLLOUT_N:-8} % (NGPUS_PER_NODE * NNODES) )) -ne 0 ]; then
    echo "${batch_var} x ROLLOUT_N ($(( ${!batch_var} * ${ROLLOUT_N:-8} ))) must be divisible by the number of student GPUs ($(( NGPUS_PER_NODE * NNODES )))." >&2
    exit 1
  fi
done
if [ "${BRIDGED_OPD_ENABLED}" = True ]; then
  if [ "${RLSD_ENABLED}" = True ] || [ "${GUIDED_RESCUE_ENABLED}" = True ]; then
    echo "BRIDGED_OPD_ENABLED=True requires RLSD_ENABLED=False and GUIDED_RESCUE_ENABLED=False" >&2
    exit 1
  fi
  if [ "${PPO_MINI_BATCH_SIZE}" != "${TRAIN_BATCH_SIZE}" ]; then
    echo "BRIDGED_OPD_ENABLED=True uses old-policy log-probs and needs PPO_MINI_BATCH_SIZE=TRAIN_BATCH_SIZE" >&2
    exit 1
  fi
fi
BRIDGED_CRITIC_HYDRA_ARGS=()
if [ "${BRIDGED_CRITIC_ENABLED}" = True ]; then
  if [ "${RLSD_ENABLED}" = True ] || [ "${GUIDED_RESCUE_ENABLED}" = True ] || [ "${BRIDGED_OPD_ENABLED}" = True ]; then
    echo "BRIDGED_CRITIC_ENABLED=True requires RLSD_ENABLED, GUIDED_RESCUE_ENABLED and BRIDGED_OPD_ENABLED to be False" >&2
    exit 1
  fi
  if [ "${GRPO_TRUNCATION_AWARE}" != True ]; then
    echo "BRIDGED_CRITIC_ENABLED=True reweights the truncation-aware GRPO advantage: set GRPO_TRUNCATION_AWARE=True" >&2
    exit 1
  fi
  if [ -z "${BRIDGED_CRITIC_CALIB_A}" ] || [ -z "${BRIDGED_CRITIC_CALIB_W}" ] || [ -z "${BRIDGED_CRITIC_CALIB_U}" ]; then
    echo "BRIDGED_CRITIC_ENABLED=True needs the per-student calibration BRIDGED_CRITIC_CALIB_A, _W and _U" >&2
    exit 1
  fi
  BRIDGED_CRITIC_HYDRA_ARGS=(
    "+bsd.bridged_critic_beta=${BRIDGED_CRITIC_BETA}"
    "+bsd.bridged_critic_cuts=${BRIDGED_CRITIC_CUTS}"
    "+bsd.bridged_critic_min_tail=${BRIDGED_CRITIC_MIN_TAIL}"
    "+bsd.bridged_critic_max_prompt_length=${BRIDGED_CRITIC_MAX_PROMPT_LENGTH}"
    "+bsd.bridged_critic_calib_a=${BRIDGED_CRITIC_CALIB_A}"
    "+bsd.bridged_critic_calib_w=${BRIDGED_CRITIC_CALIB_W}"
    "+bsd.bridged_critic_calib_u=${BRIDGED_CRITIC_CALIB_U}"
  )
fi

# Actor master weights (LoRA parameters and Adam moments) default to fp32, verl's training default: an Adam step at
# learning rate 1e-6 is below the bf16 resolution of the LoRA weights and would be lost to rounding if they were stored
# in bf16. Compute still runs under bf16 mixed precision.
export ACTOR_FSDP_MODEL_DTYPE="${ACTOR_FSDP_MODEL_DTYPE:-fp32}"
write_bsd_run_manifest "${EXPERIMENT_NAME}" \
  --model "student=${MODEL_PATH}" \
  --config_kv "method=${METHOD_NAME}" \
  --config_kv "implementation_namespace=bsd" \
  --config_kv "project_name=${PROJECT_NAME}" \
  --config_kv "model_path=${MODEL_PATH}" \
  --config_kv "train_file=${TRAIN_FILE}" \
  --config_kv "val_file=${VAL_FILE}" \
  --config_kv "scale_mode=${SCALE_MODE}" \
  --config_kv "train_batch_size=${TRAIN_BATCH_SIZE}" \
  --config_kv "rlsd_enabled=${RLSD_ENABLED}" \
  --config_kv "rlsd_lambda=${RLSD_LAMBDA}" \
  --config_kv "rlsd_lambda_warmup_steps=${RLSD_LAMBDA_WARMUP_STEPS}" \
  --config_kv "rlsd_lambda_decay_steps=${RLSD_LAMBDA_DECAY_STEPS}" \
  --config_kv "rlsd_clip_range=${RLSD_CLIP_RANGE}" \
  --config_kv "rlsd_negative_only=${RLSD_NEGATIVE_ONLY}" \
  --config_kv "rlsd_positive_mode=${RLSD_POSITIVE_MODE}" \
  --config_kv "rlsd_negative_mode=${RLSD_NEGATIVE_MODE}" \
  --config_kv "format_bonus=${BSD_FORMAT_BONUS}" \
  --config_kv "rlsd_token_mask=${RLSD_TOKEN_MASK}" \
  --config_kv "rlsd_delta_student_logp_source=${RLSD_DELTA_STUDENT_LOGP_SOURCE}" \
  --config_kv "bridged_opd_enabled=${BRIDGED_OPD_ENABLED}" \
  --config_kv "bridged_opd_scope=${BRIDGED_OPD_SCOPE}" \
  --config_kv "bridged_opd_coef=${BRIDGED_OPD_COEF}" \
  --config_kv "bridged_opd_verifier_coef=${BRIDGED_OPD_VERIFIER_COEF}" \
  --config_kv "bridged_opd_clip_abs=${BRIDGED_OPD_CLIP_ABS}" \
  --config_kv "grpo_truncation_aware=${GRPO_TRUNCATION_AWARE}" \
  --config_kv "bridged_critic_enabled=${BRIDGED_CRITIC_ENABLED}" \
  --config_kv "bridged_critic_beta=${BRIDGED_CRITIC_BETA}" \
  --config_kv "bridged_critic_cuts=${BRIDGED_CRITIC_CUTS}" \
  --config_kv "bridged_critic_calibration=${BRIDGED_CRITIC_CALIB_A},${BRIDGED_CRITIC_CALIB_W},${BRIDGED_CRITIC_CALIB_U}" \
  --config_kv "teacher_update_mode=${TEACHER_UPDATE_MODE}" \
  --config_kv "teacher_sync_interval=${TEACHER_SYNC_INTERVAL}" \
  --config_kv "teacher_max_prompt_length=${TEACHER_MAX_PROMPT_LENGTH}" \
  --config_kv "allow_teacher_prompt_truncation=${ALLOW_TEACHER_PROMPT_TRUNCATION}" \
  --config_kv "student_thinking=${STUDENT_THINKING}" \
  --config_kv "teacher_thinking=${TEACHER_THINKING}" \
  --config_kv "emit_think=${EMIT_THINK_PROMPTS}" \
  --config_kv "pi_source=${PI_SOURCE}" \
  --config_kv "pi_field_path=${PI_FIELD_PATH}" \
  ${PI_MANIFEST_ARGS[@]+"${PI_MANIFEST_ARGS[@]}"} \
  --config_kv "guided_rescue_enabled=${GUIDED_RESCUE_ENABLED}" \
  --config_kv "guided_samples=${GUIDED_SAMPLES}" \
  --config_kv "guided_rho_min=${GUIDED_RHO_MIN}" \
  --config_kv "guided_rho_max=${GUIDED_RHO_MAX}" \
  --config_kv "guided_min_length_ratio=${GUIDED_MIN_LENGTH_RATIO}" \
  --config_kv "guided_reject_regex=${GUIDED_REJECT_REGEX-builtin-default}" \
  --config_kv "max_prompt_length=${MAX_PROMPT_LENGTH}" \
  --config_kv "max_response_length=${MAX_RESPONSE_LENGTH}" \
  --config_kv "actor_lr=${ACTOR_LR}" \
  --config_kv "save_lora_only=${SAVE_LORA_ONLY}" \
  --config_kv "ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" \
  --config_kv "optimizer=AdamW" \
  --config_kv "lr_scheduler_type=constant" \
  --config_kv "lr_warmup_steps=10" \
  --config_kv "weight_decay=0.01" \
  --config_kv "clip_grad_norm=1.0" \
  --config_kv "policy_clip_ratio_low=0.2" \
  --config_kv "policy_clip_ratio_high=0.28" \
  --config_kv "loss_aggregation=token-mean" \
  --config_kv "norm_adv_by_std_in_grpo=True" \
  --config_kv "rollout_importance_sampling=token" \
  --config_kv "rollout_importance_sampling_clip=2.0" \
  --config_kv "rollout_is_batch_normalize=False" \
  --config_kv "kl_in_reward=False" \
  --config_kv "kl_coef=0.0" \
  --config_kv "kl_loss=False" \
  --config_kv "kl_loss_coef=0.0" \
  --config_kv "rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}" \
  --config_kv "resume_mode=${RESUME_MODE}" \
  --config_kv "data_shuffle=True" \
  --config_kv "model_use_fused_kernels=True" \
  --config_kv "actor_use_dynamic_bsz=True" \
  --config_kv "actor_ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-81920}" \
  --config_kv "actor_fsdp_model_dtype=${ACTOR_FSDP_MODEL_DTYPE}" \
  --config_kv "actor_fsdp_size=${ACTOR_FSDP_SIZE}" \
  --config_kv "actor_fsdp_forward_prefetch=True" \
  --config_kv "actor_fsdp_param_offload=False" \
  --config_kv "actor_fsdp_optimizer_offload=False" \
  --config_kv "rollout_log_prob_use_dynamic_bsz=True" \
  --config_kv "rollout_log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-131072}" \
  --config_kv "rollout_max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-16384}" \
  --config_kv "rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL:-0.5}" \
  --config_kv "rollout_free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE}" \
  --config_kv "total_training_steps=${TOTAL_TRAINING_STEPS}" \
  --config_kv "save_freq=${SAVE_FREQ}" \
  --config_kv "test_freq=${TEST_FREQ}" \
  --config_kv "val_rollout_n=${VAL_ROLLOUT_N}" \
  --config_kv "val_do_sample=${VAL_DO_SAMPLE}" \
  --config_kv "val_temperature=${VAL_TEMPERATURE}" \
  --config_kv "rollout_n=${ROLLOUT_N:-8}" \
  --config_kv "hydra_overrides=$*" \
  --config_kv "seed=${SEED}"

"${BSD_TRAIN_PYTHON}" -m verl.trainer.bsd.main_bsd \
  trainer.use_v1=False \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.custom_cls.path=pkg://verl.trainer.bsd.prompt_dataset \
  data.custom_cls.name=PromptTemplateRLHFDataset \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  +data.apply_chat_template_kwargs.enable_thinking="${STUDENT_THINKING}" \
  +data.emit_think=${EMIT_THINK_OVERRIDE} \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=True \
  custom_reward_function.path="${BSD_REPO_ROOT}/recipes/bsd/rewards/math_verify_reward.py" \
  custom_reward_function.name=compute_score \
  reward.custom_reward_function.path="${BSD_REPO_ROOT}/recipes/bsd/rewards/math_verify_reward.py" \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.model.bsd_teacher_adapter="${BSD_TEACHER_ADAPTER}" \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.optim.optimizer=AdamW \
  actor_rollout_ref.actor.optim.optimizer_impl=torch.optim \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.optim.clip_grad=1.0 \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  +actor_rollout_ref.actor.checkpoint.save_lora_only="${SAVE_LORA_ONLY}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-81920}" \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_FSDP_MODEL_DTYPE}" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${ACTOR_FSDP_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.seed="${SEED}" \
  actor_rollout_ref.actor.fsdp_config.full_determinism=True \
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.data_loader_seed="${SEED}" \
  actor_rollout_ref.actor.policy_loss.rlsd_enabled="${RLSD_ENABLED}" \
  actor_rollout_ref.actor.policy_loss.rlsd_lambda="${RLSD_LAMBDA}" \
  actor_rollout_ref.actor.policy_loss.rlsd_lambda_warmup_steps="${RLSD_LAMBDA_WARMUP_STEPS}" \
  actor_rollout_ref.actor.policy_loss.rlsd_lambda_decay_steps="${RLSD_LAMBDA_DECAY_STEPS}" \
  actor_rollout_ref.actor.policy_loss.rlsd_reweight_clip_range="${RLSD_CLIP_RANGE}" \
  actor_rollout_ref.actor.policy_loss.rlsd_negative_only="${RLSD_NEGATIVE_ONLY}" \
  actor_rollout_ref.actor.policy_loss.rlsd_positive_mode="${RLSD_POSITIVE_MODE}" \
  actor_rollout_ref.actor.policy_loss.rlsd_negative_mode="${RLSD_NEGATIVE_MODE}" \
  actor_rollout_ref.actor.policy_loss.rlsd_token_mask="${RLSD_TOKEN_MASK}" \
  actor_rollout_ref.actor.policy_loss.rlsd_delta_student_logp_source="${RLSD_DELTA_STUDENT_LOGP_SOURCE}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE:-1}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.5}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.seed="${SEED}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-131072}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-16384}" \
  actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_ROLLOUT_N}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  +bsd.teacher_update_mode="${TEACHER_UPDATE_MODE}" \
  +bsd.teacher_sync_interval="${TEACHER_SYNC_INTERVAL}" \
  +bsd.teacher_max_prompt_length="${TEACHER_MAX_PROMPT_LENGTH}" \
  "${PI_HYDRA_ARGS[@]}" \
  +bsd.rollout_actor_consistency_threshold="${ROLLOUT_ACTOR_CONSISTENCY_THRESHOLD:-0.99}" \
  +bsd.allow_teacher_prompt_truncation="${ALLOW_TEACHER_PROMPT_TRUNCATION}" \
  +bsd.student_thinking="${STUDENT_THINKING}" \
  +bsd.online_teacher_thinking="${TEACHER_THINKING}" \
  +bsd.emit_think=${EMIT_THINK_OVERRIDE} \
  +bsd.guided_rescue_enabled="${GUIDED_RESCUE_ENABLED}" \
  +bsd.bridged_opd_enabled="${BRIDGED_OPD_ENABLED}" \
  +bsd.bridged_opd_scope="${BRIDGED_OPD_SCOPE}" \
  +bsd.bridged_opd_coef="${BRIDGED_OPD_COEF}" \
  +bsd.bridged_opd_verifier_coef="${BRIDGED_OPD_VERIFIER_COEF}" \
  +bsd.bridged_opd_clip_abs="${BRIDGED_OPD_CLIP_ABS}" \
  +bsd.grpo_truncation_aware="${GRPO_TRUNCATION_AWARE}" \
  +bsd.bridged_critic_enabled="${BRIDGED_CRITIC_ENABLED}" \
  ${BRIDGED_CRITIC_HYDRA_ARGS[@]+"${BRIDGED_CRITIC_HYDRA_ARGS[@]}"} \
  +bsd.guided_samples="${GUIDED_SAMPLES}" \
  +bsd.guided_rho_min="${GUIDED_RHO_MIN}" \
  +bsd.guided_rho_max="${GUIDED_RHO_MAX}" \
  +bsd.guided_min_length_ratio="${GUIDED_MIN_LENGTH_RATIO}" \
  ${GUIDED_REGEX_ARGS[@]+"${GUIDED_REGEX_ARGS[@]}"} \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  algorithm.rollout_correction.rollout_is_batch_normalize=False \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.rollout_correction.bypass_mode=False \
  algorithm.rollout_correction.loss_type=ppo_clip \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.seed="${SEED}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.default_local_dir="${BSD_DATA_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
  trainer.logger="${TRAINER_LOGGER}" \
  "$@"
