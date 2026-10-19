<h1 align="center">Bridged Self-Distillation (BSD)</h1>

<p align="center">
  <b>One on-policy distillation procedure for any teacher that returns text.</b>
</p>

<p align="center">
  Code for the paper "Bridged Self-Distillation for On-Policy Learning".
</p>

<p align="center">
  Runsheng Chen<sup>1</sup>, Yihong Liu<sup>2,3</sup>, Raoyuan Zhao<sup>2,3</sup>, Michael A. Hedderich<sup>2,3</sup>, Hinrich Schütze<sup>2,3</sup>, Matthias Grabmair<sup>1</sup><br>
  <sup>1</sup>Technical University of Munich &nbsp; <sup>2</sup>LMU Munich &nbsp; <sup>3</sup>Munich Center for Machine Learning (MCML)
</p>

<p align="center">
  <img src="assets/bsd_overview.png" alt="Overview of Bridged Self-Distillation" width="100%">
</p>

## Overview

**Bridged Self-Distillation (BSD)** distills from any teacher that returns
text, whether it shares the student's tokenizer, uses a different one, or is
a closed API:

1. The external teacher is queried **once per problem, before training**, for
   a textual response $z$, its trace.
2. The student generates rollouts **without** $z$, and a verifier marks each
   rollout as correct or incorrect.
3. A frozen copy of the student that reads $z$, the **self-teacher**, scores
   the same tokens in the student's own token space.
4. The per-token log-probability difference between the two, the
   **token-level guidance**, reweights the penalty on rollouts that the
   verifier rejects, while the verifier sets the direction of every update.

This repository contains BSD, the four baselines of the paper (SFT, vanilla
OPD, xOPD and ROPD), the preparation of their training files and the
evaluation harness.

## Main results

Avg@4 on AIME 2024, AIME 2025 and HMMT 2025 (Table 1 of the paper). OPD
denotes the on-policy distillation baseline of each setting: vanilla OPD,
xOPD and ROPD. Bold marks the best distillation method within each
student-teacher setting.

**Mean across the three benchmarks**

| Student | Method | Qwen3-32B<br>(shared tokenizer) | Nemotron-3-30B-A3B<br>(cross-tokenizer) | GPT-5.6 Luna<br>(closed API) |
| --- | --- | :---: | :---: | :---: |
| **Qwen3-4B** | Base | 59.2 | 59.2 | 59.2 |
|  | + SFT | 60.6 | **63.6** | 57.2 |
|  | + OPD | 60.3 | 59.7 | 59.7 |
|  | + BSD | **63.9** | **63.6** | **61.9** |
| **Qwen3-4B-Instruct** | Base | 44.4 | 44.4 | 44.4 |
|  | + SFT | 40.8 | 38.1 | 39.7 |
|  | + OPD | 42.2 | 45.0 | 45.8 |
|  | + BSD | **45.8** | **45.6** | **46.4** |
| **Qwen3-1.7B** | Base | 35.8 | 35.8 | 35.8 |
|  | + SFT | 30.0 | 37.5 | 36.9 |
|  | + OPD | **38.6** | 35.0 | 36.4 |
|  | + BSD | **38.6** | **39.2** | **40.0** |
| **Qwen3-0.6B** | Base | 10.8 | 10.8 | 10.8 |
|  | + SFT | 9.2 | 7.5 | 2.8 |
|  | + OPD | 13.1 | **14.2** | 11.4 |
|  | + BSD | **13.6** | 12.8 | **12.5** |

## Repository structure

```text
.
├── recipes/bsd/
│   ├── scripts/
│   │   ├── 00_prepare_eval_benchmarks.sh           # AIME 2024, AIME 2025, HMMT 2025
│   │   ├── 01_prepare_training_problems.sh         # pool of training problems
│   │   ├── 02_generate_teacher_traces.sh           # traces of a local external teacher (vLLM)
│   │   ├── 03_generate_closed_teacher_traces.sh    # traces of the closed API teacher
│   │   ├── 04_build_ropd_teacher_index.sh          # teacher index of ROPD
│   │   ├── run_bridged_self_distillation_multi.sh  # BSD (calls run_bsd_multi.sh)
│   │   ├── run_sft_multi.sh                        # SFT
│   │   ├── run_opd_multi.sh                        # vanilla OPD
│   │   ├── run_xopd_multi.sh                       # xOPD
│   │   ├── run_ropd_multi.sh                       # ROPD
│   │   ├── run_online_rl_multi.sh                  # shared launcher of the OPD baselines
│   │   ├── run_eval_checkpoint.sh                  # evaluation on the development set
│   │   ├── run_official_math_eval.sh               # evaluation on the benchmarks
│   │   └── evaluate_all.sh                         # collects the result files into one table
│   ├── prompts/                                    # student, self-teacher, OPD teacher and trace-generation prompts
│   ├── rewards/math_verify_reward.py               # verifier
│   ├── configs/                                    # judge providers of ROPD
│   └── environments/                               # pinned requirements of the two environments
├── tools/bsd/                                      # Python tools that the scripts call
├── verl/                                           # the part of verl that these methods use
│   └── trainer/bsd/                                # BSD and the baselines
├── third_party/ropd_official/                      # released ROPD code, pinned
├── scripts/create_bsd_environment.sh               # creates the training or the generation environment
└── assets/bsd_overview.png
```

## Installation

Requirements: Linux on x86-64 with CUDA 12, `conda`, and NVIDIA GPUs (four by
default for training, one for evaluation).

```bash
bash scripts/create_bsd_environment.sh training     # training, data preparation, evaluation
bash scripts/create_bsd_environment.sh generation   # local teachers: traces and the xOPD teacher server
```

Everything is stored under `BSD_DATA_ROOT` (default `<repo>/workspace`).
Every script selects its interpreter by absolute path; start them with
`bash <script>` from the repository root. The closed API teacher and ROPD need API
credentials: copy `.env.example` to `.env` and fill it in. The hint and guide
variants read `OPENROUTER_API_KEY` from the environment. The header comment of every script documents its variables.

## Quick start

```bash
export BSD_DATA_ROOT="$PWD/workspace"
PY="$PWD/workspace/conda_envs/BSD/bin/python"
D="${BSD_DATA_ROOT}/datasets"
```

### 1. Data preparation

The paper trains on the mathematical-reasoning subset of OpenThoughts: 2,400
training problems per teacher-student pair with one trace per problem, and
128 development problems that select the checkpoint.

```bash
# Benchmarks, then the pool of training problems with the benchmark problems quarantined
bash recipes/bsd/scripts/00_prepare_eval_benchmarks.sh
EVAL_JSONL="$D/eval/official_math/aime24.jsonl:$D/eval/official_math/aime25.jsonl:$D/eval/official_math/hmmt25.jsonl" \
bash recipes/bsd/scripts/01_prepare_training_problems.sh

# One trace per problem from the external teacher (default: Nemotron-3-30B-A3B)
INPUT_JSONL="$D/opsd_clean/all_clean.jsonl" \
bash recipes/bsd/scripts/02_generate_teacher_traces.sh

# 2,400 training and 128 development problems, then the BSD training file
PYTHONPATH=. "$PY" tools/bsd/sample_train_dev_split.py \
  --input_jsonl "$D"/teacher_traces/nemotron_generation/worker_*/raw.jsonl \
  --output_dir "$D/splits/nemotron" --pi_trace_field nemotron_pi_trace
for split in train dev; do
  PYTHONPATH=. "$PY" tools/bsd/jsonl_to_verl_parquet.py \
    --input_jsonl "$D/splits/nemotron/$split.jsonl" \
    --output_parquet "$D/splits/nemotron/$split.parquet" --pi_trace_field nemotron_pi_trace
done
```

| External teacher | Traces | Trace field of the training file |
| --- | --- | --- |
| Nemotron-3-30B-A3B | `02_generate_teacher_traces.sh` | `nemotron_pi_trace` |
| Qwen3-32B | `02_generate_teacher_traces.sh` with `MODEL_PATH=Qwen/Qwen3-32B TRACE_FIELD=qwen32b_pi_trace SAMPLING_CONFIG_NAME=A` and its own `OUTPUT_DIR` | `qwen32b_pi_trace` |
| GPT-5.6 Luna (closed API) | `03_generate_closed_teacher_traces.sh` | `luna_pi_trace`: convert with `--pi_trace_field closed_teacher_output --output_trace_field luna_pi_trace` |

The split keeps only traces with a verified final answer; the paper's closed
API training sets also keep the others (`--keep_unverified_traces`). A closed
teacher hides its reasoning, so its trace is built from its response: the
derivation in a reasoning block, then the boxed final answer
(`tools/bsd/closed_teacher_trace.py`).

The baselines read further files built from the same split:

| Method | Training file |
| --- | --- |
| SFT | `tools/bsd/build_teacher_sft_parquet.py --response_field <trace field>` (`--preset closed` for the closed API teacher) |
| ROPD | `04_build_ropd_teacher_index.sh`, the teacher index of the closed API teacher |
| BSD-hint, BSD-guide | `tools/bsd/generate_pi_forms.py`, then `tools/bsd/build_pi_form_views.py --append-answer` |

### 2. Training

```bash
EXPERIMENT_NAME=<run> \
MODEL_PATH=Qwen/Qwen3-4B \
TRAIN_FILE="$D/splits/nemotron/train.parquet" \
VAL_FILE="$D/splits/nemotron/dev.parquet" \
PI_TRACE_FIELD=nemotron_pi_trace \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash recipes/bsd/scripts/run_bridged_self_distillation_multi.sh
```

Every launcher defaults to the training configuration of the paper and writes
checkpoints to `${BSD_DATA_ROOT}/checkpoints/<run>/global_step_<N>/actor`
every 25 updates. The baselines use the student, the problems and the
development file of the corresponding BSD run:

| Method | Launcher | Teacher-specific variables |
| --- | --- | --- |
| BSD | `run_bridged_self_distillation_multi.sh` | `PI_TRACE_FIELD` (default `qwen32b_pi_trace`) |
| SFT | `run_sft_multi.sh` | `TRAIN_FILE=<sft.parquet>`; learning rate 5e-6 |
| Vanilla OPD | `run_opd_multi.sh` | `TEACHER_MODEL` (default Qwen3-32B), `CUDA_VISIBLE_DEVICES=0,1,2`; one rollout per problem |
| xOPD | `run_xopd_multi.sh` | `TEACHER_MODEL` (default Nemotron-3-30B-A3B), `ACTOR_GPUS=0,1 TEACHER_GPUS=2` |
| ROPD | `run_ropd_multi.sh` | `ROPD_TEACHER_INDEX_PATH`, credentials in `.env`; 8,192 response tokens |
| BSD-hint, BSD-guide | `run_bridged_self_distillation_multi.sh` | `PI_TRACE_FIELD=<field>_hint` or `_guide`, `PI_TEACHER_TEMPLATE=bsd_self_teacher_hint_nonthink.jinja` or `bsd_self_teacher_guide_nonthink.jinja` |

The students Qwen3-1.7B and Qwen3-0.6B only change `MODEL_PATH`.
Qwen3-4B-Instruct takes the directory of its fine-tuned checkpoint as
`MODEL_PATH` and `EMIT_THINK_PROMPTS=True`. `TOTAL_TRAINING_STEPS=3 SAVE_FREQ=3`
gives a smoke run.

### 3. Evaluation

Evaluate every checkpoint on the development file and take the one with the
highest Avg@4 (the earliest at a tie), then evaluate it on the benchmarks:

```bash
for step in 25 50 75 100 125 150; do
  EVAL_FILE="$D/splits/nemotron/dev.parquet" MODEL_PATH=Qwen/Qwen3-4B \
  CHECKPOINT_PATH="${BSD_DATA_ROOT}/checkpoints/<run>/global_step_${step}/actor" \
  RUN_NAME=<run> CHECKPOINT_NAME="step_${step}" \
  bash recipes/bsd/scripts/run_eval_checkpoint.sh
done
bash recipes/bsd/scripts/evaluate_all.sh

MODEL_PATH=Qwen/Qwen3-4B \
CHECKPOINT_PATH="${BSD_DATA_ROOT}/checkpoints/<run>/global_step_<N>/actor" \
RUN_NAME=<run> CHECKPOINT_NAME=step_<N> \
bash recipes/bsd/scripts/run_official_math_eval.sh
```

Both scripts default to the evaluation protocol of the paper: 4 samples per
problem with seeds 0-3, thinking enabled, temperature 1.0, top-p 1.0 and
32,768 new tokens. They report Avg@4 in the field `avg` and Pass@4 in the
field `pass` of their result files.

## Citation

The citation will be added once the paper is on arXiv.

## Acknowledgements

This code builds on [verl](https://github.com/verl-project/verl), vLLM, PEFT
and math-verify. The weight $w_t$ follows the released code of Self-Distilled
RLVR ([iie-ycx/RLSD](https://github.com/iie-ycx/RLSD)), the self-teacher
prompt follows On-Policy Self-Distillation
([siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD)), and ROPD runs the
released code of its authors
([Peregrine123/ROPD_official](https://github.com/Peregrine123/ROPD_official),
vendored under `third_party/ropd_official/` with its license). The training
problems come from `siyanzhao/Openthoughts_math_30k_opsd`, and the benchmarks
from `HuggingFaceH4/aime_2024`, `yentinglin/aime_2025` and
`MathArena/hmmt_feb_2025`.

## License

Apache License 2.0; see `LICENSE`. The notice of upstream verl is in
`Notice.txt`; the vendored ROPD code keeps its own license in
`third_party/ropd_official/LICENSE`.
