#!/usr/bin/env python3
"""Evaluate checkpoints and write Bridged Self-Distillation compatible JSON metrics.

The defaults are the evaluation parameters of the paper: 4 samples per problem, seeds
0-3, thinking enabled, temperature 1.0, top-p 1.0, at most 32,768 new tokens. In the
result JSON "avg" is Avg@4 (mean correctness over all samples) and "pass" is Pass@4
(share of problems solved by at least one sample). Called directly, the tool also runs
one temperature-0 pass and reports it in the greedy_* fields; --skip_greedy leaves it
out, which is what the recipe launchers do unless EVAL_GREEDY_PASS=1.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from recipes.bsd.rewards.math_verify_reward import compute_score, verify_answer
from tools.bsd.common import read_jsonl
from verl.trainer.bsd.prompt_assembly import (
    DEFAULT_PI_TRACE_FIELD,
    optional_bool,
    render_chat_messages,
    render_student_user_prompt,
    resolve_emit_think,
)


@dataclass
class EvalItem:
    sample_id: str
    problem: str
    ground_truth: str
    messages: list[dict[str, str]]
    pi_trace: str | None = None


def _to_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return _to_python(value.as_py())
    if hasattr(value, "tolist") and not isinstance(value, str):
        return _to_python(value.tolist())
    if isinstance(value, dict):
        return {str(key): _to_python(inner) for key, inner in value.items()}
    if isinstance(value, list | tuple):
        return [_to_python(inner) for inner in value]
    return value


def _load_raw_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd

        return [_to_python(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    return read_jsonl(path)


def load_eval_items(
    path: str | Path,
    max_rows: int | None = None,
    pi_trace_field: str = DEFAULT_PI_TRACE_FIELD,
    emit_think: bool = False,
) -> list[EvalItem]:
    rows = _load_raw_rows(path)
    if max_rows is not None:
        rows = rows[:max_rows]
    items = []
    for idx, row in enumerate(rows):
        row = _to_python(row)
        extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        reward_model = row.get("reward_model") if isinstance(row.get("reward_model"), dict) else {}
        problem = row.get("problem") or extra_info.get("problem")
        if problem is None:
            raise ValueError(f"Cannot find raw problem text for prompt-template rendering in eval row {idx}")
        ground_truth = row.get("gold_answer") or reward_model.get("ground_truth")
        if ground_truth is None:
            raise ValueError(f"Cannot find ground truth in eval row {idx}")
        messages = [{"role": "user", "content": render_student_user_prompt(str(problem), emit_think=emit_think)}]
        items.append(
            EvalItem(
                # This id chain is the pairing key of the evaluation JSON files;
                # keep it as-is instead of adopting common.row_uid.
                sample_id=str(row.get("id") or extra_info.get("id") or idx),
                problem=str(problem),
                ground_truth=str(ground_truth),
                messages=[{"role": str(msg["role"]), "content": str(msg["content"])} for msg in messages],
                pi_trace=extra_info.get(pi_trace_field) or row.get(pi_trace_field),
            )
        )
    return items


def build_prompts(
    tokenizer: Any, items: list[EvalItem], enable_thinking: bool, thinking_mode: str = "enforce"
) -> list[str]:
    """Serialize eval prompts with an explicit chat-template thinking contract.

    "enforce" requires the template to honor enable_thinking and fails when it
    cannot; "template_default" applies the template without the option, for
    tokenizers whose template exposes no thinking control. There is no silent
    fallback between the two.
    """
    if thinking_mode == "template_default":
        return [
            tokenizer.apply_chat_template(item.messages, tokenize=False, add_generation_prompt=True)
            for item in items
        ]
    return [
        render_chat_messages(tokenizer, item.messages, enable_thinking=enable_thinking)
        for item in items
    ]


def _normal_answer(value: str | None) -> str:
    return "" if value is None else "".join(str(value).lower().split())


def _response_metrics(text: str, ground_truth: str) -> tuple[float, float, str]:
    result = compute_score("bsd/eval", text, ground_truth)
    return float(result["accuracy"]), float(result["format"]), str(result.get("boxed_answer", ""))


THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"


def _think_tag_metrics(text: str) -> dict[str, float]:
    """Detect the reasoning-block tags in one generation.

    A model taught to emit the block itself fails in distinguishable ways -- no tag at
    all, an open tag whose close was cut off by the token budget, repeated tags, or a
    block that starts late -- so the rates are kept apart rather than collapsed into one.
    A thinking-mode chat template emits the open tag as part of the prompt, in which case
    ``opened`` is legitimately 0 while ``closed`` is 1.

    "Well formed" means exactly one of each tag, open before close. ``open_first`` is the
    stricter bar that the block opens the response. The trainer computes the same rates
    over rollout token ids, so training and eval are directly comparable.
    """
    opens = text.count(THINK_OPEN_TAG)
    closes = text.count(THINK_CLOSE_TAG)
    well_formed = (
        opens == 1 and closes == 1 and text.index(THINK_OPEN_TAG) < text.index(THINK_CLOSE_TAG)
    )
    return {
        "opened": float(opens > 0),
        "closed": float(closes > 0),
        "well_formed": float(well_formed),
        "open_first": float(text.startswith(THINK_OPEN_TAG)),
        "duplicated": float(opens > 1 or closes > 1),
    }


def _hf_truncated(new_tokens: Any, *, eos_token_id: int | None, max_new_tokens: int) -> bool:
    token_ids = new_tokens.detach().cpu().tolist() if hasattr(new_tokens, "detach") else list(new_tokens)
    return len(token_ids) >= max_new_tokens and (eos_token_id is None or eos_token_id not in token_ids)


def _contains_default_lora_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    shard_paths = sorted(path.glob("*.safetensors"))
    if not shard_paths:
        return False
    from safetensors import safe_open

    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            if any(".lora_A.default.weight" in key or ".lora_B.default.weight" in key for key in handle.keys()):
                return True
    return False


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _is_peft_adapter_dir(path: Path) -> bool:
    """Return whether a directory contains a directly loadable PEFT adapter."""
    if not path.is_dir() or not (path / "adapter_config.json").is_file():
        return False
    return any(
        (path / filename).is_file()
        for filename in ("adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt")
    )


def _materialize_default_lora_adapter(checkpoint_path: str | Path, base_model: str) -> Path:
    checkpoint_path = Path(checkpoint_path)
    adapter_dir = checkpoint_path / ".bsd_vllm_lora_default"
    adapter_model = adapter_dir / "adapter_model.safetensors"
    adapter_config = adapter_dir / "adapter_config.json"
    if adapter_model.exists() and adapter_config.exists():
        return adapter_dir

    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    lora_rank = None
    for shard_path in sorted(checkpoint_path.glob("*.safetensors")):
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if ".lora_A.default.weight" not in key and ".lora_B.default.weight" not in key:
                    continue
                adapter_key = key.replace(".default.weight", ".weight")
                tensor = handle.get_tensor(key)
                tensors[adapter_key] = tensor
                if ".lora_A.weight" in adapter_key and lora_rank is None:
                    lora_rank = int(tensor.shape[0])
    if not tensors:
        raise ValueError(f"No default LoRA tensors found in checkpoint: {checkpoint_path}")

    tmp_dir = adapter_dir.with_name(f"{adapter_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    save_file(tensors, tmp_dir / "adapter_model.safetensors")
    rank = lora_rank or 32
    config = {
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": rank * 2,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "task_type": "CAUSAL_LM",
    }
    (tmp_dir / "adapter_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    tmp_dir.replace(adapter_dir)
    return adapter_dir


def _resolve_default_lora_adapter(checkpoint_path: str | Path, base_model: str) -> Path | None:
    """Resolve PEFT adapter exports and full-state LoRA checkpoints."""
    checkpoint_path = Path(checkpoint_path)
    candidates = [checkpoint_path, checkpoint_path / "huggingface"]
    for candidate in candidates:
        if _is_peft_adapter_dir(candidate):
            return candidate
        cached_adapter = candidate / ".bsd_vllm_lora_default"
        if _is_peft_adapter_dir(cached_adapter):
            return cached_adapter
    for candidate in candidates:
        if _contains_default_lora_weights(candidate):
            return _materialize_default_lora_adapter(candidate, base_model)
    return None


def parse_seeds(value: str | None, num_samples: int) -> list[int]:
    """Parse fixed evaluation seeds, extending or truncating to num_samples."""

    if value:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        seeds = list(range(num_samples))
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    if len(seeds) < num_samples:
        start = seeds[-1] + 1
        seeds.extend(range(start, start + num_samples - len(seeds)))
    return seeds[:num_samples]


def summarize_scored_items(
    *,
    run: str,
    checkpoint: str,
    scored_items: list[dict[str, Any]],
) -> dict[str, Any]:
    per_problem = []
    sample_accuracies = []
    sample_formats = []
    think_rates: dict[str, list[float]] = {
        key: [] for key in ("opened", "closed", "well_formed", "open_first", "duplicated")
    }
    truncations = []
    response_lengths = []
    prompt_tokens = []
    completion_tokens = []
    leakage_hits = 0
    total_samples = 0
    per_seed_scores: dict[int, list[float]] = {}
    finish_reasons: Counter[str] = Counter()

    for item in scored_items:
        sample_scores = []
        sample_metadata = []
        boxed_answers = []
        for sample_index, sample in enumerate(item["samples"]):
            accuracy, fmt, boxed = _response_metrics(sample["text"], item["ground_truth"])
            seed = int(sample.get("seed", sample_index))
            completion_length = float(sample.get("completion_tokens", len(sample["text"])))
            truncated = bool(sample.get("truncated", False))
            finish_reason = str(sample.get("finish_reason") or ("length" if truncated else "stop"))
            think = _think_tag_metrics(sample["text"])
            sample_scores.append(accuracy)
            sample_accuracies.append(accuracy)
            per_seed_scores.setdefault(seed, []).append(accuracy)
            sample_formats.append(fmt)
            for key, value in think.items():
                think_rates[key].append(value)
            boxed_answers.append(boxed)
            truncations.append(float(truncated))
            response_lengths.append(completion_length)
            prompt_tokens.append(float(sample.get("prompt_tokens", 0)))
            completion_tokens.append(float(sample.get("completion_tokens", 0)))
            finish_reasons[finish_reason] += 1
            sample_metadata.append(
                {
                    "seed": seed,
                    "accuracy": accuracy,
                    "completion_tokens": int(completion_length),
                    "truncated": truncated,
                    "finish_reason": finish_reason,
                    "think_wellformed": bool(think["well_formed"]),
                }
            )
            pi_trace = item.get("pi_trace")
            if "<reference_solution>" in sample["text"] or (pi_trace and pi_trace.strip() in sample["text"]):
                leakage_hits += 1
            total_samples += 1

        answer_counts = Counter(_normal_answer(answer) for answer in boxed_answers if answer)
        majority_answer = answer_counts.most_common(1)[0][0] if answer_counts else ""
        majority_correct = verify_answer(majority_answer, item["ground_truth"]) if majority_answer else False
        if item.get("greedy") is None:  # --skip_greedy
            greedy_accuracy, greedy_format, greedy_boxed = None, None, None
        else:
            greedy_accuracy, greedy_format, greedy_boxed = _response_metrics(item["greedy"]["text"], item["ground_truth"])
        per_problem.append(
            {
                "id": item["id"],
                "avg": sum(sample_scores) / len(sample_scores) if sample_scores else 0.0,
                "pass": float(any(score > 0 for score in sample_scores)),
                "majority": float(majority_correct),
                "greedy_accuracy": greedy_accuracy,
                "greedy_format": greedy_format,
                "greedy_boxed_answer": greedy_boxed,
                "samples": sample_metadata,
            }
        )

    problem_avgs = [row["avg"] for row in per_problem]
    mean_problem_avg = sum(problem_avgs) / len(problem_avgs) if problem_avgs else 0.0
    group_variance = (
        sum((value - mean_problem_avg) ** 2 for value in problem_avgs) / len(problem_avgs) if problem_avgs else 0.0
    )
    return {
        "run": run,
        "checkpoint": checkpoint,
        "avg": sum(sample_accuracies) / len(sample_accuracies) if sample_accuracies else 0.0,
        "majority": sum(row["majority"] for row in per_problem) / len(per_problem) if per_problem else 0.0,
        "pass": sum(row["pass"] for row in per_problem) / len(per_problem) if per_problem else 0.0,
        "greedy_accuracy": (
            None
            if any(row["greedy_accuracy"] is None for row in per_problem)
            else (sum(row["greedy_accuracy"] for row in per_problem) / len(per_problem) if per_problem else 0.0)
        ),
        "format_rate": sum(sample_formats) / len(sample_formats) if sample_formats else 0.0,
        "think_open_rate": _mean(think_rates["opened"]),
        "think_close_rate": _mean(think_rates["closed"]),
        "think_wellformed_rate": _mean(think_rates["well_formed"]),
        "think_open_first_token_rate": _mean(think_rates["open_first"]),
        "think_duplicate_tag_rate": _mean(think_rates["duplicated"]),
        "truncation_rate": sum(truncations) / len(truncations) if truncations else 0.0,
        "mean_response_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0.0,
        "median_response_length": median(response_lengths) if response_lengths else 0.0,
        "p90_response_length": _percentile(response_lengths, 0.90),
        "over_4k_response_rate": (
            sum(length > 4096 for length in response_lengths) / len(response_lengths) if response_lengths else 0.0
        ),
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "per_seed_accuracy": {str(seed): sum(scores) / len(scores) for seed, scores in sorted(per_seed_scores.items())},
        "group_variance": group_variance,
        "all_correct_rate": sum(1.0 for row in per_problem if row["avg"] == 1.0) / len(per_problem)
        if per_problem
        else 0.0,
        "all_wrong_rate": sum(1.0 for row in per_problem if row["avg"] == 0.0) / len(per_problem)
        if per_problem
        else 0.0,
        "prompt_tokens": sum(prompt_tokens),
        "completion_tokens": sum(completion_tokens),
        "total_samples": total_samples,
        "privacy_leakage_rate": leakage_hits / total_samples if total_samples else 0.0,
        "per_problem": per_problem,
    }


def save_sample_outputs(
    *,
    output_json: str | Path,
    items: list[EvalItem],
    scored_items: list[dict[str, Any]],
    count: int,
) -> Path | None:
    """Persist the full text of a fixed subset of generations for later inspection.

    Only aggregate statistics survive in the summary JSON, so there is otherwise no way to
    go back and read what the model actually wrote. Problems are picked at evenly spaced
    indices rather than at random, so the same ones are kept for every checkpoint and the
    files stay directly comparable across a run.
    """
    if count <= 0 or not scored_items:
        return None
    total = len(scored_items)
    kept = min(count, total)
    indices = [round(i * (total - 1) / (kept - 1)) for i in range(kept)] if kept > 1 else [0]
    problems = {item.sample_id: item.problem for item in items}
    path = Path(str(output_json).removesuffix(".json") + ".samples.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in indices:
            scored = scored_items[index]
            record = {
                "index": index,
                "id": scored["id"],
                "problem": problems.get(scored["id"], ""),
                "ground_truth": scored["ground_truth"],
                "greedy": scored["greedy"],
                "samples": scored["samples"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _evaluate_hf(args: argparse.Namespace, items: list[EvalItem]) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model if args.adapter_path else (args.checkpoint or args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        device_map="auto",
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    prompts = build_prompts(
        tokenizer, items, enable_thinking=args.enable_thinking, thinking_mode=args.thinking_mode
    )
    scored = []
    seeds = parse_seeds(args.seeds, args.num_samples)
    for item, prompt in zip(items, prompts, strict=True):
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = int(encoded["input_ids"].shape[-1])
        greedy_new = None
        if not getattr(args, "skip_greedy", False):
            greedy = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            greedy_new = greedy[0, prompt_len:]
        samples = []
        for seed in seeds:
            # transformers generate() rejects a per-call generator kwarg; seed
            # the global RNGs instead to keep per-seed determinism.
            torch.manual_seed(seed)
            output = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                # 0 disables top-k filtering. Without it generate() inherits top_k from the
                # model's generation_config.json (20 for Qwen3), and the samples would not
                # be drawn at plain temperature / top-p as the vLLM backend draws them.
                top_k=0,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = output[0, prompt_len:]
            truncated = _hf_truncated(
                new_tokens, eos_token_id=tokenizer.eos_token_id, max_new_tokens=args.max_new_tokens
            )
            samples.append(
                {
                    "seed": seed,
                    "text": tokenizer.decode(new_tokens, skip_special_tokens=True),
                    "truncated": truncated,
                    "finish_reason": "length" if truncated else "stop",
                    "prompt_tokens": prompt_len,
                    "completion_tokens": int(len(new_tokens)),
                }
            )
        greedy_entry = None
        if greedy_new is not None:
            greedy_truncated = _hf_truncated(
                greedy_new, eos_token_id=tokenizer.eos_token_id, max_new_tokens=args.max_new_tokens
            )
            greedy_entry = {
                "text": tokenizer.decode(greedy_new, skip_special_tokens=True),
                "truncated": greedy_truncated,
                "finish_reason": "length" if greedy_truncated else "stop",
                "prompt_tokens": prompt_len,
                "completion_tokens": int(len(greedy_new)),
            }
        scored.append(
            {
                "id": item.sample_id,
                "ground_truth": item.ground_truth,
                "pi_trace": item.pi_trace,
                "greedy": greedy_entry,
                "samples": samples,
            }
        )
    return scored


def _evaluate_vllm(args: argparse.Namespace, items: list[EvalItem]) -> list[dict[str, Any]]:
    import multiprocessing as mp
    import os

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    mp.set_start_method("spawn", force=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    adapter_path = Path(args.adapter_path) if args.adapter_path else None
    model_path = args.model if adapter_path is not None else (args.checkpoint or args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    prompts = build_prompts(
        tokenizer, items, enable_thinking=args.enable_thinking, thinking_mode=args.thinking_mode
    )
    prompt_token_counts = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    llm_kwargs = {
        "model": model_path,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    lora_request = None
    if adapter_path is not None:
        from vllm.lora.request import LoRARequest

        llm_kwargs.update({"enable_lora": True, "max_lora_rank": 32})
        lora_request = LoRARequest("bsd_default", 1, str(adapter_path))
    llm = LLM(
        **llm_kwargs,
    )
    if getattr(args, "skip_greedy", False):
        greedy_outputs: list[Any] = [None] * len(prompts)
    else:
        greedy_outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, n=1),
            lora_request=lora_request,
        )
    seeds = parse_seeds(args.seeds, args.num_samples)
    outputs_by_seed = [
        llm.generate(
            prompts,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                n=1,
                seed=seed,
            ),
            lora_request=lora_request,
        )
        for seed in seeds
    ]
    scored = []
    for item_idx, (item, prompt_tokens, greedy_output) in enumerate(
        zip(items, prompt_token_counts, greedy_outputs, strict=True)
    ):
        samples = []
        for seed_idx, seed in enumerate(seeds):
            output = outputs_by_seed[seed_idx][item_idx].outputs[0]
            finish_reason = getattr(output, "finish_reason", None)
            samples.append(
                {
                    "seed": seed,
                    "text": output.text,
                    "truncated": finish_reason == "length",
                    "finish_reason": finish_reason or "unknown",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(output.token_ids),
                }
            )
        greedy = greedy_output.outputs[0] if greedy_output is not None else None
        scored.append(
            {
                "id": item.sample_id,
                "ground_truth": item.ground_truth,
                "pi_trace": item.pi_trace,
                "greedy": None
                if greedy is None
                else {
                    "text": greedy.text,
                    "truncated": getattr(greedy, "finish_reason", None) == "length",
                    "finish_reason": getattr(greedy, "finish_reason", None) or "unknown",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(greedy.token_ids),
                },
                "samples": samples,
            }
        )
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--run", default=None)
    parser.add_argument("--checkpoint_name", default=None)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2,3", help="Comma-separated fixed seeds for prompt-seed eval pairs.")
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument(
        "--skip_greedy",
        action="store_true",
        help="Skip the temperature-0 pass; greedy_* metrics are written as null and only the sampled passes run.",
    )
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate with thinking enabled (default); --no-enable_thinking turns it off.",
    )
    parser.add_argument(
        "--thinking_mode",
        choices=("enforce", "template_default"),
        default="enforce",
        help=(
            "'enforce' fails unless the chat template honors --enable_thinking; "
            "'template_default' applies the template without the option, for tokenizers "
            "that expose no thinking control. --enable_thinking still drives the "
            "emit-think user template and the recorded metadata either way."
        ),
    )
    parser.add_argument(
        "--emit_think",
        default=None,
        help=(
            "Force the emit-think user template on or off, independently of "
            "--enable_thinking. Needed when the chat template already opens the reasoning "
            "block but the model was taught to close it under the tag instruction. "
            "Omit to follow --enable_thinking."
        ),
    )
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--pi_trace_field", default=DEFAULT_PI_TRACE_FIELD)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument(
        "--save_samples",
        type=int,
        default=8,
        help=(
            "Keep the full generations of this many evenly spaced problems in a "
            "<output_json>.samples.jsonl sidecar for later inspection. 0 disables."
        ),
    )
    args = parser.parse_args()

    if args.adapter_path is None and args.checkpoint is not None:
        resolved_adapter = _resolve_default_lora_adapter(args.checkpoint, args.model)
        if resolved_adapter is not None:
            args.adapter_path = str(resolved_adapter)

    emit_think = resolve_emit_think(args.enable_thinking, optional_bool(args.emit_think))
    items = load_eval_items(
        args.eval_file,
        max_rows=args.max_rows,
        pi_trace_field=args.pi_trace_field,
        emit_think=emit_think,
    )
    scored_items = _evaluate_vllm(args, items) if args.backend == "vllm" else _evaluate_hf(args, items)
    payload = summarize_scored_items(
        run=args.run or Path(args.checkpoint or args.model).name,
        checkpoint=args.checkpoint_name or Path(args.checkpoint or args.model).name,
        scored_items=scored_items,
    )
    payload["eval_file"] = args.eval_file
    payload["dataset"] = args.dataset_name or Path(args.eval_file).stem
    payload["pi_trace_field"] = args.pi_trace_field
    payload["model"] = args.model
    payload["checkpoint_path"] = args.checkpoint
    payload["adapter_path"] = args.adapter_path
    payload["backend"] = args.backend
    payload["seeds"] = parse_seeds(args.seeds, args.num_samples)
    # Record the sampling budget next to the scores: it decides what a number means. A
    # smaller token budget truncates more responses, which moves the format rate and can
    # be mistaken for a method effect.
    payload["max_new_tokens"] = args.max_new_tokens
    payload["temperature"] = args.temperature
    payload["top_p"] = args.top_p
    payload["num_samples"] = args.num_samples
    payload["enable_thinking"] = bool(args.enable_thinking)
    payload["thinking_mode"] = args.thinking_mode
    payload["emit_think"] = emit_think
    samples_path = save_sample_outputs(
        output_json=args.output_json,
        items=items,
        scored_items=scored_items,
        count=args.save_samples,
    )
    payload["sample_outputs_path"] = str(samples_path) if samples_path is not None else None
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
