#!/usr/bin/env python3
"""Serve selected-token prompt log probabilities from a vLLM teacher."""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import zmq

from verl.trainer.bsd.cross_vocab_opd import (
    PROTOCOL_VERSION,
    pack_rpc_message,
    tokenizer_fingerprint,
    unpack_rpc_message,
)


def _extract_selected_prompt_log_probs(output: Any, input_ids: Sequence[int]) -> list[float]:
    """Extract each actual prompt token's log probability from a vLLM output."""

    returned_ids = [int(token_id) for token_id in output.prompt_token_ids]
    expected_ids = [int(token_id) for token_id in input_ids]
    if returned_ids != expected_ids:
        raise RuntimeError("vLLM returned prompt token IDs that differ from the teacher request")
    prompt_log_probs = output.prompt_logprobs
    if prompt_log_probs is None or len(prompt_log_probs) != len(expected_ids):
        raise RuntimeError("vLLM returned an invalid prompt_logprobs length")

    selected = [0.0]
    for position in range(1, len(expected_ids)):
        token_id = expected_ids[position]
        candidates = prompt_log_probs[position]
        if candidates is None or token_id not in candidates:
            raise RuntimeError(f"vLLM omitted selected prompt token {token_id} at position {position}")
        value = candidates[token_id]
        selected.append(float(value.logprob if hasattr(value, "logprob") else value))
    return selected


class VLLMPromptLogProbEngine:
    """Minimal vLLM wrapper for deterministic prompt scoring."""

    def __init__(
        self,
        *,
        model_path: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        max_num_batched_tokens: int,
    ) -> None:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        self._tokens_prompt = TokensPrompt
        self._sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
            detokenize=False,
        )
        self._llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            enable_chunked_prefill=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
        )

    def score(self, input_ids: Sequence[Sequence[int]]) -> list[list[float]]:
        prompts = [self._tokens_prompt(prompt_token_ids=list(row)) for row in input_ids]
        outputs = self._llm.generate(
            prompts=prompts,
            sampling_params=self._sampling_params,
            use_tqdm=False,
        )
        if len(outputs) != len(input_ids):
            raise RuntimeError("vLLM returned an invalid teacher batch size")
        return [
            _extract_selected_prompt_log_probs(output, row)
            for output, row in zip(outputs, input_ids, strict=True)
        ]


def serve(
    *,
    bind: str,
    model_path: str,
    tokenizer_hash: str,
    max_model_len: int,
    engine: Any,
) -> None:
    """Run the versioned single-worker REP service."""

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(bind)
    print(f"Cross-vocabulary teacher ready at {bind}", flush=True)
    try:
        while True:
            try:
                request = unpack_rpc_message(socket.recv())
                if request.get("protocol_version") != PROTOCOL_VERSION:
                    raise ValueError("Unsupported teacher protocol version")
                action = request.get("action")
                if action == "health":
                    response = {
                        "status": "ok",
                        "protocol_version": PROTOCOL_VERSION,
                        "model_path": model_path,
                        "tokenizer_fingerprint": tokenizer_hash,
                        "max_model_len": max_model_len,
                    }
                elif action == "score":
                    rows = request.get("input_ids")
                    if not isinstance(rows, list) or not rows:
                        raise ValueError("Teacher score request requires a non-empty input_ids batch")
                    normalized = [[int(token_id) for token_id in row] for row in rows]
                    if any(len(row) > max_model_len for row in normalized):
                        raise ValueError("Teacher score request exceeds max_model_len; truncation is forbidden")
                    response = {
                        "status": "ok",
                        "protocol_version": PROTOCOL_VERSION,
                        "log_probs": engine.score(normalized),
                    }
                else:
                    raise ValueError(f"Unknown teacher action: {action!r}")
            except Exception as exc:
                response = {
                    "status": "error",
                    "protocol_version": PROTOCOL_VERSION,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            socket.send(pack_rpc_message(response))
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        context.term()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bind", default="tcp://127.0.0.1:15555")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=30720)
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192)
    args = parser.parse_args()

    if args.tensor_parallel_size <= 0:
        raise SystemExit("--tensor_parallel_size must be positive")
    if args.max_model_len <= 0 or args.max_num_batched_tokens <= 0:
        raise SystemExit("Teacher token limits must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer_hash = tokenizer_fingerprint(tokenizer)
    engine = VLLMPromptLogProbEngine(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    serve(
        bind=args.bind,
        model_path=args.model,
        tokenizer_hash=tokenizer_hash,
        max_model_len=args.max_model_len,
        engine=engine,
    )


if __name__ == "__main__":
    main()
