"""SFT entrypoint that trains selected token embeddings alongside the LoRA adapter.

Qwen3-4B-Instruct-2507 carries ``<think>``/``</think>`` in its vocabulary but never
emits them, and their tied embedding rows are close to initialization: the rows have
about a third of the mean vocabulary norm and a mutual cosine near 0.99. A LoRA that
only adapts the transformer body can push the hidden state toward that cluster but
cannot make the frozen readout prefer ``<think>`` over its neighbours, so a model
fine-tuned that way emits ``</tool_call>`` instead. PEFT's ``trainable_token_indices``
unfreezes just those rows, which is the smallest change that can separate them.

This lives in its own entrypoint so the shared SFT trainer and FSDP engine stay
unchanged for every other run. Nothing here takes effect unless a run is launched
through this module.

Usage::

    BSD_TRAINABLE_TOKEN_INDICES=151667,151668 \
        python -m verl.trainer.bsd.sft_trainer_trainable_tokens <hydra overrides>

The indices come from an environment variable rather than Hydra because the shared
model config dataclass rejects unknown keys and must not be edited here.
"""

from __future__ import annotations

import os

import verl.workers.engine.fsdp.transformer_impl as fsdp_transformer_impl
from verl.trainer.sft_trainer import main as _sft_main
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

TRAINABLE_TOKEN_INDICES_ENV = "BSD_TRAINABLE_TOKEN_INDICES"
EMBEDDING_MODULE = "embed_tokens"

# PEFT stores the trained rows as
# ``...embed_tokens.token_adapter.trainable_tokens_delta.default``. The shared
# LoRA-only checkpoint filter matches ``lora_`` or ``.adapter_``, and neither
# substring appears in that name, so without this marker the resumable shard would
# silently drop the very weights this entrypoint exists to train.
TRAINABLE_TOKEN_STATE_MARKERS = ("trainable_tokens_", "token_adapter.")


def trainable_token_indices() -> list[int]:
    """Return the token ids to unfreeze, parsed from the environment."""

    raw = os.environ.get(TRAINABLE_TOKEN_INDICES_ENV, "").strip()
    if not raw:
        return []
    indices = [int(part) for part in raw.replace(",", " ").split()]
    if len(set(indices)) != len(indices):
        raise ValueError(f"{TRAINABLE_TOKEN_INDICES_ENV} must not repeat token ids; got {raw!r}")
    return indices


class FSDPEngineWithTrainableTokens(FSDPEngineWithLMHead):
    """FSDP engine whose LoRA adapter also trains a fixed set of token embeddings."""

    def _build_lora_module(self, module):
        indices = trainable_token_indices()
        if not indices:
            return super()._build_lora_module(module)

        original_lora_config = fsdp_transformer_impl.LoraConfig

        def lora_config_with_trainable_tokens(**kwargs):
            kwargs["trainable_token_indices"] = {EMBEDDING_MODULE: indices}
            return original_lora_config(**kwargs)

        # Patch only for the duration of the base implementation so the rest of its
        # logic (pre-trained adapter reuse, teacher adapter, dtype casting) is shared
        # rather than duplicated.
        fsdp_transformer_impl.LoraConfig = lora_config_with_trainable_tokens
        try:
            return super()._build_lora_module(module)
        finally:
            fsdp_transformer_impl.LoraConfig = original_lora_config


def _install_trainable_token_support() -> None:
    """Point the language-model engine and the adapter-key filter at this variant."""

    registry = EngineRegistry._engines["language_model"]
    for backend in ("fsdp", "fsdp2"):
        for key in list(registry.get(backend, {})):
            registry[backend][key] = FSDPEngineWithTrainableTokens

    base_is_adapter_state_key = BaseCheckpointManager.is_adapter_state_key

    def is_adapter_state_key(key: str) -> bool:
        return base_is_adapter_state_key(key) or any(
            marker in key for marker in TRAINABLE_TOKEN_STATE_MARKERS
        )

    BaseCheckpointManager.is_adapter_state_key = staticmethod(is_adapter_state_key)


def main() -> None:
    indices = trainable_token_indices()
    if indices:
        _install_trainable_token_support()
        print(f"[trainable-tokens] unfreezing {EMBEDDING_MODULE} rows {indices}")
    else:
        print(f"[trainable-tokens] {TRAINABLE_TOKEN_INDICES_ENV} is unset; running plain LoRA SFT")
    _sft_main()


if __name__ == "__main__":
    main()
