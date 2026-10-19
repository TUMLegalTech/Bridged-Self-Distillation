"""Dataset adapters that render project prompts from repository templates."""

from __future__ import annotations

import traceback
from collections.abc import Mapping
from typing import Any

from verl.trainer.bsd.prompt_assembly import (
    optional_bool,
    render_student_user_prompt,
    resolve_emit_think,
)
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import normalize_token_ids


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    return value if isinstance(value, Mapping) else {}


def _problem_from_example(example: Mapping[str, Any]) -> str:
    extra_info = _as_mapping(example.get("extra_info"))
    metadata = _as_mapping(example.get("metadata"))
    problem = extra_info.get("problem") or example.get("problem") or metadata.get("problem")
    if problem is None or not str(problem).strip():
        raise ValueError("Prompt-template dataset rows require extra_info.problem, problem, or metadata.problem")
    return str(problem)


class _StudentPromptRenderer:
    """Pick the student template variant from the dataset config.

    By default the variant follows the chat-template thinking mode: the plain template
    with thinking on. Qwen3-4B-Instruct needs the emit-think template while thinking
    stays on, because its chat template opens the reasoning block and the model was
    fine-tuned to reason and close the block under that instruction, so
    ``data.emit_think`` overrides the default independently.
    """

    def _render_student_prompt(self, example: Mapping[str, Any]) -> str:
        template_kwargs = _as_mapping(getattr(self, "apply_chat_template_kwargs", None))
        config = getattr(self, "config", None)
        override = optional_bool(config.get("emit_think") if config is not None else None)
        emit_think = resolve_emit_think(template_kwargs.get("enable_thinking"), override)
        return render_student_user_prompt(_problem_from_example(example), emit_think=emit_think)


class PromptTemplateRLHFDataset(_StudentPromptRenderer, RLHFDataset):
    """Render each math rollout prompt from the active student Jinja template."""

    def _build_messages(self, example: dict, key: str):
        del key
        return [{"role": "user", "content": self._render_student_prompt(example)}]

    def maybe_filter_out_long_prompts(self, dataframe=None):
        """Apply the prompt-length limit to the student prompt that is rolled out.

        For a text-only model the parent class measures the prompt column of the file. The
        rollout uses the prompt that ``_build_messages`` renders instead, and its last line
        depends on the template variant, so it can be longer than that column. A row that
        passed the limit on the column would then be rolled out from a truncated prompt.
        """

        if not self.filter_overlong_prompts or self.processor is not None:
            return super().maybe_filter_out_long_prompts(dataframe)

        def doc2len(doc) -> int:
            # A row that cannot be rendered is a data error and raises; only the measurement
            # below skips a row, as in the parent class.
            messages = self._build_messages(doc, key=self.prompt_key)
            try:
                apply_kwargs = dict(**self.apply_chat_template_kwargs)
                if self.tool_schemas is not None:
                    apply_kwargs["tools"] = self.tool_schemas

                apply_kwargs.pop("tokenize", None)
                apply_kwargs.pop("return_dict", None)
                apply_kwargs.pop("return_tensors", None)

                tokenized_prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **apply_kwargs
                )
                return len(normalize_token_ids(tokenized_prompt))
            except Exception:
                print("Error processing one of the samples, skipping...")
                traceback.print_exc()
                return self.max_prompt_length + 1

        dataframe = dataframe.filter(
            lambda doc: doc2len(doc) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

        print(f"filter dataset len: {len(dataframe)}")
        return dataframe


class PromptTemplateSFTDataset(_StudentPromptRenderer, MultiTurnSFTDataset):
    """Replace the stored SFT user turn with the active student Jinja template."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # MultiTurnSFTDataset copies the settings it wants off config and drops the object,
        # where RLHFDataset keeps it. _render_student_prompt looks for data.emit_think there,
        # so without this the override reads as unset and the template silently falls back to
        # inverting enable_thinking -- which is the whole failure this override exists to stop.
        self.config = kwargs.get("config", args[2] if len(args) > 2 else None) or {}

    def _build_messages(self, example: dict):
        messages = super()._build_messages(example)
        if not messages or messages[0].get("role") != "user":
            raise ValueError("Prompt-template SFT rows must start with a user message")
        messages[0] = {"role": "user", "content": self._render_student_prompt(example)}
        return messages
