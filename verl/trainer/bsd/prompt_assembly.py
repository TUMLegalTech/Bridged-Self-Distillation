"""Prompt assembly for Bridged Self-Distillation recipes."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Sequence

from jinja2 import Environment, FileSystemLoader, StrictUndefined

DEFAULT_PI_TRACE_FIELD = "qwen32b_pi_trace"
PI_SOURCE_TRACE_FIELD = "trace_field"
PI_SOURCE_GROUND_TRUTH = "ground_truth"
VALID_PI_SOURCES = frozenset({PI_SOURCE_TRACE_FIELD, PI_SOURCE_GROUND_TRUTH})

PROMPT_DIR = Path(__file__).resolve().parents[3] / "recipes" / "bsd" / "prompts"
STUDENT_USER_TEMPLATE = "student_nonthink.jinja"
DIRECT_OPD_TEACHER_USER_TEMPLATE = "direct_opd_teacher_nonthink.jinja"
PI_TEACHER_USER_TEMPLATE = "bsd_self_teacher_nonthink.jinja"
ANSWER_TEACHER_USER_TEMPLATE = "answer_teacher_nonthink.jinja"
# A closed (API) teacher generates its traces from the same plain prompt as the
# open-weight teachers, so this names the shared trace-generation template.
CLOSED_TEACHER_USER_TEMPLATE = "pi_generation.jinja"
OPSD_USER_TEMPLATE = STUDENT_USER_TEMPLATE
PI_GENERATION_USER_TEMPLATE = "pi_generation.jinja"

# Emit-think variants replace the last line by the instruction to reason inside
# <think> and </think> tags. Qwen3-4B-Instruct uses them: its chat template opens the
# reasoning block, and the fine-tuned model reasons and closes the block under this
# instruction. The Qwen3 thinking students keep the templates above.
STUDENT_EMIT_THINK_USER_TEMPLATE = "student_emit_think.jinja"
DIRECT_OPD_TEACHER_EMIT_THINK_USER_TEMPLATE = "direct_opd_teacher_emit_think.jinja"
PI_TEACHER_EMIT_THINK_USER_TEMPLATE = "bsd_self_teacher_emit_think.jinja"
ANSWER_TEACHER_EMIT_THINK_USER_TEMPLATE = "answer_teacher_emit_think.jinja"


def student_user_template(emit_think: bool = False) -> str:
    return STUDENT_EMIT_THINK_USER_TEMPLATE if emit_think else STUDENT_USER_TEMPLATE


def direct_opd_teacher_user_template(emit_think: bool = False) -> str:
    return DIRECT_OPD_TEACHER_EMIT_THINK_USER_TEMPLATE if emit_think else DIRECT_OPD_TEACHER_USER_TEMPLATE


def pi_teacher_user_template(emit_think: bool = False, override: str | None = None) -> str:
    """The scorer-only PI teacher template; ``override`` names a context-specific wording (BSD-hint, BSD-guide)."""
    if override:
        return validate_pi_teacher_template(override)
    return PI_TEACHER_EMIT_THINK_USER_TEMPLATE if emit_think else PI_TEACHER_USER_TEMPLATE


def validate_pi_teacher_template(template_name: str) -> str:
    """A PI teacher template must be a repository prompt file that renders both ``problem`` and ``pi_trace``."""
    path = PROMPT_DIR / template_name
    if not path.is_file():
        raise ValueError(f"pi_teacher_template {template_name!r} is not a file under {PROMPT_DIR}")
    source = path.read_text(encoding="utf-8")
    if "{{ problem }}" not in source or "{{ pi_trace }}" not in source:
        raise ValueError(f"pi_teacher_template {template_name!r} must render {{{{ problem }}}} and {{{{ pi_trace }}}}")
    return template_name


def answer_teacher_user_template(emit_think: bool = False) -> str:
    return ANSWER_TEACHER_EMIT_THINK_USER_TEMPLATE if emit_think else ANSWER_TEACHER_USER_TEMPLATE


def optional_bool(value: Any) -> bool | None:
    """Parse a tri-state config value: ``None``/empty means "not specified"."""

    if value is None or isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if not normalized or normalized in {"none", "null", "auto"}:
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean or an empty value, got {value!r}")


def resolve_emit_think(enable_thinking: bool | None, emit_think: bool | None = None) -> bool:
    """Pick the prompt variant for a role.

    Without an explicit ``emit_think`` the variant follows the thinking mode: the
    plain templates with thinking on, the emit-think ones with thinking off.
    ``enable_thinking=None`` means the caller does not know the mode, and keeps
    the plain templates so an unspecified mode never changes prompts.
    An explicit ``emit_think`` takes precedence: Qwen3-4B-Instruct runs with
    thinking on and ``emit_think=True``.
    """

    if emit_think is not None:
        return bool(emit_think)
    if enable_thinking is None:
        return False
    return not bool(enable_thinking)


@cache
def _prompt_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(PROMPT_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=False,
    )


def load_prompt_template(template_name: str) -> str:
    """Return the prompt template source stored in the recipe prompt directory."""

    source, _, _ = _prompt_environment().loader.get_source(_prompt_environment(), template_name)
    return source


def render_prompt_template(template_name: str, **context: Any) -> str:
    """Render one repository prompt template with strict variable checking."""

    return _prompt_environment().get_template(template_name).render(**context)


def prompt_template_sha256(template_name: str) -> str:
    """Content hash of one prompt template, for generation provenance.

    Every generator records its sampling settings and the model name; the prompt is the
    other half of what produced a trace, so its hash is recorded with them. An edited
    template would otherwise silently change what a resumed or regenerated dataset
    contains, because the traces carry no marker of which wording produced them.
    """

    from hashlib import sha256

    return sha256((PROMPT_DIR / template_name).read_bytes()).hexdigest()


def render_student_user_prompt(problem: str, *, emit_think: bool = False) -> str:
    return render_prompt_template(student_user_template(emit_think), problem=problem)


_FINAL_ANSWER_MARKER = re.compile(r"\*{0,2}\s*final answer\s*\*{0,2}\s*[:\uff1a]?", re.IGNORECASE)
_BOXED_OPEN = "\\boxed{"
_ANSWER_PLACEHOLDER = "[answer omitted]"


def _blank_boxed_spans(text: str) -> str:
    """Replace every ``\\boxed{...}`` span (brace-balanced) by a placeholder."""

    out: list[str] = []
    cursor = 0
    while True:
        start = text.find(_BOXED_OPEN, cursor)
        if start < 0:
            out.append(text[cursor:])
            return "".join(out)
        out.append(text[cursor:start])
        depth = 1
        end = start + len(_BOXED_OPEN)
        while end < len(text) and depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        out.append(_ANSWER_PLACEHOLDER)
        cursor = end


def strip_trace_answer(trace: str) -> str:
    """Keep the reasoning block of a PI trace without its explicit final-answer statement.

    Qwen-style traces close the reasoning block with ``**Final Answer** ... \\boxed{..}``
    *before* ``</think>``, so dropping the text after ``</think>`` alone leaves the answer
    in place. The reasoning is cut at its last final-answer marker (else at its last
    ``\\boxed``), and any remaining boxed span is blanked. The derivation itself may
    still state the answer in prose; that residual leakage is not removed.

    Used by the guided rescue in the trainer, whose sampling prompt is conditioned on
    the stripped trace.
    """

    head, sep, _ = trace.partition("</think>")
    markers = list(_FINAL_ANSWER_MARKER.finditer(head))
    if markers:
        head = head[: markers[-1].start()]
    else:
        boxed_start = head.rfind(_BOXED_OPEN)
        if boxed_start >= 0:
            head = head[:boxed_start]
    head = _blank_boxed_spans(head).rstrip()
    return head + "\n" + sep if sep else head


def render_pi_teacher_user_prompt(
    problem: str, pi_trace: str, *, emit_think: bool = False, template: str | None = None
) -> str:
    return render_prompt_template(pi_teacher_user_template(emit_think, template), problem=problem, pi_trace=pi_trace)


def render_answer_teacher_user_prompt(problem: str, ground_truth: str, *, emit_think: bool = False) -> str:
    return render_prompt_template(
        answer_teacher_user_template(emit_think), problem=problem, ground_truth=ground_truth
    )


def render_closed_teacher_user_prompt(problem: str) -> str:
    return render_prompt_template(CLOSED_TEACHER_USER_TEMPLATE, problem=problem)


def render_opsd_user_prompt(problem: str, *, emit_think: bool = False) -> str:
    return render_prompt_template(student_user_template(emit_think), problem=problem)


def render_pi_generation_user_prompt(problem: str) -> str:
    return render_prompt_template(PI_GENERATION_USER_TEMPLATE, problem=problem)


@dataclass(frozen=True)
class PromptBundle:
    """Serialized student and scorer prompts for one problem."""

    student_prompt: str
    direct_opd_teacher_prompt: str
    pi_teacher_prompt: str | None


def _has_explicit_thinking_control(tokenizer) -> bool:
    try:
        parameters = inspect.signature(tokenizer.apply_chat_template).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "enable_thinking" in parameters:
        return True
    return "enable_thinking" in str(getattr(tokenizer, "chat_template", "") or "")


def _fixed_template_thinking_mode(rendered_prompt: str, prompt_without_generation: str) -> bool:
    if not rendered_prompt.startswith(prompt_without_generation):
        raise ValueError("Cannot isolate the assistant generation marker from the serialized chat prompt")
    generation_suffix = rendered_prompt[len(prompt_without_generation) :]
    return generation_suffix.rfind("<think>") > generation_suffix.rfind("</think>")


def render_chat_messages(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    enable_thinking: bool = False,
) -> str:
    """Render messages while requiring the requested thinking mode to be honored."""

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        raise ValueError(
            f"Tokenizer {getattr(tokenizer, 'name_or_path', type(tokenizer).__name__)} "
            "does not accept the required enable_thinking chat-template argument"
        ) from exc

    if not _has_explicit_thinking_control(tokenizer):
        prompt_without_generation = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        fixed_mode = _fixed_template_thinking_mode(rendered, prompt_without_generation)
        if fixed_mode != enable_thinking:
            raise ValueError(
                f"Tokenizer {getattr(tokenizer, 'name_or_path', type(tokenizer).__name__)} has a fixed "
                f"thinking_mode={fixed_mode}, incompatible with requested enable_thinking={enable_thinking}"
            )
    return rendered


def _render_chat(tokenizer, user_text: str, *, enable_thinking: bool = False) -> str:
    messages = [{"role": "user", "content": user_text}]
    return render_chat_messages(tokenizer, messages, enable_thinking=enable_thinking)


def build_student_prompt(
    tokenizer,
    problem: str,
    *,
    enable_thinking: bool = False,
    emit_think: bool | None = None,
) -> str:
    """Build the PI-free student rollout prompt."""

    variant = resolve_emit_think(enable_thinking, emit_think)
    return _render_chat(
        tokenizer,
        render_student_user_prompt(problem, emit_think=variant),
        enable_thinking=enable_thinking,
    )


def build_direct_opd_teacher_prompt(
    tokenizer,
    problem: str,
    *,
    enable_thinking: bool = False,
    emit_think: bool | None = None,
) -> str:
    """Build the direct OPD teacher prompt, identical to the student prompt."""

    variant = resolve_emit_think(enable_thinking, emit_think)
    return _render_chat(
        tokenizer,
        render_prompt_template(direct_opd_teacher_user_template(variant), problem=problem),
        enable_thinking=enable_thinking,
    )


def build_pi_teacher_prompt(
    tokenizer,
    problem: str,
    pi_trace: str,
    *,
    enable_thinking: bool = False,
    emit_think: bool | None = None,
    template: str | None = None,
) -> str:
    """Build the scorer-only privileged prompt."""

    variant = resolve_emit_think(enable_thinking, emit_think)
    return _render_chat(
        tokenizer,
        render_pi_teacher_user_prompt(problem, pi_trace, emit_think=variant, template=template),
        enable_thinking=enable_thinking,
    )


def build_answer_teacher_prompt(
    tokenizer,
    problem: str,
    ground_truth: str,
    *,
    enable_thinking: bool = False,
    emit_think: bool | None = None,
) -> str:
    """Build a scorer-only prompt conditioned on the correct final answer."""

    variant = resolve_emit_think(enable_thinking, emit_think)
    return _render_chat(
        tokenizer,
        render_answer_teacher_user_prompt(problem, ground_truth, emit_think=variant),
        enable_thinking=enable_thinking,
    )


def normalize_pi_source(pi_source: str) -> str:
    """Validate and normalize the privileged-information source."""

    normalized = str(pi_source).strip().lower()
    if normalized not in VALID_PI_SOURCES:
        expected = ", ".join(sorted(VALID_PI_SOURCES))
        raise ValueError(f"pi_source must be one of {expected}; got {pi_source!r}")
    return normalized


def build_prompt_bundle(
    tokenizer,
    problem: str,
    pi_trace: str | None,
    *,
    student_thinking: bool = False,
    teacher_thinking: bool = False,
) -> PromptBundle:
    """Build all prompts needed by one Bridged Self-Distillation sample."""

    student_prompt = build_student_prompt(tokenizer, problem, enable_thinking=student_thinking)
    direct_prompt = build_direct_opd_teacher_prompt(tokenizer, problem, enable_thinking=teacher_thinking)
    pi_prompt = None
    if pi_trace:
        pi_prompt = build_pi_teacher_prompt(
            tokenizer,
            problem,
            pi_trace,
            enable_thinking=teacher_thinking,
        )
    return PromptBundle(
        student_prompt=student_prompt,
        direct_opd_teacher_prompt=direct_prompt,
        pi_teacher_prompt=pi_prompt,
    )


def assert_no_pi_in_student_prompt(student_prompt: str, pi_trace: str | None) -> None:
    """Fail if privileged text leaks into the student rollout prompt."""

    reference_tags = (
        "<reference_solution>",
        "</reference_solution>",
        "<reference_answer>",
        "</reference_answer>",
    )
    if any(tag in student_prompt for tag in reference_tags):
        raise ValueError("Student prompt contains PI reference tags")
    if pi_trace and pi_trace.strip() and pi_trace.strip() in student_prompt:
        raise ValueError("Student prompt contains privileged trace text")
