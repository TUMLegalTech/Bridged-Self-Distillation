"""Trace of a closed API teacher, built from the response it returns.

A closed teacher hides its reasoning, so its trace is assembled from its final
response: the derivation is enclosed in a reasoning block (<think> and </think>)
and the boxed final answer is placed after the block, which is the format of the
students' own responses. ``cot_structure`` is used by jsonl_to_verl_parquet.py
(BSD training files) and build_teacher_sft_parquet.py (SFT training files).
"""

from __future__ import annotations

import re

from recipes.bsd.rewards.math_verify_reward import verify_answer


_BOXED_COMMAND = re.compile(r"\\boxed")


# What may sit between the answer box and its closing \] for the block to still count as
# nothing but the answer. A bare period is punctuation; "\quad\text{(B)}" or ",\qquad x\ne 0"
# is content, and dropping the block would take it with the equation it qualifies.
_ANSWER_BLOCK_TAIL = re.compile(r"^[.,;]?\s*$")


def boxed_spans(text: str) -> list[tuple[int, int, str]]:
    """Every brace-balanced ``\\boxed{...}`` as (command start, span end, content)."""

    spans: list[tuple[int, int, str]] = []
    for match in _BOXED_COMMAND.finditer(text):
        brace = text.find("{", match.end())
        if brace < 0:
            continue
        depth = 0
        for position in range(brace, len(text)):
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((match.start(), position + 1, text[brace + 1 : position]))
                    break
    return spans


def answer_box_index(spans: list[tuple[int, int, str]], ground_truth: str) -> int:
    """Pick the box this dataset should treat as the answer.

    The last box is not always it. The teacher sometimes boxes the value and then boxes the
    option letter, which is exactly what ``extract_boxed_answer``'s ``rfind`` picks up, so 17
    rows of this corpus record an option letter as their answer. Prefer the box that verifies
    against the gold answer, and fall back to the last one, which leaves a genuinely wrong
    trace exactly as wrong as it was.
    """

    for index in range(len(spans) - 1, -1, -1):
        if verify_answer(spans[index][2].strip(), ground_truth):
            return index
    return len(spans) - 1


def cot_structure(trace: str, ground_truth: str) -> tuple[str, str]:
    """Reshape a visible teacher derivation into ``<think>reasoning</think>`` + boxed answer.

    The closed teacher writes one continuous derivation ending in ``\\boxed{...}``. The student
    is trained to emit a reasoning block and then a final answer, so the block has to close
    before the answer rather than contain it.

    Three shapes need different handling; the counts are measured over the 2,400-row file of
    the paper, which the comments of this module call this corpus:

    * 2020 rows end with the box alone in its own display block. Dropping the block is
      lossless, because the derivation already concluded before it.
    * 330 rows carry the box inside the closing computation (``\\[EF=\\dots=\\boxed{14}\\]``),
      inline in a sentence, or alongside something the block still needs -- an option label,
      or a condition such as ``,\\qquad x\\ne 0``. Dropping the block there would delete the
      final step and strand a dangling lead-in, so the box is unwrapped in place instead and
      its value stays put.
    * 48 rows box more than once -- a multi-part answer, or a lemma boxed on the way. Cutting
      any single one would split the answer, so the derivation is left untouched and the
      answer is restated after the block. That repeats one box, deliberately.

    A trace with no box keeps its text and gets an empty ``\\boxed{}``: this corpus does not
    filter teacher traces on correctness, and dropping the row would.

    Returns the restructured trace and the answer it now ends with.
    """

    spans = boxed_spans(trace)
    if not spans:
        return f"<think>\n{trace.strip()}\n</think>\n\n\\boxed{{}}", ""

    answer = spans[answer_box_index(spans, ground_truth)][2].strip()
    if len(spans) > 1:
        reasoning = trace.strip()
    else:
        start, end, content = spans[0]
        block_open = trace.rfind("\\[", 0, start)
        block_close = trace.rfind("\\]", 0, start)
        block_end = trace.find("\\]", end)
        # The block counts as "nothing but the answer" only when it is empty on BOTH sides of
        # the box. Checking only the left side classifies eight rows of this corpus wrongly,
        # including one whose block reads "\boxed{x y' + 2y = x^4},\qquad x\ne 0" -- dropping
        # it deleted the differential equation and left the condition dangling.
        answer_is_whole_block = (
            block_open >= 0
            and block_open > block_close
            and block_end >= 0
            and not trace[block_open + 2 : start].strip()
            and _ANSWER_BLOCK_TAIL.match(trace[end:block_end])
        )
        if answer_is_whole_block:
            reasoning = (trace[:block_open] + trace[block_end + 2 :]).strip()
        else:
            reasoning = (trace[:start] + content + trace[end:]).strip()
    return f"<think>\n{reasoning}\n</think>\n\n\\boxed{{{answer}}}", answer
