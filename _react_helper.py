"""React-loop helpers for `plm.PLM`.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple, Type

from plm.constraint import Constraint, ConstraintViolation


# ---------- message-list utility ------------------------------------------

def clone_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return copy.deepcopy(messages)


# ---------- python tool-arg sanitization -----------------------------------

_PYTHON_FENCE_LANGS = frozenset({"python", "py", "python3", "py3", ""})


def _strip_code_fences(code: str) -> str:
    """Tolerate models that wrap the ``code`` tool-arg in markdown fences.

    Smaller / less-tool-trained models sometimes emit ``"```python\\n...\\n```"``
    as the ``code`` argument instead of bare source. ``compile()`` would then
    fail with a ``SyntaxError`` and the model burns a round to learn what we
    could have stripped here.

    Returns bare Python source. Raises :class:`SyntaxError` with an actionable
    message if the fence carries an explicit non-Python language tag.
    """
    
    code = code.strip()
    if "```" not in code:
        return code

    lines = code.splitlines()
    while len(lines) >= 2 and lines[0].strip() == "```" and lines[-1].strip() == "```":
        lines.pop(0)
        lines.pop()
    code = "\n".join(lines).strip()
    if "```" not in code:
        return code

    fence_start = code.find("```")
    lang_line, separator, remainder = code[fence_start + 3:].partition("\n")
    if not separator:
        return code

    lang = (lang_line.strip().split(maxsplit=1)[0] if lang_line.strip() else "").lower()
    if lang not in _PYTHON_FENCE_LANGS:
        raise SyntaxError(
            f"`code` tool-arg wrapped in ```{lang} fence. "
            f"Pass bare Python source as the `code` argument, not {lang}."
        )

    block_end = remainder.find("```")
    if block_end == -1:
        return remainder.strip()
    return remainder[:block_end].strip()


# ---------- REPL output rendering ------------------------------------------

def _truncate_head_tail(text: str, max_chars: int) -> str:
    """Head+tail truncation with a true-length header.

    Long values get clipped to ``max_chars`` total but preserve both ends —
    the start (where types/structure live) and the tail (where final values /
    error suffixes live). The model sees the real length up front so it can
    decide whether to slice the value in code instead of relying on the
    preview. Returns ``text`` unchanged if it already fits.
    """
    
    raw_len = len(text)
    if raw_len <= max_chars:
        return text
    half = max_chars // 2
    omitted = raw_len - max_chars
    return (
        f"[total {raw_len:,} chars; {omitted:,} omitted]\n"
        + text[:half]
        + f"\n... ({omitted:,} characters omitted) ...\n"
        + text[-half:]
    )


def _format_repl_output(result: Dict[str, Any], max_section_chars: int = 8192) -> str:
    """Render a ``repl.execute_cell`` envelope into a tool-message string.

    The envelope is ``{type, return_obj, stdout, stderr}``. Layout (each
    section omitted when empty)::

        <stdout>
        stderr:
        <stderr>

    Truncation is **per-section**: stdout and stderr are individually capped at
    ``max_section_chars`` via head/tail truncation, bounding the worst case
    (runaway print loop, huge traceback) without one section's chop bleeding
    into the other.
    """
    
    parts: List[str] = []
    stdout = (result.get("stdout") or "").rstrip()
    if stdout:
        parts.append(_truncate_head_tail(stdout, max_section_chars))
    stderr = (result.get("stderr") or "").rstrip()
    if stderr:
        parts.append(f"stderr:\n{_truncate_head_tail(stderr, max_section_chars)}")
    return "\n".join(parts) if parts else "(no output)"


# ---------- per-call usage tracking ----------------------------------------

def _track_model_call(
    response: Dict[str, Any],
    call_start: float,
    metrics: Optional[Dict[str, Any]],
    token_state: Dict[str, Any],
) -> None:
    """Mutate ``token_state`` and ``metrics`` from a model-call response.

    Keeps the cumulative-prompt-token counter and ``metrics['model_calls']``
    uniform across every per-round generate call.
    """
    
    usage = response.get("usage") or {}
    tok_total = usage.get("total_tokens", 0) or 0
    pt = usage.get("prompt_tokens", 0) or 0
    if pt > 0:
        token_state["cumulative_prompt_tokens"] = pt
    if metrics is not None:
        metrics["model_calls"].append({
            "call_duration": time.time() - call_start,
            "cumulative_prompt_tokens": token_state.get("cumulative_prompt_tokens", 0),
            "tokens_this_call": tok_total,
            "tokens": usage,
            "model": response.get("model", "unknown"),
            "backend": response.get("backend", "unknown"),
        })
        metrics["total_tokens"] += int(tok_total)


# ---------- RETURN contract ------------------------------------------------

def _format_constraint_error(
    constraint: Type[Constraint],
    err: ConstraintViolation,
) -> str:
    """Format a Constraint violation for re-injection into a python tool
    result's stderr so the model sees what to fix on the next round."""
    
    return (
        f"RETURN value failed the configured output constraint:\n"
        f"{err}\n\n"
        f"Required: {constraint.describe()}\n\n"
        f"Fix the object passed to RETURN(...)."
    )


def _build_last_round_reminder(
    constraint: Optional[Type[Constraint]],
) -> Dict[str, Any]:
    """User-role message reminding the model to call RETURN."""

    if constraint is not None:
        body = (
            f"This is your final round budget. Your task is configured with an "
            f"output constraint. You MUST conclude the task by calling "
            f"`RETURN(obj)` in a python cell, where obj satisfies the "
            f"constraint:\n"
            f"      {constraint.describe()}\n"
            f"If RETURN is not called successfully, the task fails."
        )
    else:
        body = (
            "This is your final round budget. Conclude the task by calling "
            "`RETURN(obj)` in a python cell — `obj` becomes the final answer.\n"
            "If RETURN is not called successfully, the task fails."
        )
    return {
        "role": "user",
        "content": body,
        "_timestamp": time.time(),
        "_plm_meta": "final_round_return_reminder",
    }


def _validate_return_against_constraint(
    raw_return_value: Any,
    constraint: Optional[Type[Constraint]],
) -> Tuple[bool, Any]:
    """Validate a ``RETURN(obj)`` payload against an optional Constraint.

    Contract:

    - ``constraint is None`` → ``(True, raw_return_value)`` (no validation;
      the raw object becomes the terminal answer).
    - ``constraint`` set, value satisfies it → ``(True, validated)`` where
      ``validated`` is what ``constraint.validate`` returns: a pydantic
      instance for a structural Constraint, or the (possibly coerced) scalar
      for a factory Constraint. That value is the terminal answer.
    - ``constraint`` set, value does NOT satisfy it → ``(False, error_text)``
      where ``error_text`` is ``_format_constraint_error(constraint, cv)``,
      which the caller appends to the python tool result's stderr so the model
      sees what to fix next round.

    Any non-``ConstraintViolation`` exception from ``validate`` propagates.
    """
    if constraint is None:
        return True, raw_return_value
    try:
        validated = constraint.validate(raw_return_value)
    except ConstraintViolation as cv:
        return False, _format_constraint_error(constraint, cv)
    return True, validated
