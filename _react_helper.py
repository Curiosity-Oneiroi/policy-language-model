"""React-loop helpers for `plm.PLM`.
"""

from __future__ import annotations

import copy
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Type

from plm.constraint import Constraint, ConstraintViolation


# ---------- message-list utility ------------------------------------------

def clone_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return copy.deepcopy(messages)


def public_trajectory(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deep copy of the root trajectory for seeding ``plm_messages`` in the REPL.

    Keeps the real conversation fields (``role``, ``content``, ``reasoning``,
    ``tool_calls``, ``tool_call_id``) and drops internal harness bookkeeping keys
    (anything starting with ``_`` — e.g. ``_timestamp``, ``_plm_meta``,
    ``_structured``). The result is what PLM sees as its own trajectory inside a
    cell: the full messages, including tool calls and tool results.

    A deep copy so the kernel-side ``plm_messages`` is independent of the
    parent's live ``messages`` (the loop's source of truth); a cell mutating
    ``plm_messages`` cannot corrupt the real trajectory, and the parent simply
    re-seeds a fresh snapshot next round.
    """
    return [
        {k: copy.deepcopy(v) for k, v in m.items() if not k.startswith("_")}
        for m in messages
    ]


# ---------- python tool-arg sanitization -----------------------------------

_PYTHON_FENCE_LANGS = frozenset({"python", "py", "python3", "py3", ""})

# A bare fence-opener line: ``` (or more), an optional single language token, and
# nothing else. A bare fence-closer line: ``` (or more) and nothing else. Both are
# matched against an already-stripped line, so leading/trailing whitespace is gone.
_FENCE_OPEN_RE = re.compile(r"`{3,}\s*([A-Za-z0-9_+.\-]*)\s*")
_FENCE_CLOSE_RE = re.compile(r"`{3,}\s*")


def _strip_code_fences(code: str) -> str:
    """Unwrap a markdown code fence — but ONLY when the ENTIRE arg is one fenced
    block.

    Smaller / less-tool-trained models sometimes send the ``code`` tool-arg as
    ``"```python\\n...\\n```"`` (or ``"```\\n...\\n```"``) instead of bare source;
    we unwrap that. CRUCIALLY we never touch backticks that merely appear *inside*
    the source (docstrings, string literals, comments): the arg is treated as
    fence-wrapped only when its FIRST non-blank line is a bare fence opener (```
    optionally + one language token) AND its LAST non-blank line is a bare closing
    fence. Anything else is returned UNCHANGED.

    (The earlier implementation searched for the first ``` *anywhere* and sliced to
    the next one, which silently executed a wrong fragment — or raised a misleading
    SyntaxError — for valid Python that contained ``` in a string/docstring/comment.
    Returning the source untouched in the ambiguous cases is strictly safer: a
    genuinely malformed arg simply fails to compile and the model corrects it,
    rather than running code it never wrote.)

    Returns bare Python source. Raises :class:`SyntaxError` with an actionable
    message only when it IS a fence wrapper whose opener carries a non-Python tag.
    """

    stripped = code.strip()
    if "```" not in stripped:
        return stripped

    lines = stripped.splitlines()
    nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
    if len(nonblank) < 2:                                 # need an opener AND a closer line
        return stripped
    first_i, last_i = nonblank[0], nonblank[-1]

    opener = _FENCE_OPEN_RE.fullmatch(lines[first_i].strip())
    closer = _FENCE_CLOSE_RE.fullmatch(lines[last_i].strip())
    if not (opener and closer and first_i < last_i):
        return stripped                                   # `` ``` `` is embedded, not a wrapper -> leave it

    lang = (opener.group(1) or "").lower()
    if lang not in _PYTHON_FENCE_LANGS:
        raise SyntaxError(
            f"`code` tool-arg wrapped in ```{lang} fence. "
            f"Pass bare Python source as the `code` argument, not {lang}."
        )
    # Strictly the lines BETWEEN the opener and closer; trim only surrounding
    # blank lines (never inner indentation).
    return "\n".join(lines[first_i + 1:last_i]).strip("\n")


def _strip_md_fence(text: str) -> str:
    """Language-AGNOSTIC structural fence unwrap for parsing fenced model OUTPUT
    (e.g. a ```json ... ``` reply), as opposed to `_strip_code_fences`, which is
    specific to the Python `code` tool-arg and deliberately RAISES on a
    non-Python language tag.

    Unwraps only when `text`'s first non-blank line is a bare fence opener (```
    optionally + one language token) AND its last non-blank line is a bare closing
    fence — returning the inner body. Otherwise returns `text` UNCHANGED. NEVER
    raises, so unfenced JSON (or anything else) passes straight through.
    """
    stripped = text.strip()
    if "```" not in stripped:
        return text
    lines = stripped.splitlines()
    nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
    if len(nonblank) < 2:                                 # need an opener AND a closer line
        return text
    first_i, last_i = nonblank[0], nonblank[-1]
    opener = _FENCE_OPEN_RE.fullmatch(lines[first_i].strip())
    closer = _FENCE_CLOSE_RE.fullmatch(lines[last_i].strip())
    if not (opener and closer and first_i < last_i):
        return text                                       # embedded ```, not a wrapper -> leave it
    return "\n".join(lines[first_i + 1:last_i]).strip("\n")


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
    half = max(1, max_chars // 2)            # >=1: `text[-0:]` is the WHOLE string (-0 == 0), so a
                                             # max_chars of 0/1 used to print everything while the
                                             # header claimed all of it omitted (#9)
    if 2 * half >= raw_len:                  # head+tail would meet/overlap -> nothing is genuinely
        return text                          # hidden; just show the (short) text
    omitted = raw_len - 2 * half             # count what is ACTUALLY hidden (2*half chars shown),
                                             # NOT raw_len - max_chars (wrong for odd / clamped sizes)
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
    
    # Guard a falsy/`None` envelope (e.g. a kernel call that returned no result): `not result` plus
    # the `or ""` on each section below tolerate None / missing / empty. NOT a blanket "never raises"
    # — a TRUTHY NON-STRING section (e.g. stdout=5) would still raise on `.rstrip()`. That's
    # unreachable in practice (the sole producer, `execute_cell`, always yields string sections), so
    # this guards the reachable None/missing case rather than coercing arbitrary types.
    if not result:
        return "(no output)"
    parts: List[str] = []
    stdout = (result.get("stdout") or "").rstrip()
    if stdout:
        parts.append(_truncate_head_tail(stdout, max_section_chars))
    stderr = (result.get("stderr") or "").rstrip()
    if stderr:
        parts.append(f"stderr:\n{_truncate_head_tail(stderr, max_section_chars)}")
    return "\n".join(parts) if parts else "(no output)"


def _render_answer(parsed: Any) -> str:
    """Render an accepted RETURN value to text for the transcript.

    This is PURELY COSMETIC — the real answer object is kept separately by the
    caller — so it MUST NOT raise. A successful, constraint-validated RETURN can
    carry a value whose `model_dump_json()` (pydantic serialization edge) or
    `str()`/`__str__` raises; without this guard that display failure would
    abort an already-finished task and lose a correct result. On any failure we
    fall back to a safe placeholder naming the type.
    """
    try:
        if hasattr(parsed, "model_dump_json"):
            return parsed.model_dump_json()
        return str(parsed) if parsed is not None else ""
    except Exception:
        return f"<unserializable answer: {type(parsed).__name__}>"


def _safe_describe(constraint: Any) -> str:
    """`constraint.describe()` for an error/prompt string, but NEVER raising.

    describe() can itself raise (e.g. a factory built from an exhausted one-shot
    iterable — see the one_of/coercers describe bug — or `is_instance_of=<class with
    a throwing __repr__/__name__>`). It is called from runtime paths whose only
    job is to FORMAT an error or a reminder; a failure there must not escape and
    abort the task (the never-escape contract, #6). Fall back to a placeholder."""
    try:
        return constraint.describe()
    except Exception:
        return "<constraint description unavailable>"


def _coerce_budget(value: Any, default: int) -> int:
    """Coerce a model-supplied budget arg (`max_turns` / `return_budget`) to a
    non-negative int.

    The sub-LLM policies (natural_llm / react_llm / react_verifier_llm) take
    these from kwargs the MODEL controls in a cell. A `None` / non-int value
    would otherwise raise a raw `TypeError` at the `range(...)` bound, and a
    negative value would silently yield zero rounds. Normalize: `None` or any
    non-int (incl. bool) -> `default`; a negative int -> 0 (clamped)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(0, value)


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
    
    # a malformed `usage` must not crash the root loop. `response` is normalized to a dict by the
    # caller, but the NESTED `usage` is not — a non-dict usage (str/int/list) would AttributeError on
    # `.get`, and a non-numeric token sub-field would ValueError on `int()`. Coerce both defensively.
    usage = response.get("usage")
    if not isinstance(usage, dict):
        usage = {}

    def _as_int(v: Any) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0
    tok_total = _as_int(usage.get("total_tokens", 0))
    pt = _as_int(usage.get("prompt_tokens", 0))
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

def _format_unexpected_validate_error(
    constraint: Type[Constraint],
    err: BaseException,
) -> str:
    """Format a NON-ConstraintViolation error raised by ``constraint.validate``
    (a buggy/edge predicate, a wrong-typed value, etc.) for re-injection so the
    model can adjust the RETURN value — instead of the error escaping the React
    loop and aborting the whole task."""

    return (
        f"RETURN value could not be validated — the output constraint's validator "
        f"raised {type(err).__name__}: {err}\n\n"
        f"Required: {_safe_describe(constraint)}\n\n"
        f"Fix the object passed to RETURN(...) (it may be the wrong type/shape)."
    )


def _format_constraint_error(
    constraint: Type[Constraint],
    err: ConstraintViolation,
) -> str:
    """Format a Constraint violation for re-injection into a python tool
    result's stderr so the model sees what to fix on the next round."""

    return (
        f"RETURN value failed the configured output constraint:\n"
        f"{err}\n\n"
        f"Required: {_safe_describe(constraint)}\n\n"
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
            f"      {_safe_describe(constraint)}\n"
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
      which the caller appends to the python tool result's stderr so the model
      sees what to fix next round. This covers BOTH a ``ConstraintViolation``
      (formatted via ``_format_constraint_error``) AND any OTHER exception from
      ``validate`` (a buggy/edge predicate, a wrong-typed value; formatted via
      ``_format_unexpected_validate_error``). A validator error must never escape
      the React loop and abort the whole task — it is surfaced for retry instead.
      Only ``BaseException`` non-``Exception`` control-flow (KeyboardInterrupt,
      SystemExit, the RETURN sentinel) is left to propagate.
    """
    if constraint is None:
        return True, raw_return_value
    try:
        validated = constraint.validate(raw_return_value)
    except ConstraintViolation as cv:
        return False, _format_constraint_error(constraint, cv)
    except Exception as e:
        return False, _format_unexpected_validate_error(constraint, e)
    return True, validated
