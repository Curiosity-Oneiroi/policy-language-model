"""Focused unit tests for the six small helpers in `plm._react_helper`
that were previously only exercised indirectly through PLM's React loop.

Covered:
  * _truncate_head_tail
  * _format_repl_output          (incl. the #10 None-guard regression)
  * _track_model_call
  * _build_last_round_reminder
  * _format_constraint_error
  * _strip_md_fence
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from plm._react_helper import (
    _build_last_round_reminder,
    _format_constraint_error,
    _format_repl_output,
    _strip_md_fence,
    _track_model_call,
    _truncate_head_tail,
)
from plm.constraint import Constraint, ConstraintViolation


# ---------------------------------------------------------------------------
# _truncate_head_tail
# ---------------------------------------------------------------------------


def test_truncate_head_tail_short_string_returned_unchanged():
    """A string shorter than max_chars is returned verbatim — no header,
    no ellipsis, no allocation."""
    s = "hello"
    assert _truncate_head_tail(s, 100) == s


def test_truncate_head_tail_exact_length_returned_unchanged():
    """At the boundary (len == max_chars) the string still fits."""
    s = "x" * 20
    assert _truncate_head_tail(s, 20) == s


def test_truncate_head_tail_long_string_keeps_head_and_tail():
    """A long string yields a header with the true length and the omitted
    count, then half from the start and half from the end."""
    s = "A" * 50 + "B" * 50                                     # 100 chars total
    out = _truncate_head_tail(s, 20)
    # Header reports REAL length and how many chars are omitted.
    assert out.startswith("[total 100 chars; 80 omitted]")
    # Head + tail are visible; the middle marker shows the omitted count too.
    assert "AAAAAAAAAA" in out                                  # first 10 chars
    assert "BBBBBBBBBB" in out                                  # last 10 chars
    assert "(80 characters omitted)" in out
    # And the original was longer than what we returned.
    assert len(out) < len(s) + 200                              # bounded by header overhead


def test_truncate_head_tail_odd_max_chars_uses_floor_div():
    """`max_chars=5` -> head=tail=2 (floor div), one char dropped from the
    larger of the two halves; we don't crash and we don't include all 100 chars."""
    s = "A" * 100
    out = _truncate_head_tail(s, 5)
    assert "[total 100 chars" in out
    # Boundedness: we return WAY less than the raw text.
    assert len(out) < 200


def test_truncate_head_tail_tiny_max_chars_is_honest():
    """#9: with max_chars <= 1 the old code collapsed `half` to 0, and `text[-0:]` is the WHOLE
    string (-0 == 0) — so it printed everything while the header claimed all of it omitted. Now
    `half` clamps to >=1, `omitted` reflects what is ACTUALLY hidden, and a too-small budget degrades
    to a real head+tail (or the full short text), never a lie."""
    s = "hello world"                                        # 11 chars
    for mc in (0, 1):
        out = _truncate_head_tail(s, mc)
        assert s not in out                                  # NOT the whole string (the old bug)
        assert "[total 11 chars; 9 omitted]" in out          # 2 shown (h + d), 9 hidden — honest
        assert "(9 characters omitted)" in out
    # negative budget doesn't blow up or over-count
    assert "9 omitted" in _truncate_head_tail(s, -4)
    # degenerate: budget so small head+tail would overlap a SHORT string -> show it whole, no lie
    assert _truncate_head_tail("hi", 0) == "hi"


# ---------------------------------------------------------------------------
# _format_repl_output  (incl. #10 — None envelope must not raise)
# ---------------------------------------------------------------------------


def test_format_repl_output_none_envelope_does_not_raise():
    """#10 regression: a `None` result must produce a sentinel string, NOT
    an AttributeError from `None.get(...)`. Other helpers in the module
    guard the same way; this one matches that contract."""
    assert _format_repl_output(None) == "(no output)"


def test_format_repl_output_empty_dict_returns_sentinel():
    """An envelope with no stdout/stderr is the same as 'nothing to show'."""
    assert _format_repl_output({}) == "(no output)"


def test_format_repl_output_stdout_only():
    out = _format_repl_output({"stdout": "hello\n"})
    assert out == "hello"                                       # rstripped, no stderr block


def test_format_repl_output_stderr_only_is_prefixed():
    out = _format_repl_output({"stderr": "oops"})
    assert out == "stderr:\noops"


def test_format_repl_output_both_streams():
    out = _format_repl_output({"stdout": "out", "stderr": "err"})
    # stdout first, then a separate stderr block.
    assert out == "out\nstderr:\nerr"


def test_format_repl_output_truncates_per_section():
    """Each section gets its own head/tail cap — one section's chop must not
    bleed into the other's budget."""
    long_stdout = "X" * 200
    out = _format_repl_output({"stdout": long_stdout}, max_section_chars=20)
    assert "[total 200 chars" in out
    assert len(out) < len(long_stdout)


# ---------------------------------------------------------------------------
# _track_model_call
# ---------------------------------------------------------------------------


def _empty_metrics() -> Dict[str, Any]:
    return {"model_calls": [], "total_tokens": 0}


def test_track_model_call_appends_metrics_and_updates_tokens():
    metrics = _empty_metrics()
    token_state: Dict[str, Any] = {}
    response = {
        "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
        "model": "gpt-X",
        "backend": "openai",
    }
    _track_model_call(response, call_start=0.0, metrics=metrics, token_state=token_state)
    assert token_state["cumulative_prompt_tokens"] == 100
    assert metrics["total_tokens"] == 125
    assert len(metrics["model_calls"]) == 1
    rec = metrics["model_calls"][0]
    assert rec["tokens_this_call"] == 125
    assert rec["cumulative_prompt_tokens"] == 100
    assert rec["model"] == "gpt-X"
    assert rec["backend"] == "openai"
    assert "call_duration" in rec


def test_track_model_call_metrics_none_still_updates_token_state():
    """`metrics=None` is a valid configuration (caller doesn't want a trace);
    the cumulative-prompt-tokens counter still advances so subsequent calls
    can compute deltas."""
    token_state: Dict[str, Any] = {}
    response = {"usage": {"prompt_tokens": 7, "total_tokens": 9}}
    _track_model_call(response, 0.0, None, token_state)
    assert token_state["cumulative_prompt_tokens"] == 7


def test_track_model_call_missing_usage_uses_zero_defaults():
    """A response with NO usage block must not crash — both counters stay 0."""
    metrics = _empty_metrics()
    token_state: Dict[str, Any] = {}
    _track_model_call({}, 0.0, metrics, token_state)
    # No prompt tokens reported -> cumulative_prompt_tokens NOT set (the helper
    # only writes when pt > 0); metrics still record the empty call so the
    # caller can see SOMETHING happened.
    assert "cumulative_prompt_tokens" not in token_state
    assert metrics["total_tokens"] == 0
    assert len(metrics["model_calls"]) == 1


# ---------------------------------------------------------------------------
# _build_last_round_reminder
# ---------------------------------------------------------------------------


def test_build_last_round_reminder_without_constraint():
    msg = _build_last_round_reminder(None)
    assert msg["role"] == "user"
    assert "RETURN" in msg["content"]
    assert msg["_plm_meta"] == "final_round_return_reminder"
    assert isinstance(msg["_timestamp"], float)


def test_build_last_round_reminder_with_constraint_includes_describe():
    """When a constraint IS configured, its describe() text is woven into
    the reminder so the model sees the contract it must satisfy."""

    class Ans(Constraint):
        x: Constraint.field(type=int)

    msg = _build_last_round_reminder(Ans)
    assert msg["role"] == "user"
    # Constraint surface (the field name and the contract phrasing).
    assert "constraint" in msg["content"].lower()
    assert "RETURN" in msg["content"]


def test_build_last_round_reminder_describe_failure_does_not_raise():
    """`_safe_describe` catches a throwing `describe()`. The reminder builder
    runs that path inline — a broken describe must not abort the React loop."""

    class Broken(Constraint):
        @classmethod
        def describe(cls):                                       # noqa: D401
            raise RuntimeError("boom")

    msg = _build_last_round_reminder(Broken)
    assert "constraint description unavailable" in msg["content"]


# ---------------------------------------------------------------------------
# _format_constraint_error
# ---------------------------------------------------------------------------


def test_format_constraint_error_includes_violation_text_and_required():
    """The formatter must surface BOTH the violation text and the contract,
    so the model can see what failed AND what to satisfy."""

    class Ans(Constraint):
        x: Constraint.field(type=int)

    try:
        Ans.validate({"x": "not an int"})
    except ConstraintViolation as cv:
        out = _format_constraint_error(Ans, cv)
    else:                                                        # pragma: no cover
        pytest.fail("validate should have raised ConstraintViolation")

    assert "RETURN value failed" in out
    assert "Required:" in out
    assert "Fix the object passed to RETURN" in out


def test_format_constraint_error_safe_on_broken_describe():
    """A broken describe() must NOT raise inside the error formatter — the
    error path is what surfaces the original problem to the model."""

    class Broken(Constraint):
        @classmethod
        def describe(cls):                                       # noqa: D401
            raise RuntimeError("boom")

    cv = ConstraintViolation("synthetic")
    out = _format_constraint_error(Broken, cv)
    assert "constraint description unavailable" in out
    assert "synthetic" in out


# ---------------------------------------------------------------------------
# _strip_md_fence
# ---------------------------------------------------------------------------


def test_strip_md_fence_unwraps_full_block():
    text = "```json\n{\"a\": 1}\n```"
    assert _strip_md_fence(text) == '{"a": 1}'


def test_strip_md_fence_no_language_tag():
    text = "```\nhello\n```"
    assert _strip_md_fence(text) == "hello"


def test_strip_md_fence_unfenced_passes_through():
    """Plain JSON (no fence) is returned unchanged — must NOT raise."""
    raw = '{"a": 1}'
    assert _strip_md_fence(raw) == raw


def test_strip_md_fence_unknown_language_does_not_raise():
    """Unlike _strip_code_fences (which is python-specific and raises on a
    foreign language), _strip_md_fence is language-AGNOSTIC and unwraps any
    fence; raising would break JSON-parsing callers."""
    text = "```yaml\na: 1\n```"
    assert _strip_md_fence(text) == "a: 1"


def test_strip_md_fence_embedded_backticks_left_alone():
    """A string that merely CONTAINS ``` (e.g. inside the body) is not a
    wrapper and must be returned unchanged."""
    raw = 'value with ``` backticks but no fence'
    assert _strip_md_fence(raw) == raw


def test_strip_md_fence_blank_lines_around_fence_tolerated():
    text = "\n\n```json\n[1, 2]\n```\n\n"
    assert _strip_md_fence(text) == "[1, 2]"
