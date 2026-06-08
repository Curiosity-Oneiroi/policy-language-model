"""Pure, dependency-free helpers shared across the policy layer.

This module imports ONLY stdlib (`ast`, `textwrap`) so it sits at the bottom
of the policy import graph — `proxy` and `decorator` both depend on it, and it
depends on nothing of ours. That placement is what breaks the otherwise-present
proxy<->decorator cycle (`proxy._rewrite` needs the @policy-stripper; `decorator`
needs `_FunctionPolicy`).

Two responsibilities:
  * Source-string edit math (`_compute_edit`/`_compute_insert`/`_compute_delete`)
    — ONE source of truth for the edit + no-op semantics, used by BOTH the
    function-proxy methods and the class-policy helpers so they can't drift.
  * Verbatim source extraction / @policy-stripping (`_slice_policy_source`,
    `_strip_policy_decorator`, ...) — all line-based / comment-preserving, never
    `ast.unparse` (which would normalize away comments and formatting).
"""

from __future__ import annotations

import ast
import textwrap


# --- @policy decorator detection / stripping (line-based, comment-preserving) ---

def _is_policy_decorator(d: ast.expr) -> bool:
    """True if AST decorator node `d` is `@policy` (bare or called)."""
    t = d.func if isinstance(d, ast.Call) else d
    return isinstance(t, ast.Name) and t.id == "policy"


def _iter_funcs(v):
    """Yield the CODE-BEARING function object(s) of a class-dict value — a plain
    method, a static/classmethod (via __func__), or a property's fget/fset/fdel.

    One definition shared by closure-detection (`_warn_if_captures_locals`) and
    __class__-cell rebinding (`_rebind_class_cells`) so both treat every method
    kind identically (can't drift). Only yields objects that have `__code__`, so
    callers can use `fn.__code__` unconditionally (a static/classmethod or
    property wrapping a C builtin yields nothing).
    """
    if isinstance(v, (staticmethod, classmethod)):
        cands = [getattr(v, "__func__", None)]
    elif isinstance(v, property):
        cands = [v.fget, v.fset, v.fdel]
    else:
        cands = [v]
    for f in cands:
        if f is not None and hasattr(f, "__code__"):
            yield f


def _node_span(node: ast.AST) -> tuple[int, int]:
    """(first_line, last_line) of a def/class INCLUDING its decorators."""
    start = node.lineno
    for d in node.decorator_list:
        start = min(start, d.lineno)
    return start, node.end_lineno


def _policy_decorator_lines(node: ast.AST) -> set[int]:
    """Line numbers occupied by @policy decorators on a def/class node."""
    drop: set[int] = set()
    for d in node.decorator_list:
        if _is_policy_decorator(d):
            for ln in range(d.lineno, (d.end_lineno or d.lineno) + 1):
                drop.add(ln)
    return drop


def _slice_policy_source(full: str, node: ast.AST) -> str:
    """VERBATIM source of `node` (def/class) with @policy decorator lines removed
    and common indentation stripped (so nested policies parse). Keeps comments,
    blank lines, and any NON-@policy decorators. Line-based — never ast.unparse.
    """
    lines = full.splitlines(keepends=True)
    start, end = _node_span(node)
    drop = _policy_decorator_lines(node)
    seg = "".join(lines[i - 1] for i in range(start, end + 1) if i not in drop)
    return textwrap.dedent(seg)


def _strip_policy_decorator(source: str) -> str:
    """Remove @policy decorator lines from top-level defs/classes, preserving
    everything else verbatim (comments, formatting, other decorators). Defensive
    against `_rewrite(source-that-still-has-@policy)` -> infinite re-decoration.

    Raises SyntaxError if `source` doesn't parse — callers run this inside their
    own try so the SyntaxError is surfaced as a note, not a raise.
    """
    tree = ast.parse(source)
    drop: set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            drop |= _policy_decorator_lines(node)
    if not drop:
        return source
    lines = source.splitlines(keepends=True)
    return "".join(l for i, l in enumerate(lines, 1) if i not in drop)


# --- source-string edit math (one source of truth; None == no-op) ---

def _compute_edit(src: str, old: str, new: str, replace_all: bool):
    """New source from a find/replace, or None for a no-op."""
    if not old:                                       # empty `old` -> str.replace would
        return None                                   # splice between every char
    ns = src.replace(old, new, -1 if replace_all else 1)
    return ns if ns != src else None


# The FULL set of line boundaries `str.splitlines()` breaks on — checking only
# '\n'/'\r' in the newline guards below would mis-handle a line that ends in one
# of the others (e.g. \v in a string literal, \u2028 in a comment): the guard
# would wrongly append a spurious '\n' right after the existing separator.
_LINE_BOUNDARIES = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"  # full str.splitlines() set


def _compute_insert(src: str, after_line: int, content: str):
    """New source after inserting `content` after `after_line` lines, or None."""
    if (not content or not isinstance(after_line, int) or isinstance(after_line, bool)
            or after_line < 0):                       # after_line: a 0-based int count;
        return None                                   # non-int -> no-op, not a raw TypeError
    L = src.splitlines(keepends=True)
    before = "".join(L[:after_line])
    tail = "".join(L[after_line:])
    # If the prefix doesn't already END on a line boundary (e.g. inserting after
    # the final line of a source whose last line has no trailing newline), add one
    # so `content` starts on its own line instead of being glued onto that line.
    if before and before[-1] not in _LINE_BOUNDARIES:
        before += "\n"
    # Symmetric guard on the OTHER side: for a mid-source insert (there ARE
    # following lines) where `content` doesn't end on a boundary, add one so
    # `content` isn't glued onto the first following line.
    if tail and content and content[-1] not in _LINE_BOUNDARIES:
        content += "\n"
    return before + content + tail


def _compute_delete(src: str, start: int, end):
    """New source after deleting 1-based lines [start, end], or None for a no-op."""
    if not isinstance(start, int) or isinstance(start, bool):   # non-int -> no-op,
        return None                                              # not a raw slice TypeError
    end = start if end is None else end
    if not isinstance(end, int) or isinstance(end, bool):
        return None
    if start < 1 or start > end:                       # lines are 1-based; start=0 invalid
        return None                                    # (negative slicing would duplicate)
    L = src.splitlines(keepends=True)
    if start - 1 >= len(L):
        return None
    ns = "".join(L[:start - 1] + L[end:])
    return ns if ns != src else None


# --- top-level def/class rename (for duplicate_policy) ---

import re as _re                                       # local import to keep stdlib-only contract


def _compute_rename(src: str, new_name: str) -> str:
    """Rename ONLY the top-level def/class name in `src` to `new_name`.

    Anchored on the `def`/`async def`/`class` keyword + the original name + a
    word boundary, so internal references to the same name (e.g. a recursive
    call inside the body, or an arg with the same name) are left untouched.
    Non-`@policy` decorators above the def are preserved verbatim. Used by
    `duplicate_policy` to fork a policy under a new name; internal recursive
    calls keep resolving to the ORIGINAL by design (predictable + simple —
    if the user wants the copy to self-reference, they `._edit(...)` after).
    """
    tree = ast.parse(src)
    # First top-level FunctionDef / AsyncFunctionDef / ClassDef.
    node = next(
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    lines = src.splitlines(keepends=True)
    i = node.lineno - 1                                # node.lineno is 1-based; lines is 0-based
    lines[i] = _re.sub(
        r"(\b(?:async\s+def|def|class)\s+)" + _re.escape(node.name) + r"\b",
        r"\g<1>" + new_name,
        lines[i],
        count=1,
    )
    return "".join(lines)
