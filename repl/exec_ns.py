"""`exec_ns` — the shared REPL code-runner.

Runs a code STRING in a namespace the way PLM's kernel runs a cell and the way
`react_llm` runs a sub-agent's tool code: ONE dict used as BOTH globals and locals,
stdout+stderr captured, any exception's traceback captured INTO the output (the
caller's loop never crashes on user/model code), and a linecache slot so tracebacks
show the real source lines.

This is the single core every "run this code" site builds on:
  * `react_llm._exec` — the sub-agent tool runner (adds RETURN/constraint handling
    via `on_return`); behavior is identical to the old inline version.
  * verifiers — apply an approved trajectory edit against a restricted namespace.
  * any policy / PLM that holds code-as-data and wants to run it.

Safety is the CALLER's namespace, not this function's: pass a restricted
`{"__builtins__": <small dict>}` in `ns` to sandbox (e.g. a verifier applying a
proposed edit), or a full namespace for a trusted REPL. `exec_ns` only runs `code`
in whatever `ns` it's given.

Stdlib-only, so it imports cleanly in the venv'd subprocess kernel and creates no
import cycle (no policy/repl imports).
"""

from __future__ import annotations

import io
import linecache
import traceback
import types
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, Optional, Tuple


# ===== sealed-namespace helpers — the react_llm sub-LLM ns is a CONTAINMENT boundary (D1) =====
#
# A sub-LLM's ns must hold ONLY `__builtins__` + `RETURN` + deliberate grants. Two
# leaks made that a fiction: a FULL `builtins` module (so `__import__('sys').modules
# ['__main__']` reached the kernel + both gates) and a `RETURN` whose `__globals__`
# WAS the kernel `__main__` (so `RETURN.__globals__['_PLM_POLICIES']` did the same).
# These helpers close the one-liner / accidental doors:
#   * `safe_builtins()` removes the import/code-eval/file builtins,
#   * `make_sealed_return()` rebuilds RETURN over a MINIMAL globals dict.
# What they CANNOT close (fundamental to in-process Python): the deliberate
# `().__class__.__base__.__subclasses__()` walk eventually finds some reachable
# function whose `__globals__['__builtins__']` is the real one -> `__import__` again.
# TRUE containment needs subprocess/container isolation (a production concern). This
# is the same irreducible residual base_verifier's restricted exec has.

class _REPLReturn(BaseException):
    """Loop-termination sentinel a SEALED `RETURN` raises. Lives in THIS clean,
    stdlib-only module on purpose: a sub-LLM reaching `RETURN.__globals__` finds only
    this class — whose module has no policy registry / depth gate / blessed-caller set
    (just stdlib). `exec_ns` matches it BY NAME, so the kernel's own `_REPLReturn` (for
    trusted cells) and this one both terminate identically."""
    def __init__(self, value: Any) -> None:
        self.value = value


# Builtins a sealed sub-LLM must NOT get: module import + code-eval + filesystem.
_UNSAFE_BUILTINS = frozenset({"__import__", "eval", "exec", "compile", "open"})


def safe_builtins() -> Dict[str, Any]:
    """A curated `__builtins__` dict for a sealed namespace: every builtin EXCEPT the
    out-of-sandbox ones (`__import__`/`eval`/`exec`/`compile`/`open`). `print`/`len`/
    `range`/`dict`/`class`-machinery all work; `import os` raises ImportError (the
    sub-LLM gets capabilities only via deliberate kwargs grants, never ambiently — a
    GRANTED callable still imports fine, since it runs in its OWN module globals). A
    fresh dict per call so a cell mutating it cannot poison the next run."""
    import builtins
    return {n: getattr(builtins, n) for n in dir(builtins) if n not in _UNSAFE_BUILTINS}


def make_sealed_return(safe_b: Dict[str, Any]):
    """A `RETURN(value)` for a sealed sub-LLM ns. **BEHAVES IDENTICALLY** to the
    kernel's RETURN for every caller — it raises `_REPLReturn(value)`, which
    `exec_ns`/`_exec` match by NAME to terminate the loop with `value`. The ONLY change
    is containment: built via `types.FunctionType` over a MINIMAL globals dict, so
    `RETURN.__globals__` is `{__builtins__: <curated>, _REPLReturn}` — NOT the kernel
    `__main__` — closing the `RETURN.__globals__['_PLM_POLICIES']` / `['__import__']`
    one-liners. `safe_b` MUST be the same curated builtins seeded into the ns."""
    def _RETURN(value):
        raise _REPLReturn(value)
    g = {"__builtins__": safe_b, "_REPLReturn": _REPLReturn}
    return types.FunctionType(_RETURN.__code__, g, "RETURN")


def exec_ns(
    code: str,
    ns: Dict[str, Any],
    *,
    slot: str = "exec_ns",
    on_return: Optional[Callable[[Any], Tuple[Optional[str], Any]]] = None,
) -> Tuple[str, Optional[str], Any]:
    """Run `code` in namespace `ns` (one dict = globals AND locals; mutated in place).

    Returns ``(output, term, val)``:
      * ``output`` — captured stdout+stderr, or ``"(no output)"``. stdout emitted
        BEFORE an exception is PRESERVED: prints land in the stdout buffer, the
        exception's traceback in the stderr buffer, and `output` is the two
        CONCATENATED (stdout THEN traceback) — so a print-then-error run surfaces
        BOTH to the caller, never just the error. The traceback is captured HERE,
        NOT raised, so a REPL loop never crashes on user/model code; that also means
        a CALLER's surrounding try/except is for ITS OWN orchestration (fence-strip,
        depth descend, ...), NOT for the run's code errors — those already sit inside
        `output`. A `SyntaxError` from `compile` and any ordinary exception are
        captured; `KeyboardInterrupt`/`SystemExit`/`GeneratorExit` are RE-RAISED so an
        operator Ctrl-C (or a deliberate exit) can actually interrupt the loop. The
        `exec_ns` frame is stripped from captured tracebacks.
      * ``term`` / ``val`` — set only when `code` raises the REPL RETURN sentinel
        (``_REPLReturn``, matched BY NAME like the kernel loop, so no import coupling):
        ``on_return(value) -> (term, val)`` decides the outcome; the default (no
        `on_return`) finalizes it as ``("return", value)``. Otherwise ``(None, None)``.

    `slot` keys a `linecache` entry (``"<{slot}>"``) so tracebacks show real source;
    callers that run repeatedly should pass a unique slot per run (e.g. a per-round
    id) to avoid clobbering a still-referenced earlier source.

    Why ONE namespace (globals == locals): a `def`/`class` in `code` captures only
    its module globals, so a split globals/locals would hide injected names and
    sibling helpers from a function body. One dict makes everything behave like a
    global — exactly like a REPL cell.
    """
    fn = f"<{slot}>"
    linecache.cache[fn] = (len(code), None, code.splitlines(True), fn)
    buf_o, buf_e = io.StringIO(), io.StringIO()
    term: Optional[str] = None
    val: Any = None
    with redirect_stdout(buf_o), redirect_stderr(buf_e):
        try:
            exec(compile(code, fn, "exec"), ns)         # ONE namespace = globals AND locals
        except BaseException as e:
            if type(e).__name__ == "_REPLReturn":       # termination sentinel (name-based, like the kernel loop)
                proposed = getattr(e, "value", None)
                term, val = (
                    on_return(proposed) if on_return is not None else ("return", proposed)
                )
            elif isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise                                   # operator/runtime control — NEVER swallow; a Ctrl-C must interrupt (K-F9/R-F7)
            else:
                # Write the traceback STRAIGHT to buf_e: a cell rebinding `sys.stderr`
                # defeats redirect_stderr, so `print_exc()` could vanish — formatting to
                # the buffer is rebind-proof. Drop THIS frame (the `exec(compile(...))`
                # line) so the captured output shows only the run's own source.
                tb = e.__traceback__
                buf_e.write("".join(traceback.format_exception(type(e), e, tb.tb_next if tb else None)))
    return (buf_o.getvalue() + buf_e.getvalue()) or "(no output)", term, val
