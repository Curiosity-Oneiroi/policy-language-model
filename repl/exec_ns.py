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
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, Optional, Tuple


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
        `output`. (exec_ns catches `BaseException`, so even a `SyntaxError` from
        `compile` or a `SystemExit` lands in `output`, never escapes.)
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
            if type(e).__name__ == "_REPLReturn":       # name-based, like the kernel loop
                proposed = getattr(e, "value", None)
                term, val = (
                    on_return(proposed) if on_return is not None else ("return", proposed)
                )
            else:
                traceback.print_exc()                   # captured into buf_e for the caller
    return (buf_o.getvalue() + buf_e.getvalue()) or "(no output)", term, val
