"""Developer-controlled helpers for the immutable LLM-default policies.

NOT a policy. Real Python module imported (in-call) by the immutable
un-duplicable LLM-policy bodies. PLM has no `list_policies` / `read_policy` /
`duplicate` path to this code — and each public entry ENFORCES a blessed-caller
frame check so even if PLM imports the module, the public functions refuse to
run from anywhere except the sanctioned policy bodies.

Public surface (importable from LLM-policy bodies):
  * `_make_backend()`         — reconstruct AFW backend from `_PLM_BACKEND_SPEC`.
  * `check_depth_or_raise()`  — hard ceiling on `_LLM_DEPTH`; refuse at <= 0.
  * `llm_call(depth=None)`    — voluntary lowering CM.
  * `descend()`               — decrement-around-act-phase CM.
  * `LLMDepthExceeded`        — exception type the gate raises.
  * `_BLESSED_CALLERS`        — frozenset; populated by the bootstrap loader.
  * `_check_blessed_caller`   — frame-2 caller-code-object check.

Module-internal (NOT imported by policy bodies):
  * `_LLM_DEPTH`              — sealed ContextVar; only mutated by `llm_call` /
                                `descend` (both blessed-caller-gated).
  * `_root_depth`             — read `AGENT_DEPTH` env (fail-closed default 0).
  * `_remaining`              — current value or root.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar


class LLMDepthExceeded(RuntimeError):
    """Raised when an LLM policy body's `check_depth_or_raise()` finds the
    remaining `_LLM_DEPTH` budget at <= 0. The hard refuse-check; the backend
    wrapper is depth-agnostic."""


# Populated at boot by the PREFIX bootstrap loader (and refreshed after
# rehydrate) via `_bless_llm_callers()`. A frozenset means PLM cannot `.add()`
# even if it obtains a reference. Module-attribute reassignment
# (`_llm_infra._BLESSED_CALLERS = my_set`) remains the irreducible
# Python-no-privacy boundary (documented in the depth section).
_BLESSED_CALLERS: frozenset = frozenset()


def _check_blessed_caller(name: str) -> None:
    """Raise unless the function 2 frames up is one of the blessed LLM-policy
    bodies. (frame 0 = this helper; frame 1 = the public fn; frame 2 = caller.)

    NB: the check runs at the public function's CALL TIME — for `llm_call`
    and `descend` (both return @contextmanager-wrapped CMs), this means the
    check fires when the body evaluates `llm_call(...)` / `descend(...)`,
    NOT inside the generator (which would run at __enter__ with extra
    `_GeneratorContextManager` frames between).
    """
    caller_code = sys._getframe(2).f_code
    if caller_code not in _BLESSED_CALLERS:
        raise RuntimeError(
            f"_llm_infra.{name}: callable ONLY from inside the sanctioned LLM "
            f"policies (natural_llm / react_llm / react_verifier_llm). To call "
            f"the model, use those policies; do not author your own LLM primitive."
        )


# Sealed source of truth for LLM-recursion depth. Lives in a non-policy module
# never bound in __main__; PLM has no `read_policy` / `duplicate` / cell-global
# path to write it (the only "escape" is a deliberate
# `import plm.policy.defaults._llm_infra; _llm_infra._LLM_DEPTH.set(...)`,
# which is the documented Python-no-privacy boundary).
_LLM_DEPTH: ContextVar[int | None] = ContextVar("_LLM_DEPTH", default=None)


def _root_depth() -> int:
    """Read the kernel-subprocess root depth from `AGENT_DEPTH` (set by parent
    PLM at session start). Fail-closed: any parse failure or missing var means
    root=0, which makes every LLM call refuse — a misconfigured kernel cannot
    silently grant a budget. In the KERNEL this is read ONCE at boot to seed
    `_LLM_DEPTH` (PREFIX), so a cell can't poison the ceiling by reassigning the
    env var; the live read here remains the fallback for unseeded
    (e.g. in-process test) contexts where `_LLM_DEPTH` is None."""
    try:
        return int(os.environ.get("AGENT_DEPTH", "0"))
    except Exception:
        return 0


def _remaining() -> int:
    d = _LLM_DEPTH.get()
    return _root_depth() if d is None else d


def check_depth_or_raise() -> None:
    """Hard ceiling: raises `LLMDepthExceeded` when remaining `_LLM_DEPTH` <= 0.
    The policy body calls this before every `backend.generate(...)` — depth
    lives in the policy, not the backend wrapper."""
    if _remaining() <= 0:
        raise LLMDepthExceeded(
            f"LLM-recursion depth budget exhausted (root={_root_depth()}). "
            "Compose the existing LLM policies; do not deepen further."
        )


def descend():
    """Decrement the LLM-depth budget for the dynamic extent of the agent's
    act phase. Used by react_llm and react_verifier_llm around their act-phase
    code execution — and, in react_verifier_llm, around the per-round verifier
    hook, so a verifier's react_llm/natural_llm circuits are accounted depth-1.
    natural_llm is single-shot (no act phase) and does
    NOT descend — it only checks before its generate calls.

    The blessed-caller check fires HERE (function-call time, frame 2 = the
    policy body), NOT inside the returned CM's generator body.
    """
    _check_blessed_caller("descend")

    @contextmanager
    def _cm():
        tok = _LLM_DEPTH.set(_remaining() - 1)
        try:
            yield
        finally:
            _LLM_DEPTH.reset(tok)
    return _cm()


def llm_call(depth=None):
    """Voluntary lowering CM. Called at the start of each LLM policy invocation.

    `depth=None`        -> no-op (use current scope; nested LLM calls inherit
                           the contextvar value set by an outer descend()).
    `depth=int (>= 0)`  -> set `_LLM_DEPTH = min(depth, current)` for the scope.
                           PLM can voluntarily LOWER, but cannot raise above the
                           sealed ceiling. `depth=999` is silently clamped to `current`.

    Depth is validated STRICTLY: it must be None or a NON-NEGATIVE INTEGER. A negative
    value, a non-integer float (e.g. `1.5`), a bool, or any non-int RAISES ValueError rather
    than being silently truncated/clamped — a bad depth is a bug, surfaced loudly.
    """
    _check_blessed_caller("llm_call")
    if depth is None:
        d = None
    elif isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError(
            "llm_call: depth must be None or a non-negative integer; got " + repr(depth)
            + " (no silent coercion — a float like 1.5, a negative, or a non-int is rejected)")
    else:
        d = depth

    @contextmanager
    def _cm():
        if d is None:
            yield
            return
        tok = _LLM_DEPTH.set(min(d, _remaining()))
        try:
            yield
        finally:
            _LLM_DEPTH.reset(tok)
    return _cm()


# Local model-backend class-name → module suffix (plm.model_backend.*).
_MOD = {
    "SlateBackend":     "slate_backend",
    "OpenAIBackend":    "openai_backend",
    "VLLMBackend":      "vllm_backend",
    "AnthropicBackend": "anthropic_backend",
}


async def _aclose_backend(be):
    """Close a backend's underlying HTTP transport after a one-shot `.generate`.

    The ported backends don't agree on the close method: `httpx.AsyncClient`
    (SlateBackend) exposes `aclose()`, while the `openai`/`anthropic` async
    clients expose an async `close()` and no `aclose`. The old cleanup only
    handled `aclose`, so it silently leaked the openai/anthropic transports
    every call. Handle both spellings (and a sync `close()` for safety): prefer
    `aclose`, else await/await-or-call `close`. Never let cleanup mask the real
    result — swallow close-time errors.
    """
    # Read the ALREADY-instantiated client from the instance __dict__ — never via
    # getattr, because OpenAIBackend.client is a @property that CONSTRUCTS a client
    # on access (so if .generate raised before first use, getattr would build a
    # throwaway client just to close it). Eager backends store it under "client";
    # OpenAIBackend's lazy holder is "_client" (None until first use).
    _d = getattr(be, "__dict__", None) or {}      # __slots__ backend has no __dict__:
    c = _d.get("client") or _d.get("_client")     # treat as "no cached client", don't AttributeError
    if c is None:
        return
    closer = getattr(c, "aclose", None) or getattr(c, "close", None)
    if closer is None:
        return
    try:
        res = closer()
        if inspect.isawaitable(res):
            await res
    except Exception:
        pass                                            # transport teardown is best-effort


def _make_backend():
    """Reconstruct the parent PLM's model backend in this kernel from
    `_PLM_BACKEND_SPEC` env. Returns a SYNC handle that is a THIN transport
    wrapper — no depth logic inside `.generate()`. The depth check lives in
    the policy bodies (each LLM policy calls `check_depth_or_raise()` before
    its `backend.generate`).

    Builds a fresh backend instance per call via `from_spec(spec, None)` and
    awaits its async `.generate`. Fresh client per call sidesteps cross-event-
    loop httpx reuse bugs (cheap; no network until `.generate`). The backends
    live in `plm.model_backend` — ported from AFramework with the @resource /
    ResourceManager / registry couplings stripped — so the kernel imports
    nothing from AFramework.
    """
    _check_blessed_caller("_make_backend")
    raw = os.environ.get("_PLM_BACKEND_SPEC")
    if not raw:
        raise RuntimeError(
            "_make_backend: no _PLM_BACKEND_SPEC in env "
            "(PLM sets it from its model_backend)."
        )
    spec = json.loads(raw)
    cls_name = spec["model_backend_class_name"]
    if cls_name not in _MOD:
        raise RuntimeError(
            f"_make_backend: unsupported backend {cls_name!r}; "
            f"known backends: {sorted(_MOD)}."
        )
    mod = importlib.import_module("plm.model_backend." + _MOD[cls_name])
    backend_cls = getattr(mod, cls_name)

    class _SyncBackend:
        def generate(self, messages, tools=None, **kw):
            # NO depth logic here — this is a thin transport. The policy body
            # calls `check_depth_or_raise()` BEFORE each .generate(), so the
            # backend only handles the API round-trip. The local backends carry
            # no @resource/ResourceManager wrapper (that AFW coupling was
            # stripped on the port), so we just await `.generate` directly.
            async def _run():
                be = backend_cls.from_spec(spec, None)
                try:
                    return await be.generate(messages, tools=tools, **kw)
                finally:
                    await _aclose_backend(be)
            return asyncio.run(_run())

    return _SyncBackend()
