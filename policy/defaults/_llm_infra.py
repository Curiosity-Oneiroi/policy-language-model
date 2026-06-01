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
            f"policies (natural_llm / react_llm). To call the model, use those "
            f"policies; do not author your own LLM primitive."
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
    silently grant a budget."""
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
    act phase. Used by react_llm (and the deferred react_llm_verifier) around
    their code execution. natural_llm is single-shot (no act phase) and does
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
    `depth=int`         -> set `_LLM_DEPTH = min(int(depth), current)` for
                           the scope. PLM can voluntarily LOWER, but cannot
                           raise above the sealed ceiling. `depth=999` is
                           silently clamped to `current`.
    `depth=garbage`     -> treated as None (no-op).
    """
    _check_blessed_caller("llm_call")
    try:
        d = None if depth is None else int(depth)
    except (TypeError, ValueError):
        d = None

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


# AFW backend class-name → module suffix dispatch table.
_MOD = {
    "SlateBackend":     "slate_backend",
    "OpenAIBackend":    "openai_backend",
    "VLLMBackend":      "vllm_backend",
    "AnthropicBackend": "anthropic_backend",
}


def _make_backend():
    """Reconstruct the parent PLM's AFramework backend in this kernel from
    `_PLM_BACKEND_SPEC` env. Returns a SYNC handle that is a THIN transport
    wrapper — no depth logic inside `.generate()`. The depth check lives in
    the policy bodies (each LLM policy calls `check_depth_or_raise()` before
    its `backend.generate`).

    The sync wrapper builds a fresh AFW backend instance per call via
    `from_spec(spec, None)`, then calls the UNWRAPPED `.generate` (bypassing
    the `@resource(...)` decorator) so the kernel doesn't need to initialize
    AFW's ResourceManager (which lives in the AFW worker process, not here,
    and would also pull in `redis` as a transitive dep). Fresh client per
    call sidesteps cross-event-loop httpx reuse bugs (cheap; no network
    until `.generate`).
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
    mod = importlib.import_module("AFramework.model_backend." + _MOD[cls_name])
    backend_cls = getattr(mod, cls_name)

    class _SyncBackend:
        model = spec.get("model")

        def generate(self, messages, tools=None, **kw):
            # NO depth logic here — this is a thin transport. The policy body
            # calls `check_depth_or_raise()` BEFORE each .generate(), so the
            # backend only handles the AFW round-trip.
            #
            # BYPASS @resource: AFW backends decorate `generate` with
            # @resource(...), which calls get_resource_manager() — a
            # process-local singleton init'd by the AFW worker, NOT by the
            # kernel subprocess. Initializing it here would also pull in
            # `redis` (transitive import in resource_manager). functools.wraps
            # exposes the unwrapped async fn at `.__wrapped__`; call that
            # directly. Consistent with "depth lives in the policy, not the
            # backend" — AFW rate/concurrency stays in the worker; the
            # kernel-side handle is purely transport.
            async def _run():
                be = backend_cls.from_spec(spec, None)
                try:
                    raw = type(be).generate
                    fn = getattr(raw, "__wrapped__", raw)   # bypass @resource if present
                    return await fn(be, messages, tools=tools, **kw)
                finally:
                    c = getattr(be, "client", None)
                    if c is not None and hasattr(c, "aclose"):
                        await c.aclose()
            return asyncio.run(_run())

    return _SyncBackend()
