"""plm.policy.defaults — the v1 default LLM policies + their developer-controlled
helpers.

Layout (per the plan):
  * `_llm_infra.py` — NOT a policy. Real Python module imported (in-call) by
    the LLM-policy bodies; owns `_make_backend()`, the depth ContextVar +
    `llm_call`/`descend`/`check_depth_or_raise`/`LLMDepthExceeded`, and the
    blessed-caller gate (`_BLESSED_CALLERS` / `_check_blessed_caller`).
  * `llm.py`, `react_auto.py`, `react_verifier_llm.py` — the immutable
    LLM-loop policy source files, and `base_verifier.py` — a MUTABLE +
    DUPLICABLE reference verifier. All are read as TEXT by
    `iter_default_policies` and replayed as synthetic cells via the bootstrap
    loader, NOT imported as Python modules — that's how the policy source
    extraction + linecache pipeline works uniformly for defaults + extras.
    Sealing/blessing is decided per-name by `_LLM_DEFAULT_POLICIES` (below), NOT
    by being a default file: `base_verifier` is a default file that is deliberately
    NOT in that set, so it stays mutable + duplicable + unblessed (it reaches the
    model only via `llm` / `react_auto`).

Two DISTINCT concepts (kept separate):
  * `_LLM_DEFAULT_POLICIES` (here) — the LLM-loop default NAMES. The bootstrap
    BLESSES these (grants their bodies raw access to `_make_backend`/`descend`/
    `llm_call`) and also seals them. This is the BLESS set.
  * `registry._SEALED_POLICIES` — the RUNTIME SEAL set, populated by
    `registry._seal`. It holds EVERY sealed (immutable + un-duplicable) policy: the
    LLM defaults, a metaparam's sealed extras, and (in future) any policy PLM itself
    seals — sealed but NOT blessed by membership. A SUPERSET of the bless set.

This package exports three names the PREFIX bootstrap uses:
  * `iter_default_policies()` — yields (name, full_source) for each policy file.
  * `_LLM_DEFAULT_POLICIES` — the names blessed (and sealed) after bootstrap.
  * `_bless_llm_callers()` — refreshes `_BLESSED_CALLERS` against the
    `_inner.__code__` of the LLM-default proxies. Called from PREFIX
    (post-install) AND from `kernel.py`'s rehydrate handler (post-restore).
"""

from __future__ import annotations

import pathlib


_DIR = pathlib.Path(__file__).parent


def iter_default_policies():
    """Yield (name, source_text) for each default-policy file in sorted order.

    `_llm_infra.py` is skipped (leading underscore filter). Source is read
    verbatim from disk; the bootstrap loader replays it as a synthetic cell
    via `_install_policy_source` so source-extraction, `<policy-{name}>`
    linecache slots, and guard machinery work uniformly for defaults + extras.
    """
    for p in sorted(_DIR.glob("*.py")):
        if p.stem.startswith("_"):
            continue
        try:
            src = p.read_text()
        except OSError as e:                       # permission flip / removed mid-iteration
            raise RuntimeError(f"could not read default policy file {p.name!r}: {e}") from e
        yield p.stem, src


# The LLM-loop default NAMES — the BLESS set. Bootstrap blesses these (raw
# _make_backend/descend/llm_call access) AND seals them (immutable + un-duplicable).
# This is DISTINCT from the runtime SEAL set (`registry._SEALED_POLICIES`),
# which also contains a metaparam's sealed extras — those are sealed but NOT blessed.
# NOTE: `base_verifier` ships as a default file too but is intentionally ABSENT here —
# that keeps it mutable + duplicable + unblessed (a PLM-forkable reference verifier).
# `react` is the de-bundled capability (R13): present in FULL only; in harnesses
# without it the name simply resolves to None below and is skipped.
_LLM_DEFAULT_POLICIES = frozenset({"llm", "react", "react_auto",
                                   "react_verifier_llm"})


def _bless_llm_callers() -> None:
    """(Re-)seal `_BLESSED_CALLERS` to the code objects of the current
    immutable LLM-policy bodies.

    Called TWICE in the kernel's lifecycle:
      1. From PREFIX bootstrap, AFTER the LLM defaults are installed and
         `_SEALED_POLICIES` is populated.
      2. From `plm/repl/kernel.py`'s rehydrate handler, AFTER `_PLM_POLICIES`
         is reconciled (boot's v0 defaults preserved). Defensively uniform
         — for the LLM defaults whose code we kept-from-PREFIX, it's a no-op;
         for any future immutable default that legitimately rehydrates
         (none in v1), it would bless the rehydrated code.

    Single-shot frozenset assignment — never expose a writable set.
    """
    from plm.policy.registry import _PLM_POLICIES
    from plm.policy.defaults import _llm_infra
    codes = []
    # Bless ONLY the LLM-loop defaults (the BLESS set) — NOT the runtime SEAL set
    # (`_SEALED_POLICIES`). Those were once identical, but a metaparam's SEALED extra is in
    # the SEAL set while deliberately NOT blessed (it reaches the model via llm/react_auto
    # like any policy); blessing the whole sealed set would wrongly grant it raw access.
    for pn in _LLM_DEFAULT_POLICIES:
        p = _PLM_POLICIES.get(pn)
        if p is None:
            continue
        inner = getattr(p, "_inner", None)         # function policy: bless _inner
        if inner is not None:
            if hasattr(inner, "__code__"):         # bless it; if it somehow lacks __code__, SKIP —
                codes.append(inner.__code__)       # do NOT fall through to the class-policy path and
            continue                               # bless an unrelated object's __call__
        # class policy: bless EVERY plain function defined on the class, not just
        # __call__ — a session-style capability (react, R13) makes its LLM calls
        # from methods (step/run/_round), and those bodies are the sealed default
        # code exactly as a function policy's _inner is.
        import inspect as _inspect
        for _name, _fn in vars(p).items():
            _fn = getattr(_fn, "__wrapped__", _fn)     # skip _policy_call wraps
            if _inspect.isfunction(_fn) and hasattr(_fn, "__code__"):
                codes.append(_fn.__code__)
    _llm_infra._BLESSED_CALLERS = frozenset(codes)
