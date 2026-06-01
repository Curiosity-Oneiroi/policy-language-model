"""plm.policy.defaults — the v1 default LLM policies + their developer-controlled
helpers.

Layout (per the plan):
  * `_llm_infra.py` — NOT a policy. Real Python module imported (in-call) by
    the LLM-policy bodies; owns `_make_backend()`, the depth ContextVar +
    `llm_call`/`descend`/`check_depth_or_raise`/`LLMDepthExceeded`, and the
    blessed-caller gate (`_BLESSED_CALLERS` / `_check_blessed_caller`).
  * `natural_llm.py`, `react_llm.py` — the v1 policy source files (read as
    TEXT by `iter_default_policies` and replayed as synthetic cells via the
    bootstrap loader, NOT imported as Python modules — that's how the policy
    source extraction + linecache pipeline works uniformly for defaults + extras).
  * (deferred v1.5) `react_llm_verifier.py`.

This package exports three names the PREFIX bootstrap uses:
  * `iter_default_policies()` — yields (name, full_source) for each policy file.
  * `UNDUPLICABLE_DEFAULTS` — names that are sealed immutable + un-duplicable
    after the bootstrap install.
  * `_bless_llm_callers()` — refreshes `_BLESSED_CALLERS` against the
    `_inner.__code__` of the immutable LLM-default proxies. Called from
    PREFIX (post-install) AND from `kernel.py`'s rehydrate handler
    (post-restore).
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
        yield p.stem, p.read_text()


# The v1 LLM-default names. Bootstrap seals these as immutable + un-duplicable.
# v1.5 will add `"react_llm_verifier"` here (and ship a corresponding .py).
UNDUPLICABLE_DEFAULTS = frozenset({"natural_llm", "react_llm"})


def _bless_llm_callers() -> None:
    """(Re-)seal `_BLESSED_CALLERS` to the code objects of the current
    immutable LLM-policy bodies.

    Called TWICE in the kernel's lifecycle:
      1. From PREFIX bootstrap, AFTER the LLM defaults are installed and
         `_IMMUTABLE_POLICIES`/`_UNDUPLICABLE_POLICIES` are populated.
      2. From `plm/repl/kernel.py`'s rehydrate handler, AFTER `_PLM_POLICIES`
         is reconciled (boot's v0 defaults preserved). Defensively uniform
         — for the LLM defaults whose code we kept-from-PREFIX, it's a no-op;
         for any future immutable default that legitimately rehydrates
         (none in v1), it would bless the rehydrated code.

    Single-shot frozenset assignment — never expose a writable set.
    """
    from plm.policy.registry import _PLM_POLICIES, _UNDUPLICABLE_POLICIES
    from plm.policy.defaults import _llm_infra
    codes = []
    for pn in _UNDUPLICABLE_POLICIES:
        p = _PLM_POLICIES.get(pn)
        if p is None:
            continue
        inner = getattr(p, "_inner", None)         # function policy: bless _inner
        if inner is not None and hasattr(inner, "__code__"):
            codes.append(inner.__code__)
            continue
        cm = getattr(p, "__call__", None)          # class policy: bless original
        orig = getattr(cm, "__wrapped__", cm)      # __wrapped__ skips _policy_call wrap
        if hasattr(orig, "__code__"):
            codes.append(orig.__code__)
    _llm_infra._BLESSED_CALLERS = frozenset(codes)
