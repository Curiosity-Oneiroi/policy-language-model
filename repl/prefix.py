"""Single PLM REPL prefix.

Sets up stdout/stderr capture, the single-blob per-cell var snapshot (for
crash-restart rehydrate only) + linecache rebuild, RETURN, and
loads the `@policy` layer (`from plm.policy import ...`) so the decorator and
by-name helpers are available in every cell.

Everything except PLM is a policy that runs in this same REPL — there are no
separate sub-LLM or verifier REPLs, no injected-objects sidecar mechanism, and
no OUTPUT_SCHEMA (the Constraint class handles validation at the PLM layer).
`llm()` / `llm_batch()` are delivered as the `natural_llm` / `react_llm`
policies loaded by the bootstrap loop below; depth gating lives entirely in
`plm/policy/defaults/_llm_infra`.
"""

from __future__ import annotations


PREFIX = r'''
import sys as _sys, os as _os, io as _io
import traceback as _traceback
import dill as _dill
import keyword as _keyword
import linecache as _linecache
from plm.policy import (
    policy, list_policies, get_policy, read_policy, edit_policy,
    rewrite_policy, insert_into, delete_lines, delete_policy,
    duplicate_policy,
    _PLM_POLICIES, _audit_cell, _post_cell_guard,
)
from plm.repl.parallel import parallel

_repl_stdout_buf = _io.StringIO()
_repl_stderr_buf = _io.StringIO()
_sys.stdout = _repl_stdout_buf
_sys.stderr = _repl_stderr_buf


def _repl_excepthook(_exc_type, _exc_val, _exc_tb):
    _repl_stderr_buf.write(
        "".join(_traceback.format_exception(_exc_type, _exc_val, _exc_tb))
    )


_sys.excepthook = _repl_excepthook


_REPL_VAR_SIZE_MAX = int(_os.environ.get("_REPL_VAR_SIZE_MAX", str(10 * 1024 * 1024)))

_REPL_INJECTED = None


def _repl_collect_vars():
    # Snapshot the kernel's user globals for crash-restart rehydrate as ONE
    # dill blob: within a single pickle, dill memoizes objects, so every
    # reference to a policy (the cell global, _PLM_POLICIES entry, any container)
    # serializes once and all restore to the SAME object — identity is airtight
    # with no __reduce__/contextvar machinery. The per-value probe drops
    # unpicklable/oversized values; the guarded final dump means even a value
    # that pickles alone but breaks the combined blob degrades to "no snapshot
    # this round" instead of crashing the kernel. Builtins via _builtins so a
    # cell shadowing set/list/globals/len in __main__ can't break collection.
    # _PLM_POLICIES is subtracted back in so the registry IS snapshotted.
    _blocked = (_REPL_INJECTED or _builtins.set()) - {"_PLM_POLICIES"}
    _snap = {}
    for _k, _v in _builtins.list(_builtins.globals().items()):
        if _k in _blocked or _k.startswith(("__", "_repl", "_REPL")):
            continue
        try:
            _probe = _dill.dumps(_v)
        except Exception:
            continue
        if _builtins.len(_probe) > _REPL_VAR_SIZE_MAX:
            continue
        _snap[_k] = _v
    try:
        return _dill.dumps(_snap)
    except Exception:
        return b""


def _rebuild_linecache_from_policies():
    # After respawn, policy code lives under <policy-{name}> (NOT <cell-N>, which
    # is gone) — rebuild those linecache slots from each policy's _p_source so
    # tracebacks-from-policies and inspect.getsource keep working.
    for _p in _builtins.list(_PLM_POLICIES.values()):
        _src, _fn = _p._p_source, _p._p_filename
        _linecache.cache[_fn] = (_builtins.len(_src), None, _src.splitlines(True), _fn)


def _repl_resolve_names(_names, _frame):
    _named = {}
    _seen = set()
    for _name in (_names or []):
        if not isinstance(_name, str):
            raise TypeError(
                "objects items must be variable-name strings, "
                "got " + type(_name).__name__ + ". Use objects=['ctx'] not objects=[ctx]."
            )
        if not _name.isidentifier() or _keyword.iskeyword(_name):
            raise ValueError(
                "'" + _name + "' is not a valid variable name"
            )
        if _name in _seen:
            continue
        _seen.add(_name)

        _f = _frame
        _found = False
        while _f is not None:
            if _name in _f.f_locals:
                _named[_name] = _f.f_locals[_name]
                _found = True
                break
            _f = _f.f_back
        if not _found:
            raise NameError(
                "variable '" + _name + "' not found in any scope"
            )
    return _named


class _REPLReturn(BaseException):
    def __init__(self, value):
        super().__init__("REPL_RETURN")
        self.value = value


def RETURN(obj):
    raise _REPLReturn(obj)


_REPL_INJECTED = {
    _k for _k in globals().keys()
    if not _k.startswith("__")
       and not _k.startswith("_repl")
       and not _k.startswith("_REPL")
}

# ============================================================================
# Phase 2 bootstrap: install the immutable LLM defaults + any extras, seal
# their names as immutable + un-duplicable, then bless their `_inner.__code__`
# for the blessed-caller gate in `_llm_infra`.
#
# Appended AFTER the `_REPL_INJECTED = {...}` freeze (so the policy NAMES like
# `natural_llm` / `react_llm` are NOT in the freeze -> they're snapshotted and
# accepted by `@policy`; immutability is enforced separately via
# `_IMMUTABLE_POLICIES`, not the freeze). All temps `_repl_`-prefixed -> never
# snapshotted, never visible to dir() filtering.
# ============================================================================

import json as _repl_json
from plm.policy.defaults import (
    iter_default_policies as _repl_iter_defaults,
    UNDUPLICABLE_DEFAULTS as _repl_undup,
    _bless_llm_callers as _repl_bless,
)
from plm.policy.registry import (
    _install_policy_source as _repl_install,
    _IMMUTABLE_POLICIES as _repl_immset,
    _UNDUPLICABLE_POLICIES as _repl_undup_set,
)

try:
    _repl_extra_pols = _repl_json.loads(_os.environ.get("_PLM_EXTRA_POLICIES") or "{}")
except Exception:
    _repl_extra_pols = {}

# Defaults come from real .py files (read as text by iter_default_policies);
# extras are passed as a {name: source} dict in the env. Both run through the
# same @policy install path, so both become normal policies in _PLM_POLICIES.
_repl_boot = list(_repl_iter_defaults()) + list(_repl_extra_pols.items())
for _repl_pn, _repl_ps in _repl_boot:           # source text already includes the "@policy" line
    try:
        _repl_install(_repl_ps, "<policy-bootstrap-" + _repl_pn + ">")
    except Exception:
        _traceback.print_exc()                  # a broken (e.g. injected) policy can't brick boot

# Mark ONLY the LLM defaults (UNDUPLICABLE_DEFAULTS = {"natural_llm",
# "react_llm"}) as immutable + un-duplicable. Extras stay mutable + duplicable.
# An extra that loaded with a default's name (re-decoration BEFORE this mark)
# replaces the default in _PLM_POLICIES; then this loop seals the REPLACEMENT
# as immutable + un-duplicable (the seal follows the name, not the original).
for _repl_pn in _repl_undup:
    _repl_immset.add(_repl_pn)
    _repl_undup_set.add(_repl_pn)

# Bless ONLY the immutable un-duplicable LLM policies' bodies as legitimate
# callers of _llm_infra's public functions. Anything else calling
# _make_backend / descend / llm_call (e.g. a PLM-authored mutable policy that
# imports the helper) will raise. The shared helper is called AGAIN from
# kernel.py's rehydrate handler — same code path, refreshes against the
# post-rehydrate code objects.
_repl_bless()
'''
