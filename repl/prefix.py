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
import linecache as _linecache
import warnings as _repl_warnings
# Silence dill's PicklingWarning kernel-wide: the per-cell snapshot routinely
# probes unpicklable values (every class policy is unpicklable by value; we handle
# that by skipping / source-fallback), and the warning would otherwise spam the
# captured cell stderr that PLM surfaces to the model. Carries no signal here.
try:
    _repl_warnings.filterwarnings("ignore", category=_dill.PicklingWarning)
except Exception:
    _repl_warnings.filterwarnings("ignore", module=r"dill.*")
try:
    import plm as _plm_check                # fail LOUDLY + clearly if plm isn't importable
except Exception as _plm_import_err:
    raise RuntimeError(
        "PLM kernel cannot import `plm`. The per-session venv does NOT install it — "
        "`plm` must be importable as live source via PYTHONPATH (set by the parent "
        "from the package's own location). PYTHONPATH="
        + repr(_os.environ.get("PYTHONPATH", ""))
    ) from _plm_import_err
from plm.policy import (
    policy, list_policies, get_policy, read_policy, edit_policy,
    rewrite_policy, insert_into, delete_lines, delete_policy,
    duplicate_policy,
    _PLM_POLICIES, _audit_cell, _post_cell_guard,
)
from plm.repl.parallel import parallel
from plm.repl.exec_ns import exec_ns       # shared code-runner (verifiers / code-as-data)
# Pydantic-backed authoring surface (Constraint & friends), BEST-EFFORT + GUARDED:
# standard kernels carry pydantic (DEFAULT_PREINSTALL), so expose the WHOLE constraint
# surface AMBIENTLY for cells + editable policies — no import needed, like the names
# above. A minimal / dill-only kernel has no pydantic → skip and STILL BOOT (the hard
# boot path stays pydantic-free). Done BEFORE the `_REPL_INJECTED` freeze so, when
# present, the names join the frozen set (snapshot-skipped + shadow-protected). Sealed
# machinery (react_llm) still lazy-imports for self-containment.
try:
    from plm.constraint import (
        Constraint, ConstraintViolation, Field, constraint, get_description, REGISTRY,
    )
except ImportError:
    pass

# Raise Python's recursion limit so the UNIFORM @policy depth cap
# (POLICY_CALL_DEPTH_CAP in plm.policy.proxy) — not Python's uneven native-frame
# limit — is what bounds @policy recursion. A function policy spends ~2 native
# frames per call, so at the default 1000 the native limit pre-empts the cap at
# ~half its value. 20000 leaves generous headroom (the cap fires far below it),
# and is safe on py>=3.12 where Python->Python frames are heap-allocated; the
# hard C-recursion guard still protects the C stack independently.
_sys.setrecursionlimit(20000)

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
    # `_PLM_POLICIES` stays BLOCKED from this generic per-global loop (it's in
    # _REPL_INJECTED) and is snapshotted PER-ENTRY below instead — a single
    # unpicklable policy must not sink the whole registry as one monolithic value.
    # (dill's PicklingWarning, emitted routinely now that every class policy is
    # unpicklable by value, is silenced once at boot — see the PREFIX top — so it
    # never leaks into the captured cell stderr surfaced to the model.)
    _blocked = _REPL_INJECTED or _builtins.set()
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
    # Per-entry registry snapshot. Before, `_PLM_POLICIES` was dumped as ONE value,
    # so a SINGLE unpicklable policy sank the ENTIRE registry blob — and
    # crash-restart then lost ALL authored policies AND resurrected deleted defaults
    # (#H11/#H12). Now each policy is handled on its own:
    #   * picklable  -> kept BY VALUE (dill memoization within the single final blob
    #                   keeps a cell var aliasing it restored to the SAME object, so
    #                   liveness/identity across respawn is preserved);
    #   * unpicklable -> fall back to its `_p_source` string so rehydrate can re-exec
    #                   `@policy` on it. A CLASS policy is the live example: its
    #                   depth-cap `__call__` wrap closes over the _POLICY_CALL_DEPTH
    #                   ContextVar, and dill pickles such a __main__ class by-reference
    #                   (unresolvable on the fresh respawn), so by-value can't work and
    #                   source-rebuild is the ONLY way it survives.
    _reg = {}
    _pol_sources = {}
    for _pn, _pol in _builtins.list(_PLM_POLICIES.items()):
        try:
            _dill.dumps(_pol)
            _reg[_pn] = _pol
        except Exception:
            _src = _builtins.getattr(_pol, "_p_source", None)
            if _builtins.isinstance(_src, str):
                _pol_sources[_pn] = _src
    _snap["_PLM_POLICIES"] = _reg
    if _pol_sources:
        _snap["_PLM_POLICY_SOURCES"] = _pol_sources
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


class _REPLReturn(BaseException):
    def __init__(self, value):
        super().__init__("REPL_RETURN")
        self.value = value


def RETURN(obj):
    raise _REPLReturn(obj)


# The root agent's own React trajectory (full messages — system, user,
# assistant-with-tool_calls, and tool results), streamed in by the parent before
# every root cell. `_repl_plm_messages` is the hidden accumulator the parent
# extends additively (delta per round); `plm_messages` is the PUBLIC view the
# kernel rebinds to a fresh copy of it each round (so a cell can read its own
# trajectory but cannot corrupt the accumulation). `plm_messages` is defined
# PRE-FREEZE so it is a protected kernel name (snapshot-skipped, not @policy-
# shadowable); `_repl_plm_messages` is `_repl_`-prefixed so it is likewise
# snapshot-skipped and rebuilt from the parent's resend after any respawn.
_repl_plm_messages = []
plm_messages = []


_REPL_INJECTED = {
    _k for _k in globals().keys()
    if not _k.startswith("__")
       and not _k.startswith("_repl")
       and not _k.startswith("_REPL")
}

# Canonical values of the injected CALLABLE helpers (policy ops, parallel, exec_ns,
# the constraint surface, ...), captured ONCE post-freeze. The KERNEL_LOOP's post-cell
# guard restores any a cell rebinds or deletes — so a granted helper can't be clobbered
# for the next cell, the same protection immutable policies get (via _post_cell_guard).
# CALLABLES ONLY on purpose: it must NOT touch reseeded state like `plm_messages` (a
# per-round list) or in-place registries (`_PLM_POLICIES`, `REGISTRY`) — restoring those
# to a boot value would wipe live state. `_REPL`-prefixed -> not snapshotted, re-derived
# each boot, so a fresh canonical map is rebuilt on crash-restart too.
_REPL_INJECTED_CANON = {
    _k: globals()[_k] for _k in _REPL_INJECTED if callable(globals().get(_k))
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
    _seal as _repl_seal,
)

try:
    _repl_extra_pols = _repl_json.loads(_os.environ.get("_PLM_EXTRA_POLICIES") or "{}")
except Exception:
    _repl_extra_pols = {}
if not isinstance(_repl_extra_pols, dict):
    # Valid JSON but not an object (e.g. "[]", "null", "123", '"x"'): `.items()`
    # below would raise OUTSIDE the per-item try and hard-brick boot, violating
    # the "extras stay soft" contract (#26). Drop it softly + surface a note via
    # boot_stderr. (The parent also fail-fast-validates extra_policies; this is
    # defense in depth for any path that sets the env var directly.)
    _repl_stderr_buf.write(
        "[boot] _PLM_EXTRA_POLICIES was valid JSON but not an object; ignoring extras.\n"
    )
    _repl_extra_pols = {}

# Defaults come from real .py files (read as text by iter_default_policies);
# extras are passed as a {name: source} dict in the env. Both run through the
# same @policy install path, so both become normal policies in _PLM_POLICIES.
# A DEFAULT failing to install means a BROKEN KERNEL (a core policy is unusable),
# so we DON'T swallow it: the exception propagates out of boot, the wrapper emits
# a `boot_error` frame, and the parent raises a clear PLM-process error (#26) —
# instead of limping with a missing default whose traceback is lost to the reset
# boot buffer. EXTRAS (user/injected {name: source}) must NOT brick boot, so they
# stay soft; their tracebacks are captured in `_repl_stderr_buf` and surfaced via
# the `ready` frame's `boot_stderr` (see KERNEL_LOOP) so they aren't lost either.
for _repl_pn, _repl_ps in _repl_iter_defaults():   # source already includes the "@policy" line
    _repl_install(_repl_ps, "<policy-bootstrap-" + _repl_pn + ">")
for _repl_pn, _repl_ps in _repl_extra_pols.items():
    try:
        _repl_install(_repl_ps, "<policy-bootstrap-" + _repl_pn + ">")
    except Exception:
        _traceback.print_exc()                  # an injected/extra policy must not brick boot

# Seal ONLY the names in UNDUPLICABLE_DEFAULTS (the immutable LLM-loop defaults
# — natural_llm / react_llm / react_llm_verifier) as DEFAULT policies: `_seal`
# sets the intrinsic `_p_immutable` flag on each object AND records the name in
# the immutable/un-duplicable name-sets. Other defaults (e.g. the mutable
# base_verifier) and extras stay mutable + duplicable.
# An extra that loaded with a default's name (re-decoration BEFORE this seal)
# replaces the default in _PLM_POLICIES; then this loop seals the REPLACEMENT
# (the seal follows the name, not the original).
for _repl_pn in _repl_undup:
    _repl_seal(_repl_pn)

# Bless ONLY the immutable un-duplicable LLM policies' bodies as legitimate
# callers of _llm_infra's public functions. Anything else calling
# _make_backend / descend / llm_call (e.g. a PLM-authored mutable policy that
# imports the helper) will raise. The shared helper is called AGAIN from
# kernel.py's rehydrate handler — same code path, refreshes against the
# post-rehydrate code objects.
_repl_bless()
'''
