"""The policy registry and the by-name helpers PLM calls from cells.

`_PLM_POLICIES` is the single canonical name->policy map. It is mutated IN
PLACE across the whole policy layer (every module imports THIS dict), and on
crash-restart rehydrate it's `.clear()`/`.update()`d in place too — so the
binding here, in `proxy`/`decorator`/`guard`, and in the kernel's `__main__`
all stay one object.

This module imports ONLY stdlib and never imports proxy/decorator — the by-name
helpers duck-type on the policy objects (`._rewrite`, `._edit`, `._p_source`,
...), which works identically for the `_FunctionPolicy` proxy and the real
class policy.

One set gates the seal extensions:
  * `_SEALED_POLICIES` — the SEALED policy NAMES: their proxy/class refuses
    _rewrite/_edit/_insert/_delete_lines/_remove, `duplicate_policy` refuses to
    fork them, Guard A rejects rebinding their `__main__` name, and Guard C
    restores it. "Sealed" = immutable AND un-duplicable (one concept).
The PREFIX bootstrap fills it via `_seal`: the LLM-loop defaults, plus a
metaparam's sealed extras, plus (in future) any policy PLM itself seals. This is
DISTINCT from `defaults._LLM_DEFAULT_POLICIES` (the BLESS set — those defaults
also get raw-primitive access); `_SEALED_POLICIES` is a SUPERSET that is only
sealed, never blessed by virtue of membership.
"""

from __future__ import annotations

import linecache as _linecache
import sys
from contextlib import contextmanager as _contextmanager


_MISSING = object()                 # sentinel: shared with guard (it imports this one)

# Module-level (NOT __main__) so ordinary cell code can't hold/mutate these by
# name. Bootstrap populates them via `_seal` after the LLM defaults install;
# they are rebuilt from scratch on each boot, so no snapshot is needed. They
# carry the SEALED (immutable + un-duplicable) NAMES used by the name-based checks
# (Guard A, duplicate_policy); the per-OBJECT truth is the policy's `_p_immutable` flag.
_SEALED_POLICIES: set = set()


# --- the single policy store + its default-policy protection ------------------
#
# `_PLM_POLICIES` is the ONE registry (name -> _FunctionPolicy proxy | class
# policy). Immutability is a property of the policy OBJECT — its `_p_immutable`
# flag, i.e. a "default policy" vs a "normal policy" — NOT a parallel store.
# While SEALED (the normal state during cell execution) the store refuses to
# REPLACE or REMOVE an entry whose current value is a default policy, closing the
# ordinary tamper paths (`store[k]=x`, `del store[k]`, pop/popitem/update/clear/
# |=). The harness lifts the seal via `_unsealed()` for the full-reset paths it
# owns (crash-restart rehydrate, test teardown). Installing a default works
# because the key is absent at install time; `_seal` flips the flag afterwards.
#
# Irreducible boundary (documented; same class as `_BLESSED_CALLERS`): a
# deliberate `dict.__setitem__(store, ...)` or flipping `_STORE_UNSEALED` from an
# imported module reference bypasses this — and the blessed-caller gate prevents
# privilege escalation regardless (a hijacked default still cannot reach
# `_make_backend`). This is integrity hardening of the edit API, not a sandbox.

_STORE_UNSEALED: bool = False


@_contextmanager
def _unsealed():
    """Temporarily lift default-policy protection on `_PLM_POLICIES`. HARNESS ONLY
    (crash-restart rehydrate / test reset). Re-entrant-safe."""
    global _STORE_UNSEALED
    _prev = _STORE_UNSEALED
    _STORE_UNSEALED = True
    try:
        yield
    finally:
        _STORE_UNSEALED = _prev


def _is_default(value) -> bool:
    """True iff `value` is a DEFAULT (immutable) policy — intrinsic to the object."""
    return getattr(value, "_p_immutable", False) is True


class _PolicyStore(dict):
    """The single policy registry; refuses to replace/remove a default-policy
    entry while sealed (see module notes above)."""

    def _blocked(self, key) -> bool:
        if _STORE_UNSEALED:
            return False
        if _is_default(dict.get(self, key)):
            from .proxy import _policy_note           # local import: break import cycle
            _policy_note(
                f"{key!r} is an immutable default policy; the registry refuses to "
                f"replace/remove it. Use `duplicate_policy({key!r}, '<new_name>')` to fork."
            )
            return True
        return False

    def __setitem__(self, key, value):
        if dict.get(self, key) is value:              # idempotent re-set -> no-op
            return
        if self._blocked(key):
            return
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        if self._blocked(key):
            return
        dict.__delitem__(self, key)

    def pop(self, key, *default):
        if self._blocked(key):
            # An immutable default is PROTECTED — it can't be popped/removed. Do NOT
            # silently return its live value (the caller would believe they removed it):
            # honor the pop(k[, default]) contract — return `default` if given, else
            # raise KeyError to signal the key could not be popped.
            if default:
                return default[0]
            raise KeyError(key)
        return dict.pop(self, key, *default)

    def popitem(self):
        if _STORE_UNSEALED:
            return dict.popitem(self)
        for _k in reversed(list(self.keys())):        # remove the last NON-default entry
            if not _is_default(dict.get(self, _k)):
                _v = dict.__getitem__(self, _k)
                dict.__delitem__(self, _k)
                return (_k, _v)
        raise KeyError("popitem: only immutable default policies remain")

    def setdefault(self, key, default=None):
        return dict.setdefault(self, key, default)    # never REPLACES an existing key

    def update(self, other=(), **kwargs):
        if hasattr(other, "keys"):
            for _k in other.keys():
                self[_k] = other[_k]
        else:
            for _k, _v in other:
                self[_k] = _v
        for _k, _v in kwargs.items():
            self[_k] = _v

    def __ior__(self, other):
        self.update(other)
        return self

    def clear(self):
        if _STORE_UNSEALED:
            dict.clear(self)
            return
        for _k in [k for k in list(self.keys()) if not _is_default(dict.get(self, k))]:
            dict.__delitem__(self, _k)


_PLM_POLICIES = _PolicyStore()      # the single name -> policy registry


def _seal(name: str) -> None:
    """Mark a registered policy as a DEFAULT (immutable) + un-duplicable: set the
    intrinsic `_p_immutable` flag on the object (the default-vs-normal
    distinction) and record the name in the immutable/un-duplicable name-sets
    (used by Guard A and the by-name checks). Idempotent."""
    p = dict.get(_PLM_POLICIES, name)
    if p is not None:
        try:
            p._p_immutable = True                     # idempotent (proxy allows re-set to True)
        except Exception:
            pass
    _SEALED_POLICIES.add(name)


def _main() -> dict:
    """The one true namespace — all cell code/policies live in kernel __main__."""
    return sys.modules["__main__"].__dict__


def _sync() -> None:
    """Drop only DELETED policies from the registry.

    A name ABSENT from __main__ was `del`'d -> remove it. A REBIND (name present
    but bound to something != the canonical) is LEFT for Guard C to RESTORE
    post-cell — `_sync` must NOT evict rebinds, or Guard C would have nothing to
    restore and the policy would be lost (e.g. a helper called right after
    `globals()["p"] = x`).

    Immutability extension: NEVER evict an immutable policy from the registry,
    even if its __main__ binding is missing — Guard C must restore it on the
    next post-cell pass, and a mid-cell `del <immutable>` + helper-call cannot
    drop it before then.

    NOTE : `@policy` is the ONLY supported install surface — it writes BOTH
    the registry AND __main__. A DIRECT registry-only write (`_PLM_POLICIES[n] = x`
    without binding `n` in __main__) is therefore indistinguishable from a deleted
    policy here and is SILENTLY EVICTED on the next `_sync`. Future code touching
    the registry must go through `@policy` / `_install_policy_source`.
    """
    g = _main()
    for n in list(_PLM_POLICIES):
        if n in _SEALED_POLICIES:
            continue                       # immutable: registry-canonical, never evict
        if n not in g:
            _PLM_POLICIES.pop(n, None)


# --- by-name helpers (called from cells) -------------------------------------

def list_policies():
    _sync()
    return sorted(_PLM_POLICIES)


def get_policy(name):
    _sync()
    return _PLM_POLICIES[name]                     # KeyError if absent


def read_policy(name):
    _sync()
    return _PLM_POLICIES[name]._p_source


def rewrite_policy(name, src):
    _sync()
    _PLM_POLICIES[name]._rewrite(src)


def edit_policy(name, old, new, **kw):
    _sync()
    _PLM_POLICIES[name]._edit(old, new, **kw)


def insert_into(name, after, content):
    _sync()
    _PLM_POLICIES[name]._insert(after, content)


def delete_lines(name, a, b=None):
    _sync()
    _PLM_POLICIES[name]._delete_lines(a, b)


def delete_policy(name):
    _sync()
    _PLM_POLICIES[name]._remove()


# --- shared replay helper (bootstrap loader + duplicate_policy) --------------

def _install_policy_source(full_src: str, slot: str) -> None:
    """Install a policy from a full `@policy`-decorated source string by
    populating `linecache` under `slot` and exec'ing under __main__.

    The exec runs `@policy def/class <name>(...)`, which (a) creates the
    policy object via the normal decorator path, (b) re-execs the stripped
    source under the canonical `<policy-{name}>` slot via `policy()`, and
    (c) registers the result in `_PLM_POLICIES` + `__main__`. The transient
    `slot` linecache entry is only needed for install-time tracebacks, so it
    is dropped once the exec SUCCEEDS (the policy now lives in `<policy-{name}>`)
    — otherwise each distinct dup/bootstrap name would leak a slot. On
    failure it is left in place so the surfacing traceback can read the source.

    Shared by:
      * PREFIX bootstrap loop (installs default LLM policies + extras under
        `<policy-bootstrap-{name}>`).
      * `duplicate_policy` (installs the renamed copy under
        `<policy-dup-{new_name}>`).
    Both go through the same code path, so source-extraction, linecache,
    guards, and snapshot identity work uniformly.
    """
    _linecache.cache[slot] = (len(full_src), None, full_src.splitlines(True), slot)
    g = _main()
    exec(compile(full_src, slot, "exec"), g, g)        # @policy resolves via __main__ globals
    _linecache.cache.pop(slot, None)                   # success: drop the transient install slot


# --- duplicate_policy: fork any DUPLICABLE policy into a fresh mutable copy --

def duplicate_policy(name: str, new_name: str):
    """Fork a duplicable policy under a new name. Returns the new policy (mutable).

    Refuses (with a `_policy_note`, no raise) when:
      * `name` is not registered.
      * `name` is in `_SEALED_POLICIES` (the LLM defaults — model-access base).
      * `new_name` fails the shared `@policy` name rule (`_reserved_name_reason`):
        not a valid identifier, a reserved kernel name, or a kernel-internal
        prefix (`__` / `_repl` / `_REPL`) — the same names `@policy` rejects.
      * `new_name` is already in `_PLM_POLICIES` / `__main__`.

    The copy is created via the normal `@policy` install path under a fresh
    `<policy-dup-{new_name}>` linecache slot. Internal references inside the
    body to the OLD name are NOT rewritten — they continue to resolve to the
    original at call time (predictable + simple; if the copy should
    self-reference, `_edit(...)` it after). The fresh proxy is mutable (not
    in `_SEALED_POLICIES`).
    """
    from .edits import _compute_rename
    from .proxy import _policy_note
    from .decorator import _reserved_name_reason

    _sync()
    if name not in _PLM_POLICIES:
        # Docstring promises ALL refusals are a gentle note + return None (no
        # raise) — the _duplicate/_class_duplicate proxy wrappers rely on it.
        _policy_note(f"duplicate: {name!r} is not a registered policy")
        return None
    if name in _SEALED_POLICIES:
        _policy_note(
            f"{name!r} is sealed (immutable + un-duplicable); cannot duplicate"
        )
        return None
    # Pre-check the SAME name rule `@policy` enforces, so a name it would later
    # reject (reserved / __ / _repl / _REPL prefix / kernel-internal) is refused
    # here with a gentle note instead of blowing up with a raw ValueError deep in
    # the install path below.
    reason = _reserved_name_reason(new_name)
    if reason is not None:
        _policy_note(f"duplicate: {reason}")
        return None
    if new_name in _PLM_POLICIES or new_name in _main():
        _policy_note(
            f"duplicate: {new_name!r} is already taken"
        )
        return None
    new_src = _compute_rename(_PLM_POLICIES[name]._p_source, new_name)
    _install_policy_source("@policy\n" + new_src, "<policy-dup-" + new_name + ">")
    return _PLM_POLICIES.get(new_name)                 # fresh → MUTABLE
