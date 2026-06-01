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

Two sets gate the LLM-default extensions added by the Phase 2 plan:
  * `_IMMUTABLE_POLICIES`  — names whose proxy/class refuses _rewrite/_edit/
    _insert/_delete_lines/_remove and whose `__main__` binding Guard A
    rejects and Guard C restores.
  * `_UNDUPLICABLE_POLICIES` — names that `duplicate_policy` refuses to fork.
The PREFIX bootstrap fills both with the LLM-default names at boot. Kept
distinct (even though they coincide in v1) so future immutable-but-duplicable
defaults can land without re-plumbing.
"""

from __future__ import annotations

import linecache as _linecache
import sys


_PLM_POLICIES: dict = {}            # name -> _FunctionPolicy proxy | real class
_MISSING = object()                 # sentinel: shared with guard (it imports this one)

# Module-level (NOT __main__) so ordinary cell code can't hold/mutate these by
# name. Bootstrap populates them after the LLM defaults install; they are
# rebuilt from scratch on each boot, so no snapshot is needed.
_IMMUTABLE_POLICIES: set = set()
_UNDUPLICABLE_POLICIES: set = set()


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
    """
    g = _main()
    for n in list(_PLM_POLICIES):
        if n in _IMMUTABLE_POLICIES:
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
    `slot` linecache entry stays cached (small, name-stable, used only for
    install-time tracebacks).

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


# --- duplicate_policy: fork any DUPLICABLE policy into a fresh mutable copy --

def duplicate_policy(name: str, new_name: str):
    """Fork a duplicable policy under a new name. Returns the new policy (mutable).

    Refuses (with a `_policy_note`, no raise) when:
      * `name` is not registered.
      * `name` is in `_UNDUPLICABLE_POLICIES` (the LLM defaults — model-access base).
      * `new_name` is not a valid identifier, OR already in `_PLM_POLICIES` /
        `__main__`.

    The copy is created via the normal `@policy` install path under a fresh
    `<policy-dup-{new_name}>` linecache slot. Internal references inside the
    body to the OLD name are NOT rewritten — they continue to resolve to the
    original at call time (predictable + simple; if the copy should
    self-reference, `_edit(...)` it after). The fresh proxy is mutable (not
    in `_IMMUTABLE_POLICIES`).
    """
    from .edits import _compute_rename
    from .proxy import _policy_note

    _sync()
    if name not in _PLM_POLICIES:
        raise KeyError(name)
    if name in _UNDUPLICABLE_POLICIES:
        _policy_note(
            f"{name!r} is un-duplicable (model-access base); cannot duplicate"
        )
        return None
    if not new_name.isidentifier() or new_name in _PLM_POLICIES or new_name in _main():
        _policy_note(
            f"duplicate: {new_name!r} is taken or not a valid name"
        )
        return None
    new_src = _compute_rename(_PLM_POLICIES[name]._p_source, new_name)
    _install_policy_source("@policy\n" + new_src, "<policy-dup-" + new_name + ">")
    return _PLM_POLICIES.get(new_name)                 # fresh → MUTABLE
