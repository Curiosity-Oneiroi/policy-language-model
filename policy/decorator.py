"""The `@policy` decorator + its source-extraction and closure-warning helpers.

`@policy` turns a function or class into a live, editable policy registered in
`_PLM_POLICIES` and injected into the kernel's `__main__`. It works from ANY
scope (policies create policies). Validation is a HARD ERROR (raise), never a
silent skip, and runs BEFORE any injection — so a rejected name never overwrites
a kernel internal and no half-made policy is created.
"""

from __future__ import annotations

import ast
import inspect
import linecache
import sys
import textwrap

from .edits import _slice_policy_source, _strip_policy_decorator, _iter_funcs
from .proxy import (
    _FunctionPolicy,
    _attach_class_policy_metadata,
    _class_origin_filename,
    _policy_note,
    _rewrite_class_policy,
)
from .registry import _PLM_POLICIES


_RESERVED = frozenset({
    "policy", "list_policies", "get_policy", "read_policy", "edit_policy",
    "rewrite_policy", "insert_into", "delete_lines", "delete_policy",
    "duplicate_policy",
    "RETURN", "_PLM_POLICIES",
})


def _reserved_name_reason(name: str) -> str | None:
    """The SINGLE source of truth for whether `name` may be a policy name.

    Returns a human-readable reason it is UNusable, or None if it is fine.
    Shared by `@policy` (which raises ValueError) and `duplicate_policy` (which
    emits a gentle `_policy_note`), so both refuse the SAME names for the SAME
    stated reason — instead of `duplicate_policy` accepting a name `@policy`
    will later reject with a raw traceback from deep in the install path.
    """
    if not isinstance(name, str):                         # e.g. duplicate_policy(p, 123)
        return f"{name!r} is not a string name"           # guard BEFORE .isidentifier()
    if not name.isidentifier():                           # e.g. lambdas -> '<lambda>'
        return f"{name!r} is not a valid identifier (needs a named def/class)"
    if name in _RESERVED:
        return f"{name!r} is a reserved kernel name"
    # Kernel-internal prefixes are filtered by the snapshot collector, so such a
    # policy's global binding wouldn't be saved and it would vanish on respawn.
    if name.startswith(("__", "_repl", "_REPL")):
        return (f"{name!r} uses a kernel-internal prefix (__ / _repl / _REPL); "
                f"it wouldn't survive crash-restart")
    # A name shadowing a PREFIX-injected name (a module alias like `_dill`, or a
    # helper like `policy`/`list_policies`) would break the kernel, which runs in
    # these same globals. `_REPL_INJECTED` is the frozen set of those names;
    # `or ()` tolerates its pre-freeze None / absence in in-process unit tests.
    injected = sys.modules["__main__"].__dict__.get("_REPL_INJECTED") or ()
    if name in injected:
        return f"{name!r} shadows a kernel-internal name"
    return None


def policy(obj):
    if not (inspect.isfunction(obj) or inspect.isclass(obj)):
        raise TypeError("@policy supports functions and classes only")
    name = obj.__name__
    reason = _reserved_name_reason(name)                  # shared rule (see duplicate_policy)
    if reason is not None:
        raise ValueError(f"@policy: {reason}")
    main_globals = sys.modules["__main__"].__dict__       # ALWAYS the kernel main

    # Re-decoration of an IMMUTABLE default is refused OUTRIGHT — the new body is
    # discarded and the canonical policy kept. Do this FIRST, before
    # _warn_if_captures_locals and before reading source: the rejected body is
    # never re-exec'd, so emitting a closure "will NameError at call" warning about
    # it (immediately followed by the "ignored" note) is misleading stderr noise;
    # and reading its source is pure waste (could even raise spuriously on an
    # unreadable body). Covers BOTH same-kind (would funnel to _rewrite's gate
    # anyway) AND the kind-change path below that would otherwise pop+recreate and
    # clobber the default. One note, no half-work. (#R5-7)
    _existing0 = _PLM_POLICIES.get(name)
    if _existing0 is not None and getattr(_existing0, "_p_immutable", False):
        _policy_note(
            f"@policy {name!r}: immutable default policy; re-decoration ignored. "
            f"Use `duplicate_policy({name!r}, '<new_name>')` to fork it."
        )
        main_globals[name] = _existing0                   # keep the canonical binding
        return _existing0

    _warn_if_captures_locals(obj)                         # closures don't survive re-exec
    # Functions use inspect.getsource (cell_file unused); classes need their
    # origin file to read the cell from linecache (no co_firstlineno to use).
    cell_file = None if inspect.isfunction(obj) else _class_origin_filename(obj)
    source = _get_source_for_obj(obj, cell_file)          # verbatim, @policy stripped
    stable = f"<policy-{name}>"
    # NB: do NOT write linecache[stable] here. The same-kind in-place path re-syncs
    # linecache itself inside `_rewrite`/`_rewrite_class_policy`; only the
    # FRESH-install path (new policy or kind change) needs the slot populated here
    # — written just before its exec below (#4). (The immutable-default refusal,
    # which must NOT touch this slot, already returned above, so there is no
    # rejected source that could poison the <policy-{name}> slot.)

    if name in _PLM_POLICIES:                             # re-decoration (Guard B; non-immutable)
        existing = _PLM_POLICIES[name]
        same_kind = isinstance(existing, _FunctionPolicy) == inspect.isfunction(obj)
        if same_kind:                                     # edit in place (identity kept where possible)
            if isinstance(existing, _FunctionPolicy):
                existing._rewrite(source)
            else:
                _rewrite_class_policy(existing, source)
            # Re-fetch the CANONICAL object from the registry rather than blindly
            # reusing `existing`. A class rewrite that needs a structural change
            # (metaclass / __slots__ / incompatible bases) can't apply in place —
            # `_rewrite_class_policy` recreates the class via `_remove_and_recreate`,
            # which installs the NEW class in `_PLM_POLICIES[name]` AND `__main__`.
            # Rebinding/returning the old `existing` here would clobber `__main__`
            # back to the dead class and hand the caller a stale object (registry
            # vs __main__/return divergence). The function path and the in-place
            # class path keep identity, so the registry value IS `existing` there —
            # so reading it back is correct in every case.
            result = _PLM_POLICIES.get(name, existing)
            main_globals[name] = result
            return result
        _PLM_POLICIES.pop(name, None)                     # kind changed (fn<->class):
        main_globals.pop(name, None)                      # drop old, fall through to fresh
        # create — identity is NOT preserved across a kind change (can't be).

    # FRESH install (new policy or kind change): NOW populate the linecache slot
    # (deferred from the top so a refused/no-op re-decoration never touches it).
    linecache.cache[stable] = (len(source), None, source.splitlines(True), stable)
    # Re-exec under <policy-{name}> (NOT obj as-is) so policy code lives in its OWN
    # linecache slot — the cell's <cell-N> slot is disposable and is NOT rebuilt
    # after respawn, whereas <policy-*> IS. Makes tracebacks/getsource work
    # pre-edit, post-edit, and post-respawn for functions and class methods alike.
    if inspect.isfunction(obj):
        ns = {}
        exec(compile(source, stable, "exec"), main_globals, ns)   # under __main__; a `def`
        proxy = _FunctionPolicy(ns[name], source, stable)         # has no side effects
        _PLM_POLICIES[name] = main_globals[name] = proxy
        return proxy

    ns = {}
    exec(compile(source, stable, "exec"), main_globals, ns)       # under __main__ too
    cls = ns[name]
    _attach_class_policy_metadata(cls, source, stable)
    _PLM_POLICIES[name] = main_globals[name] = cls
    return cls


def _get_source_for_obj(obj, cell_filename):
    """Recover obj's VERBATIM source (comments + formatting preserved), @policy
    line removed.

    FUNCTIONS -> inspect.getsource(obj): OBJECT-SPECIFIC via the code object's own
    co_firstlineno, so same-name siblings / nested same-name defs / a redefinition
    in one cell / policy-creates-policy all resolve to exactly this object — no
    hand-rolled matching. dedent (nested defs come back indented) then strip.

    CLASSES -> inspect.getsource can't be used (a class whose module lacks
    __file__ raises "is a built-in class") and classes have no co_firstlineno, so
    parse the cell and take the FIRST same-named ClassDef. Don't define the same
    CLASS policy name twice in one cell. Raises on source-unavailable / not-found.
    """
    if inspect.isfunction(obj):
        try:
            raw = inspect.getsource(obj)
        except (OSError, TypeError) as e:
            raise RuntimeError(f"@policy: cannot read source for {obj.__name__!r}: {e}")
        return _strip_policy_decorator(textwrap.dedent(raw))   # dedent THEN strip
    full = "".join(linecache.getlines(cell_filename))
    if not full:
        raise RuntimeError(f"@policy: no linecache for {cell_filename!r}")
    for node in ast.walk(ast.parse(full)):
        if isinstance(node, ast.ClassDef) and node.name == obj.__name__:
            return _slice_policy_source(full, node)            # verbatim slice
    raise RuntimeError(f"@policy: no class {obj.__name__!r} in {cell_filename!r}")


def _warn_if_captures_locals(obj):
    # Free vars = names captured from an ENCLOSING FUNCTION's locals (true
    # closures). They don't survive standalone re-exec -> NameError at call. REPL
    # globals are NOT free vars (resolve via __globals__) and are fine. __class__
    # is the normal super()/__class__ cell, not a real capture — skip it.
    if inspect.isfunction(obj):
        free = tuple(n for n in obj.__code__.co_freevars if n != "__class__")
    else:
        seen = set()
        for v in vars(obj).values():                     # _iter_funcs handles methods,
            for fn in _iter_funcs(v):                     # static/classmethods, property
                seen.update(n for n in fn.__code__.co_freevars if n != "__class__")
        free = tuple(sorted(seen))
    if free:
        _policy_note(
            f"@policy {obj.__name__!r}: captures enclosing-function local(s) {free} "
            f"— lost when the policy is re-exec'd standalone; will NameError at call "
            f"unless they exist as REPL globals. Reference REPL globals or inline.")
