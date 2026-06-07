"""The canonical policy objects and their edit machinery.

Function policy = a thin `_FunctionPolicy` proxy holding `_inner` (edits swap
`_inner`; every reference shares the one proxy). Class policy = the REAL class,
edited by in-place `__dict__` mutation (subclassing needs a true type; no
metaclass). Both expose the same edit API (`_rewrite`/`_edit`/`_insert`/
`_delete_lines`/`_remove`/`_duplicate`) — methods on the proxy, classmethods on
the class.

Feedback (edit failures, closures, structural fallback) goes through
`_policy_note` — a DIRECT write to the captured stderr, so PLM sees EVERY
message (no `warnings` dedup, no clobberable filter, no `proxy.py:line` noise).

Phase 2 extensions:

* **Immutability seal**: a DEFAULT policy carries an intrinsic `_p_immutable`
  flag (set once by `registry._seal`). `_rewrite` / `_rewrite_class_policy` /
  `_remove` / `_class_remove` early-return with a `_policy_note(... immutable;
  duplicate it ...)` when that flag is True, and `_rewrite`/`_rewrite_class_policy`
  also refuse a RENAME whose target name is already a policy or a kernel-internal
  name (no clobbering a default via a rename collision). `_FunctionPolicy.__setattr__`
  additionally freezes `_inner`, `_p_name`, and the `_p_immutable` flag itself on a
  sealed proxy — tight scope keeps `functools.update_wrapper` and dill rehydration
  working (the flag is the last `_p_*` slot, so it flips True only after the others
  restore). The single registry (`registry._PolicyStore`) backstops all of this by
  refusing to replace/remove a default-valued entry while sealed.

* **Policy-call depth cap** (`POLICY_CALL_DEPTH_CAP`): a `ContextVar` +
  `_policy_call(name)` context manager bound at EVERY `@policy` boundary
  (function via `_FunctionPolicy.__call__`, class via the wrap installed in
  `_attach_class_policy_metadata`). It increments ONCE per @policy call — so it
  is a UNIFORM logical-depth bound (a function policy costs 2 native frames, a
  class policy more, a plain fn 1; the counter ignores that and counts each call
  as 1). The kernel raises `sys.setrecursionlimit` so Python's uneven native
  frame limit doesn't pre-empt this counter, and the cap sits below CPython's
  hard C-recursion guard so the clear message fires first. Distinct from
  `_LLM_DEPTH` (which bounds sub-LLM-agent recursion specifically).

* **`_duplicate`/`_class_duplicate`** thin wrappers calling
  `duplicate_policy(name, new_name)`.
"""

from __future__ import annotations

import ast
import functools
import inspect
import linecache
import sys
import types
from contextlib import contextmanager
from contextvars import ContextVar

from .edits import (
    _compute_delete,
    _compute_edit,
    _compute_insert,
    _iter_funcs,
    _strip_policy_decorator,
)
from .registry import (
    _PLM_POLICIES,
    _main,
)


def _policy_note(msg: str) -> None:
    # Straight to the captured stderr (PREFIX points sys.stderr at the cell's
    # buffer). No dedup, no warnings-filter dependence, no proxy.py:line noise —
    # PLM sees every edit failure, even N within one cell.
    sys.stderr.write(f"[policy] {msg}\n")


# --- policy-call depth cap (Phase 2) -----------------------------------------

# UNIFORM recursion bound: counts each @policy boundary as 1 — a function policy
# (proxy __call__ + body = 2 native frames) and a class policy alike — so the
# limit is on LOGICAL @policy depth, not on Python's uneven native frame count.
# The kernel raises sys.setrecursionlimit (see repl/prefix.py) so the native
# frame limit no longer pre-empts this counter; the cap sits BELOW CPython's hard
# C-recursion guard (which caps @policy dispatch at a few thousand regardless of
# setrecursionlimit) so THIS deterministic, clearer message fires first. Distinct
# from _LLM_DEPTH (which bounds sub-LLM-agent recursion specifically).
# 3000 sits comfortably under CPython's hard C-recursion guard (~3331 @policy calls
# when measured on 3.13; we target 3.12, where it is in the same ballpark). On any
# version/platform whose C-guard is lower, the native RecursionError fires first
# instead (still safe, just without the friendlier message).
POLICY_CALL_DEPTH_CAP: int = 3000
_POLICY_CALL_DEPTH: ContextVar[int] = ContextVar(
    "_POLICY_CALL_DEPTH", default=0
)


@contextmanager
def _policy_call(name: str):
    """Increment `_POLICY_CALL_DEPTH` at every @policy boundary; raise
    `RecursionError` at `POLICY_CALL_DEPTH_CAP`. This is a UNIFORM bound — each
    @policy call counts as 1 (function or class policy alike), regardless of how
    many native Python frames it actually spends — so PLM-authored recursion is
    bounded by logical @policy depth, not by Python's uneven per-call frame cost.
    Plain (non-@policy) Python recursion is bounded separately by the kernel's
    sys.setrecursionlimit.
    """
    d = _POLICY_CALL_DEPTH.get()
    if d >= POLICY_CALL_DEPTH_CAP:
        raise RecursionError(
            f"Policy call depth {d} reached cap ({POLICY_CALL_DEPTH_CAP}) "
            f"calling {name!r}. Refactor to iteration or compose with react_llm."
        )
    tok = _POLICY_CALL_DEPTH.set(d + 1)
    try:
        yield
    finally:
        _POLICY_CALL_DEPTH.reset(tok)


# --- function policy ----------------------------------------------------------

class _FunctionPolicy:
    __slots__ = ("_inner", "_p_source", "_p_version", "_p_filename",
                 "_p_name", "_p_immutable", "__dict__")     # __dict__ -> functools.update_wrapper

    def __init__(self, fn, source, filename):
        self._inner = fn
        self._p_source = source
        self._p_version = 0
        self._p_filename = filename          # stable "<policy-{name}>" for linecache
        self._p_name = fn.__name__           # canonical name; survives rename
        self._p_immutable = False            # a "default policy" iff True; set only by _seal()
        functools.update_wrapper(self, fn)

    def __setattr__(self, name, value):
        # Seal: once this proxy is a DEFAULT policy (`_p_immutable` True), freeze
        # the body (`_inner`), the canonical name (`_p_name`), and the flag itself
        # (no un-sealing) — so a default can't be hot-swapped, renamed, or
        # un-sealed. Immutability is INTRINSIC to the object, not a name-set
        # lookup. Writes during __init__ / dill-restore pass because
        # `_p_immutable` is still False/unset then (it is the last `_p_*` slot, so
        # `_inner`/`_p_name` restore before it flips True). Other attrs
        # (`_p_source`/`_p_version`/`_p_filename`/__dict__/update_wrapper) are
        # introspection-only and pass.
        if getattr(self, "_p_immutable", False):
            if name in ("_inner", "_p_name") or (name == "_p_immutable" and value is not True):
                raise TypeError(
                    f"{getattr(self, '_p_name', '?')!r} is an immutable default policy; "
                    f"cannot set {name!r}. Use `duplicate_policy("
                    f"{getattr(self, '_p_name', '?')!r}, '<new_name>')` to fork it."
                )
        super().__setattr__(name, value)

    def __call__(self, *a, **k):
        # Wrap every call with `_policy_call` so the policy-call-depth cap
        # increments at this boundary regardless of caller (LLM defaults,
        # PLM-authored mutables, top-level cells — all the same).
        with _policy_call(self._p_name):
            return self._inner(*a, **k)      # ~50ns forward + 1 contextvar inc/dec

    @property
    def __signature__(self):
        return inspect.signature(self._inner)

    def __repr__(self):
        return f"<policy {self._p_name} v{self._p_version}>"

    # --- edit API (thin wrappers over the shared _compute_* + the _rewrite choke point) ---

    def _edit(self, old, new, *, replace_all=False):
        ns = _compute_edit(self._p_source, old, new, replace_all)
        if ns is not None:
            self._rewrite(ns)

    def _insert(self, after_line, content):
        ns = _compute_insert(self._p_source, after_line, content)
        if ns is not None:
            self._rewrite(ns)

    def _delete_lines(self, start, end=None):
        ns = _compute_delete(self._p_source, start, end)
        if ns is not None:
            self._rewrite(ns)

    def _remove(self):
        # Immutability check: refuse to remove a default policy. Other
        # paths (`del <name>`, `globals().pop(<name>)`) are handled by
        # Guard A (pre-reject) and Guard C (restore canonical), and the registry
        # store refuses to pop a default — this catches the explicit
        # `proxy._remove()` call too.
        if getattr(self, "_p_immutable", False):
            _policy_note(
                f"{self._p_name!r} is immutable; cannot remove. "
                f"It is part of the LLM-default base."
            )
            return
        g = _main()                          # kernel __main__ (where @policy injected)
        g.pop(self._p_name, None)
        _PLM_POLICIES.pop(self._p_name, None)

    def _duplicate(self, new_name):
        """Fork this policy under `new_name`. Returns the new mutable copy,
        or None if duplication is refused (un-duplicable / name taken)."""
        from .registry import duplicate_policy
        return duplicate_policy(self._p_name, new_name)

    def _rewrite(self, new_source):
        # Immutability gate: refuse to rewrite a DEFAULT policy. This also covers
        # `_edit`/`_insert`/`_delete_lines` (they funnel through `_rewrite`)
        # and `@policy` re-decoration of a default (the decorator's
        # same-kind path calls `_rewrite` on the existing proxy).
        if getattr(self, "_p_immutable", False):
            _policy_note(
                f"{self._p_name!r} is immutable; source unchanged. "
                f"Use `duplicate_policy({self._p_name!r}, '<new_name>')` "
                f"to fork it (refused for un-duplicable LLM defaults)."
            )
            return
        # Single choke point: @policy-strip, parse, one-def rule, compile+exec,
        # swap _inner, linecache sync (name-tracking filename), version, rename.
        try:
            new_source = _strip_policy_decorator(new_source)  # strip ALSO parses -> keep
            tree = ast.parse(new_source)                      # inside the try (else a
        except SyntaxError as e:                              # SyntaxError escapes via strip)
            _policy_note(f"{self._p_name}._rewrite: SyntaxError {e}; source unchanged")
            return
        defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(tree.body) != 1 or not defs:
            _policy_note(f"{self._p_name}._rewrite: a policy is exactly one top-level "
                         f"def (got {len(tree.body)} statement(s)); source unchanged. "
                         f"Nest helpers inside the function body; reference REPL globals.")
            return
        new_name = defs[0].name
        # Rename-collision guard: a rename must not CLOBBER another policy or a
        # kernel-internal name. Without this, `mutable._rewrite("def natural_llm():
        # ...")` would take over the immutable default's slot (the gate above only
        # checks THIS proxy's flag). Editing in place (new_name == old) is fine.
        if new_name != self._p_name and (
            new_name in _PLM_POLICIES or new_name in (_main().get("_REPL_INJECTED") or ())
            or new_name in _main()                       # also: don't silently clobber a plain global
        ):
            _policy_note(
                f"{self._p_name}._rewrite: refusing to rename to {new_name!r} — that name "
                f"is already a policy, a kernel-internal name, or an existing variable. Edit "
                f"in place (keep the def name) or rename to a fresh name."
            )
            return
        new_filename = f"<policy-{new_name}>"   # filename TRACKS the name -> no stale-slot collision
        cell_globals = _main()                  # re-exec'd fn closes over kernel __main__ (REPL globals)
        ns = {}
        try:
            exec(compile(tree, new_filename, "exec"), cell_globals, ns)  # compile INSIDE the try:
        except Exception as e:                  # catches compile-only SyntaxErrors ast.parse
            _policy_note(f"{self._p_name}._rewrite: bad source ({e!r}); source unchanged")
            return                              # accepts (await-in-sync, dup arg, ...) + def errors
        new_obj = ns[new_name]
        old_name, old_filename = self._p_name, self._p_filename
        self._inner, self._p_source = new_obj, new_source
        self._p_version += 1
        self._p_name, self._p_filename = new_name, new_filename
        functools.update_wrapper(self, new_obj)
        linecache.cache[new_filename] = (
            len(new_source), None, new_source.splitlines(True), new_filename)
        if new_filename != old_filename:
            linecache.cache.pop(old_filename, None)   # drop the renamed-from slot
        if new_name != old_name:
            cell_globals.pop(old_name, None); cell_globals[new_name] = self
            _PLM_POLICIES.pop(old_name, None)
        _PLM_POLICIES[new_name] = self


# --- class policy -------------------------------------------------------------

# Kept OUT of the diff. __dict__/__weakref__/__qualname__/__module__ are
# structural/identity. __doc__ and __annotations__ are user content BUT are
# updated by explicit setattr after the diff (NOT through it) — because
# delattr(cls, "__doc__") raises TypeError (not the AttributeError the diff
# catches), so a diff that put __doc__ in the "gone" set could break the rewrite.
_CLASS_PRESERVE = frozenset({"__dict__", "__weakref__", "__doc__",
                             "__qualname__", "__annotations__", "__module__"})
_POLICY_API = frozenset({"_rewrite", "_edit", "_insert", "_delete_lines",
                         "_remove", "_duplicate"})


class PolicyStructuralFallbackWarning(UserWarning):
    """Marker type for the (rare) class-edit-couldn't-apply-in-place event.

    Surfaced to PLM as a `_policy_note` to stderr — kept as a class only so
    callers/tests can refer to the concept by name.
    """


def _class_origin_filename(cls):
    # @policy on a class runs right after `class X:` — the caller frame is the
    # cell (or a parent policy's <policy-*> body). Walk past plm.policy.* frames.
    f = sys._getframe(1)
    while f is not None:
        if not (f.f_globals.get("__name__") or "").startswith("plm.policy"):
            return f.f_code.co_filename
        f = f.f_back
    return "<unknown>"


def _rebind_class_cells(v, cls):
    # A method using zero-arg super() (or bare __class__) closes over a __class__
    # cell bound to the class it was DEFINED in. Copying it off the throwaway
    # new_cls onto the live cls leaves that cell pointing at new_cls -> super()
    # raises "obj must be an instance or subtype of type" on cls instances.
    # Rebind the cell to cls. (One cell is shared by all of a class's super()-using
    # methods, so rebinding each copied method converges.) Covers methods,
    # static/classmethods, and property accessors via the shared _iter_funcs.
    for fn in _iter_funcs(v):
        code = fn.__code__
        if "__class__" in code.co_freevars and fn.__closure__:
            try:
                fn.__closure__[code.co_freevars.index("__class__")].cell_contents = cls
            except Exception:
                pass                         # cell_contents is writable in 3.7+


def _attach_class_policy_metadata(cls, source, filename):
    cls._p_source, cls._p_version, cls._p_filename = source, 0, filename
    cls._p_immutable = False             # a "default policy" iff True; set only by _seal()
                                         # (preserved across rewrites — _p_* keys skip the diff)
    # Attach the edit API as CLASSMETHODS so `Net._rewrite(src)` mirrors
    # `predict._rewrite(src)`. Attach the MODULE-LEVEL functions DIRECTLY (each
    # already takes cls first) — a named module function pickles BY REFERENCE, so
    # a class policy survives crash-restart cleanly (lambdas would pickle by
    # value and are the most fragile thing to serialize). Preserved across
    # rewrites (the __dict__ diff skips _POLICY_API).
    cls._rewrite = classmethod(_rewrite_class_policy)
    cls._edit = classmethod(_class_edit)
    cls._insert = classmethod(_class_insert)
    cls._delete_lines = classmethod(_class_delete_lines)
    cls._remove = classmethod(_class_remove)
    cls._duplicate = classmethod(_class_duplicate)

    # Wrap `cls.__call__` with `_policy_call` so EVERY class-policy invocation
    # increments the policy-call-depth contextvar. functools.wraps preserves
    # `__wrapped__` so the bootstrap blessing loop (and any future
    # source-extraction) can recover the original via `__wrapped__`. No LLM
    # default is a class in v1, so this only fires for PLM-authored class
    # policies — it's a structural cap on their depth, advisory in the sense
    # that a mutable class policy CAN edit its own __call__ to remove the wrap
    # (documented irreducible boundary; doesn't bypass the LLM-depth gate
    # because `_make_backend`/`descend`/`llm_call` are blessed-caller-gated
    # regardless of any class-policy wrap).
    # Only wrap a PLAIN instance-method __call__ (types.FunctionType). A
    # @staticmethod/@classmethod __call__ stored in cls.__dict__ is a descriptor
    # object, not a function: calling it as `_orig_call(self, *a, **k)` would
    # mis-forward `self` (staticmethod: "takes 0 positional args but 1 given") or
    # fail outright (classmethod object isn't callable). Such a __call__ ignores
    # instance state and is degenerate, so skip the depth-wrap — it's safe: the
    # LLM-depth gate is enforced independently by the blessed-caller checks in
    # _llm_infra, not by this advisory class-call cap. (#R5-6)
    _orig_call = cls.__dict__.get("__call__")
    if isinstance(_orig_call, types.FunctionType):
        @functools.wraps(_orig_call)
        def _wrapped_call(self, *a, **k):
            with _policy_call(type(self).__name__):
                return _orig_call(self, *a, **k)
        _wrapped_call._plm_depth_wrapped = True       # OUR marker; the re-wrap path keys on
        cls.__call__ = _wrapped_call                  # this (NOT __wrapped__) to detect double-wrap


def _class_edit(cls, old, new, *, replace_all=False):
    ns = _compute_edit(cls._p_source, old, new, replace_all)
    if ns is not None:
        _rewrite_class_policy(cls, ns)


def _class_insert(cls, after_line, content):
    ns = _compute_insert(cls._p_source, after_line, content)
    if ns is not None:
        _rewrite_class_policy(cls, ns)


def _class_delete_lines(cls, start, end=None):
    ns = _compute_delete(cls._p_source, start, end)
    if ns is not None:
        _rewrite_class_policy(cls, ns)


def _class_remove(cls):
    # Immutability check: refuse to remove an immutable class policy. v1 has
    # no immutable class default, but v1.5 may; keep the gate here so adding
    # one later doesn't require touching this method.
    if getattr(cls, "_p_immutable", False):
        _policy_note(
            f"{cls.__name__!r} is immutable; cannot remove. "
            f"It is part of the LLM-default base."
        )
        return
    g = _main()
    g.pop(cls.__name__, None)
    _PLM_POLICIES.pop(cls.__name__, None)


def _class_duplicate(cls, new_name):
    """Fork this class policy under `new_name`. Returns the new mutable copy,
    or None if duplication is refused (un-duplicable / name taken)."""
    from .registry import duplicate_policy
    return duplicate_policy(cls.__name__, new_name)


def _rewrite_class_policy(cls, new_source):
    # Immutability gate: refuse to rewrite a DEFAULT class policy.
    if getattr(cls, "_p_immutable", False):
        _policy_note(
            f"{cls.__name__!r} is immutable; source unchanged. "
            f"Use `duplicate_policy({cls.__name__!r}, '<new_name>')` "
            f"to fork it (refused for un-duplicable LLM defaults)."
        )
        return
    try:
        new_source = _strip_policy_decorator(new_source)   # strip ALSO parses -> keep inside
        tree = ast.parse(new_source)
    except SyntaxError as e:
        _policy_note(f"{cls.__name__}._rewrite: SyntaxError {e}; source unchanged")
        return
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
        _policy_note(f"{cls.__name__}._rewrite: a policy is exactly one top-level "
                     f"class (got {len(tree.body)} statement(s)); source unchanged. "
                     f"Nest helpers/attrs inside the class body; reference REPL globals.")
        return
    new_name = tree.body[0].name
    # Rename-collision guard (mirrors _FunctionPolicy._rewrite): a rename must not
    # clobber another policy or a kernel-internal name. Edit in place (new_name ==
    # old) is fine.
    if new_name != cls.__name__ and (
        new_name in _PLM_POLICIES or new_name in (_main().get("_REPL_INJECTED") or ())
        or new_name in _main()                           # also: don't silently clobber a plain global
    ):
        _policy_note(
            f"{cls.__name__}._rewrite: refusing to rename to {new_name!r} — that name "
            f"is already a policy, a kernel-internal name, or an existing variable. Edit "
            f"in place or rename to a fresh name."
        )
        return
    new_filename = f"<policy-{new_name}>"      # filename tracks name; methods compiled
    ns = {}                                    # under it get the matching co_filename
    try:
        exec(compile(tree, new_filename, "exec"), _main(), ns)
    except Exception as e:                     # compile-only SyntaxError OR class-body error
        _policy_note(f"{cls.__name__}._rewrite: bad source ({e!r}); source unchanged")
        return
    new_cls = ns[new_name]

    # Structural checks FIRST (can't mutate metaclass/slots/incompatible bases in
    # place) -> fall back to remove+recreate, leaving cls.__dict__ untouched. Pass
    # the already-built new_cls so the class body isn't re-executed.
    if type(new_cls) is not type(cls):
        return _remove_and_recreate(cls, new_cls, new_source)
    if new_cls.__dict__.get("__slots__") != cls.__dict__.get("__slots__"):
        return _remove_and_recreate(cls, new_cls, new_source)   # slots define layout
    if cls.__bases__ != new_cls.__bases__:
        try:
            cls.__bases__ = new_cls.__bases__
        except TypeError:
            return _remove_and_recreate(cls, new_cls, new_source)

    old_filename = cls._p_filename
    linecache.cache[new_filename] = (
        len(new_source), None, new_source.splitlines(True), new_filename)
    if new_filename != old_filename:
        linecache.cache.pop(old_filename, None)    # drop the renamed-from slot
    # Diff user keys, PRESERVING: policy metadata (_p_*), _CLASS_PRESERVE, the
    # edit API (_POLICY_API), AND the slot machinery (__slots__ + member
    # descriptors). __slots__ is guaranteed unchanged here (a slots diff already
    # fell back), so cls's OWN descriptors are correct — copying new_cls's would
    # rebind __objclass__ and break attribute access on existing instances.
    _sl = cls.__dict__.get("__slots__", ())
    _slot_names = {_sl} if isinstance(_sl, str) else set(_sl or ())
    _skip = _CLASS_PRESERVE | _POLICY_API | {"__slots__"} | _slot_names
    user = lambda d: {k for k in d if not k.startswith("_p_") and k not in _skip}
    for gone in user(cls.__dict__) - user(new_cls.__dict__):
        try:
            delattr(cls, gone)
        except AttributeError:
            pass
    for k in user(new_cls.__dict__):
        v = new_cls.__dict__[k]
        _rebind_class_cells(v, cls)        # fix zero-arg super(): rebind __class__ cell
        setattr(cls, k, v)
    # __doc__/__annotations__: user content, updated by explicit setattr (NOT via
    # the diff) — delattr(cls, "__doc__") raises TypeError, so they must never go
    # through the diff's delattr path. Mirrors how functions update __doc__.
    cls.__doc__ = new_cls.__dict__.get("__doc__")
    cls.__annotations__ = dict(new_cls.__dict__.get("__annotations__") or {})

    cls._p_source = new_source
    cls._p_version += 1
    cls._p_filename = new_filename
    old = cls.__name__
    if new_name != old:
        cls.__name__ = cls.__qualname__ = new_name
        g = _main()
        g.pop(old, None); g[new_name] = cls
        _PLM_POLICIES.pop(old, None)
    _PLM_POLICIES[new_name] = cls

    # If the rewrite re-introduced a fresh `__call__` (user content changed),
    # re-wrap it with `_policy_call` so the depth cap continues to fire. Gate on
    # OUR OWN private marker, NOT the generic `__wrapped__`: an LLM-authored
    # `__call__` decorated with `@functools.wraps`/`@lru_cache`/etc. carries
    # `__wrapped__` too, so the old `__wrapped__ is None` check skipped wrapping
    # it — leaving that class policy's `__call__` UNCAPPED (#12). `_plm_depth_wrapped`
    # is set only by us, so it distinguishes "already wrapped by us" from "user
    # used a wraps-style decorator".
    _inner_call = cls.__dict__.get("__call__")
    # Same guard as the install path: only re-wrap a plain instance-method
    # __call__, never a staticmethod/classmethod descriptor (#R5-6). The
    # isinstance check short-circuits before the _plm_depth_wrapped lookup, so a
    # descriptor is skipped cleanly.
    if isinstance(_inner_call, types.FunctionType) and not getattr(
        _inner_call, "_plm_depth_wrapped", False
    ):
        @functools.wraps(_inner_call)
        def _wrapped_call(self, *a, **k):
            with _policy_call(type(self).__name__):
                return _inner_call(self, *a, **k)
        _wrapped_call._plm_depth_wrapped = True
        cls.__call__ = _wrapped_call


def _remove_and_recreate(old_cls, new_cls, new_source):
    # new_cls was already built by the caller's exec under <policy-{new_name}>
    # (no re-exec -> no double class-body side effects; methods' co_filename
    # matches new_filename). Re-attach the edit API so the replacement is itself
    # a fully-editable policy, with its filename tracking its (possibly new) name.
    name = old_cls.__name__
    g = _main()
    _PLM_POLICIES.pop(name, None); g.pop(name, None)
    new_filename = f"<policy-{new_cls.__name__}>"
    _attach_class_policy_metadata(new_cls, new_source, new_filename)
    new_cls._p_version = old_cls._p_version + 1
    linecache.cache[new_filename] = (
        len(new_source), None, new_source.splitlines(True), new_filename)
    if new_filename != old_cls._p_filename:
        linecache.cache.pop(old_cls._p_filename, None)
    _PLM_POLICIES[new_cls.__name__] = g[new_cls.__name__] = new_cls
    _policy_note(
        f"{name!r}: structural edit couldn't apply in place; replaced the class. "
        f"Existing instances created before this edit remain on the OLD class and "
        f"won't see the new behavior.")
