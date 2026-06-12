"""Per-branch execution state for `parallel()` — output routing + mutation-capability gates.

A pure LEAF module (stdlib only). It is imported by BOTH the kernel layer (`plm.repl.*`) and the
domain layer (`plm.policy.*`, `plm.constraint.*`), so it must not import either — and it lives at
the package ROOT (`plm._branch_state`) so importing it triggers only the light `plm.__init__`, never
the heavy `plm.repl.__init__` (which would risk a boot-time import cycle).

It carries the two mechanisms that make a `parallel()` branch safe (see the design plan):

  * OUTPUT ROUTING (fixes the parallel stdout/stderr cross-contamination, "H1"). `sys.stdout`/
    `sys.stderr` become ONE dispatching proxy installed at boot; the real target buffer lives in a
    ContextVar. `exec_ns` `.set()`s it to a private buffer instead of doing a process-global
    `redirect_stdout` swap — so `copy_context()` gives each branch its OWN capture, concurrently,
    with no global swap to poison. Correct off the parallel path too: a plain cell / nested
    `exec_ns` just routes to the live main buffer.

  * MUTATION GATES (closes the concurrent policy-world mutation hazards, "H2-H5"). A branch may
    edit a policy ONLY via an explicit, exclusive grant (`with_edit`), and only an IN-PLACE body
    edit. Structural / by-name / authoring ops (rename, remove, duplicate, `@policy`, unseal,
    `Constraint.registry` writes) are refused inside a branch — they change the shared registry
    membership / global namespace, which no single branch owns. Capability-security: the edit
    capability is HANDED IN, never ambient.
"""

from __future__ import annotations

import sys as _sys
from contextvars import ContextVar


class ParallelMutationError(RuntimeError):
    """A `parallel()` branch attempted a world mutation it isn't allowed to.

    Raised from the mutation entry point (caught by `parallel()`'s per-branch shield and returned
    in that branch's result slot, so PLM sees exactly which branch broke the contract), or — for a
    double-grant — directly from `parallel()` itself before any branch runs.
    """


# --------------------------------------------------------------------------------------------
# (A) Output routing
# --------------------------------------------------------------------------------------------
# "Which buffer does THIS context's stdout/stderr write to?" `None` -> the live main cell buffer
# (a plain cell, or a top-level branch `print`). `exec_ns` / any nested capture `.set()`s these;
# because `parallel()` copies the context per branch, each branch's `.set()` is private to it.
_CURRENT_OUT: ContextVar = ContextVar("_plm_current_out", default=None)
_CURRENT_ERR: ContextVar = ContextVar("_plm_current_err", default=None)


class _StreamProxy:
    """A `sys.stdout`/`sys.stderr` stand-in installed ONCE at boot. Every write routes to the
    buffer named by its ContextVar (per-context), so concurrent branches never share a capture;
    all other file attributes delegate to that buffer."""

    __slots__ = ("_var", "_fallback")

    def __init__(self, var, fallback):
        self._var = var
        self._fallback = fallback        # the real stream; used only before boot sets a buffer

    def _target(self):
        t = self._var.get()
        return t if t is not None else self._fallback

    def write(self, s):
        return self._target().write(s)

    def flush(self):
        return self._target().flush()

    def __getattr__(self, name):         # encoding / isatty / fileno / errors / ... -> the buffer
        return getattr(self._target(), name)


# Installed as sys.stdout/sys.stderr by the bootstrap (see prefix.py / kernel.py).
_OUT_PROXY = _StreamProxy(_CURRENT_OUT, _sys.__stdout__)
_ERR_PROXY = _StreamProxy(_CURRENT_ERR, _sys.__stderr__)


def set_main_streams(out, err):
    """Point the top-level (non-branch) stdout/stderr at the kernel's CURRENT main buffers.

    Called by the bootstrap and by `_repl_reset_buffers` each cell (the kernel recreates the
    buffers per cell). Sets the ContextVars in the kernel-loop context, which every cell execs in;
    `exec_ns` and `parallel()` branches `.set()` over this and restore it.
    """
    _CURRENT_OUT.set(out)
    _CURRENT_ERR.set(err)


# --------------------------------------------------------------------------------------------
# (B) Mutation-capability gates
# --------------------------------------------------------------------------------------------
_IN_PARALLEL_BRANCH: ContextVar = ContextVar("_plm_in_parallel_branch", default=False)
# frozenset of id(policy) THIS branch may edit in place (granted via `with_edit`).
_BRANCH_EDIT_GRANTS: ContextVar = ContextVar("_plm_branch_edit_grants", default=frozenset())


def _policy_label(target):
    return getattr(target, "_p_name", None) or getattr(target, "__name__", None) or "?"


def _edit_guard(target):
    """Gate an IN-PLACE policy edit (proxy `_rewrite` / `_rewrite_class_policy`). Outside a branch:
    no-op. Inside a branch: allowed ONLY if `target` was granted to THIS branch via `with_edit`."""
    if _IN_PARALLEL_BRANCH.get() and id(target) not in _BRANCH_EDIT_GRANTS.get():
        name = _policy_label(target)
        raise ParallelMutationError(
            f"cannot edit policy {name!r} inside a parallel() branch: it was not granted to this "
            f"branch. Wrap the task as `with_edit(task, {name})` (and only ONE branch may be granted "
            f"a given policy), or do the edit in the main loop."
        )


def _rename_guard(target, new_name):
    """A branch grant is for an IN-PLACE body edit; renaming a policy (changing its def name) claims
    a new global name — a namespace change no single branch owns, so it is parent-only."""
    if _IN_PARALLEL_BRANCH.get():
        old = getattr(target, "_p_name", None)
        if new_name != old:
            raise ParallelMutationError(
                f"cannot rename policy {old!r} -> {new_name!r} inside a parallel() branch: a grant "
                f"allows an in-place body edit only (keep the def name). Rename in the main loop."
            )


def _no_branch_mutation(op, *, hint=""):
    """Gate a STRUCTURAL / by-name / authoring op (remove, `@policy`, duplicate, by-name edit,
    unseal, `Constraint.registry` write). These change the policy SET or the global namespace, which
    no single branch owns — always refused inside a branch."""
    if _IN_PARALLEL_BRANCH.get():
        msg = (f"{op} is not allowed inside a parallel() branch — it changes shared "
               f"registry/namespace state. Do it in the main loop.")
        raise ParallelMutationError(msg + (f" {hint}" if hint else ""))


# --------------------------------------------------------------------------------------------
# Granting — `with_edit`
# --------------------------------------------------------------------------------------------
class _EditGrant:
    """Marker returned by `with_edit()`: pairs a branch task with the policies it may edit in place.
    `parallel()` recognizes it, derives the grant id-set, and enforces exclusivity."""

    __slots__ = ("task", "policies")

    def __init__(self, task, policies):
        self.task = task
        self.policies = policies


def with_edit(task, *policies):
    """Grant a `parallel()` branch permission to edit specific policies IN PLACE.

    Wrap a branch task to declare which policies it may mutate:

        parallel(
            with_edit(lambda: predict._edit("old", "new"), predict),  # may edit predict's body
            lambda: score(items),                                     # read-only branch
        )

    Contract: the grant is for an IN-PLACE body edit of the named policy only — NOT a rename,
    remove, duplicate, by-name edit, or `@policy`. At most ONE branch may be granted a given policy,
    and a granted policy is that branch's ALONE for the call (no sibling should read, call, or edit
    it meanwhile). Everything else — authoring, structural changes — belongs to the main loop.
    """
    if not callable(task):
        raise TypeError("with_edit(task, *policies): `task` must be a zero-arg callable (a branch).")
    for p in policies:
        if not hasattr(p, "_p_name"):
            raise TypeError(
                f"with_edit: {p!r} is not a policy — you can only grant edit access to a policy "
                f"object (e.g. `with_edit(task, predict)`)."
            )
    return _EditGrant(task, policies)
