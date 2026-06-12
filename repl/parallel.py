"""Generic REPL helper: run zero-arg callables in parallel.

NOT a policy; not an `@policy`-machinery feature. Just an always-present REPL
builtin imported by PREFIX BEFORE the `_REPL_INJECTED = {...}` freeze, so the
name joins the kernel-internal frozen set (snapshot-skipped; always available
after respawn; @policy refuses to shadow it).

Multi-threaded (the work runs in OS threads), asyncio-orchestrated (for clean
gather/order/exception semantics). The current contextvar context is copied into
each branch via `copy_context().run(...)`, so whatever is in context carries through
— e.g. the LLM-depth gate stays in effect for any sub-LLM work a branch happens to do.
"""

from __future__ import annotations

from plm._branch_state import (
    _BRANCH_EDIT_GRANTS,
    _EditGrant,
    _IN_PARALLEL_BRANCH,
    ParallelMutationError,
    with_edit,
)

__all__ = ["parallel", "with_edit", "ParallelMutationError"]


def parallel(*tasks, max_workers=None):
    """Run zero-arg callables in parallel; return a list of results (or exceptions) in
    input order.

    A GENERAL parallel-execution helper — run ANY work concurrently: parallel policy
    calls, parallel sub-LLM calls, heavy computation, I/O, whatever. It copies the
    current contextvar context into each branch, so anything in context carries through
    (e.g. the LLM-depth gate keeps any nested sub-LLM work a branch does bounded).

    Pool sizing:
      - Default: `min(len(tasks), 256)` — sized to the batch, capped at 256
        to keep thread creation bounded.
      - Override with `max_workers=N` (clamped to `[1, 1024]`).
      - Invalid `max_workers` (non-numeric, None, `inf`/overflow) falls back to default.

    Contract:
      - Each task is a zero-arg callable (typically a lambda capturing args).
      - Returns a LIST, same length + order as `tasks`. Each slot is the task's return
        value (`None` if it returned nothing), OR — if the task RAISED — the EXCEPTION
        OBJECT it raised. `parallel()` is FULLY ISOLATING: it NEVER propagates anything a
        task does. That includes control-flow `BaseException`s a policy may raise
        (`SystemExit`/`KeyboardInterrupt`/`GeneratorExit`/`CancelledError`) — those are
        RETURNED in the slot too, never re-raised. So check a slot with
        `isinstance(r, BaseException)` (NOT `Exception`, which would miss those and read a
        returned `SystemExit` as a successful result).
      - ISOLATED: every task runs to completion; one task raising does NOT abort the
        others. `parallel()` waits for ALL of them (you always get every result), so it
        returns once the slowest task finishes.
      - The REPL is synchronous: calling `parallel()` from inside a running event loop
        raises a clear RuntimeError (it cannot nest into one) — that is parallel()'s OWN
        error, the ONLY thing it ever raises, never a task's. Nested `parallel(...)` from a
        worker thread (no loop there) works — each makes its own loop.

    World mutation inside a branch (closed-to-mutation, opt-in by grant):
      - A branch READS the world freely (call any policy, read vars) and RETURNS a value. It may
        NOT mutate the shared policy world by default — that would race a sibling.
      - To EDIT a policy in a branch you must hand that branch the capability explicitly, with
        `with_edit(task, that_policy)`. The grant is for an IN-PLACE body edit only, and at most
        ONE branch may be granted a given policy (exclusive). A granted policy is that branch's
        alone for the call — no sibling should read, call, or edit it meanwhile.
      - Renaming, removing, duplicating, by-name edits (`edit_policy(...)`), `@policy`, and unseal
        change the shared registry/namespace and are NOT allowed in a branch — do them in the main
        loop. A violation raises `ParallelMutationError`, returned in that branch's slot (or, for a
        double-grant, raised by `parallel()` itself before anything runs).
      - Don't mutate plain shared data (a global list/dict) from two branches either — RETURN
        values and combine them in the parent, or `deepcopy` isolated input.

    Example:
        # any zero-arg callables — plain functions, policies, sub-LLM calls, ...
        a, b, c = parallel(lambda: heavy_compute(data),
                           lambda: my_policy(x),
                           lambda: natural_llm(m))

        # editing one policy in a branch: grant it (read-only branches stay bare lambdas):
        r_edit, score = parallel(
            with_edit(lambda: predict._edit("old", "new"), predict),
            lambda: evaluate(x),
        )

        # a wide batch; one branch failing leaves a result-or-exception in its slot:
        results = parallel(*[lambda i=i: my_policy(items[i]) for i in range(500)],
                           max_workers=64)
        ok   = [r for r in results if not isinstance(r, BaseException)]
        errs = [r for r in results if isinstance(r, BaseException)]
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    if not tasks:
        return []

    # Unwrap with_edit() grants and enforce EXCLUSIVE borrow UP FRONT (before anything runs): a
    # branch may edit a policy only if granted, and a given policy may be granted to AT MOST ONE
    # branch. A second grant of the same policy is a contract breach raised HERE, by parallel()
    # itself (not inside a branch), so PLM gets it directly rather than in a result slot.
    _plan = []                                         # [(zero-arg callable, frozenset(id(policy))), ...]
    _granted = {}                                      # id(policy) -> policy (for a clear dup message)
    for _t in tasks:
        if isinstance(_t, _EditGrant):
            _ids = set()
            for _p in _t.policies:
                _pid = id(_p)
                if _pid in _granted:
                    raise ParallelMutationError(
                        f"policy {getattr(_p, '_p_name', '?')!r} is granted to more than one "
                        f"parallel() branch; at most ONE branch may edit a given policy.")
                _granted[_pid] = _p
                _ids.add(_pid)
            _plan.append((_t.task, frozenset(_ids)))
        elif callable(_t):
            _plan.append((_t, frozenset()))
        else:
            raise TypeError(
                f"parallel(): each task must be a zero-arg callable or with_edit(task, ...); "
                f"got {_t!r}.")

    if max_workers is None:
        n = min(len(tasks), 256)
    else:
        try:
            n = max(1, min(int(max_workers), 1024))
        except Exception:                              # non-numeric / None / inf-overflow -> default
            n = min(len(tasks), 256)

    # The REPL is synchronous. Detect a running loop BEFORE building the coroutine, so we
    # raise a clear error and never leak an un-awaited coroutine (asyncio.run would also
    # reject it, but only after the coro object exists -> RuntimeWarning).
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass                                           # no running loop in this thread -> proceed
    else:
        raise RuntimeError(
            "parallel() cannot be called from inside a running event loop (the REPL is "
            "synchronous). Call it from ordinary cell / policy code.")

    async def _all():
        loop = asyncio.get_running_loop()
        # An EXPLICIT pool we own and submit to via run_in_executor — NOT the loop's DEFAULT
        # executor: asyncio.run's cleanup joins only the default executor (which we never
        # touch), so we never block ~300s on `shutdown_default_executor`. A fresh
        # `copy_context()` per task carries the current context (e.g. the depth gate) into the
        # worker. `return_exceptions=True` -> gather runs ALL tasks to completion and returns
        # a RESULT-OR-EXCEPTION per task, so one failing branch never aborts the others.
        def _shielded(_fn, _grants):
            # FULL isolation + per-branch state. Before running, mark this context as a parallel
            # branch and install its edit-grant id-set; restore both after. These `.set()`s land in
            # the branch's OWN copied context (copy_context() below), so they're private to it — and
            # the mutation gates (plm._branch_state) read them to allow only a granted in-place edit.
            # A task that raises ANYTHING — a normal Exception OR a control-flow BaseException
            # (SystemExit/KeyboardInterrupt/GeneratorExit/CancelledError), which a policy can
            # legitimately raise, OR a ParallelMutationError from a contract breach — has it CAUGHT
            # and RETURNED in its slot, never propagated out of parallel().
            def _run():
                _tok_b = _IN_PARALLEL_BRANCH.set(True)
                _tok_g = _BRANCH_EDIT_GRANTS.set(_grants)
                try:
                    return _fn()
                except BaseException as _e:
                    return _e
                finally:
                    _BRANCH_EDIT_GRANTS.reset(_tok_g)
                    _IN_PARALLEL_BRANCH.reset(_tok_b)
            return _run
        pool = ThreadPoolExecutor(max_workers=n)         # own pool, not a per-call default
        try:
            futs = [loop.run_in_executor(pool, copy_context().run, _shielded(_fn, _g))
                    for (_fn, _g) in _plan]
            return await asyncio.gather(*futs, return_exceptions=True)   # belt-and-suspenders
        finally:
            pool.shutdown(wait=False)

    return asyncio.run(_all())
