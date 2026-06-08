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

    Example:
        # any zero-arg callables — plain functions, policies, sub-LLM calls, ...
        a, b, c = parallel(lambda: heavy_compute(data),
                           lambda: my_policy(x),
                           lambda: natural_llm(m))

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
        def _shielded(_t):
            # FULL isolation: a task that raises ANYTHING — a normal Exception OR a control-flow
            # BaseException (SystemExit/KeyboardInterrupt/GeneratorExit/CancelledError), which a
            # policy can legitimately raise — has it CAUGHT and RETURNED in its slot, never
            # propagated out of parallel(). (Runs in a worker thread, so such an exception never
            # reached the caller anyway; this just makes the return a guarantee, not a gather
            # quirk.) The task's behaviour/side-effects/context are untouched — only its
            # exception is captured, with its traceback preserved on the object.
            def _run():
                try:
                    return _t()
                except BaseException as _e:
                    return _e
            return _run
        pool = ThreadPoolExecutor(max_workers=n)         # own pool, not a per-call default
        try:
            futs = [loop.run_in_executor(pool, copy_context().run, _shielded(t)) for t in tasks]
            return await asyncio.gather(*futs, return_exceptions=True)   # belt-and-suspenders
        finally:
            pool.shutdown(wait=False)

    return asyncio.run(_all())
