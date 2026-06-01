"""Generic REPL helper: run zero-arg callables in parallel.

NOT a policy; not an `@policy`-machinery feature. Just an always-present REPL
builtin imported by PREFIX BEFORE the `_REPL_INJECTED = {...}` freeze, so the
name joins the kernel-internal frozen set (snapshot-skipped; always available
after respawn; @policy refuses to shadow it).

Multi-threaded (the work runs in OS threads), asyncio-orchestrated (for clean
gather/order/exception semantics + free contextvar propagation via
`asyncio.to_thread`'s `copy_context().run(...)`).
"""

from __future__ import annotations


def parallel(*tasks, max_workers=None):
    """Run zero-arg callables in parallel; return results in input order.

    Use this for parallel LLM calls / heavy work. It propagates the current
    contextvar context (including `_LLM_DEPTH`) into each branch via
    `asyncio.to_thread` (which internally calls `copy_context().run(target)`),
    so each stack trace is independently bounded by the LLM-depth gate.

    Pool sizing:
      - Default: `min(len(tasks), 256)` — sized to the batch, capped at 256
        to keep thread creation bounded.
      - Override with `max_workers=N` (clamped to `[1, 1024]`).
      - Invalid `max_workers` (non-numeric, None) falls back to default.

    Contract:
      - Each task is a zero-arg callable (typically a lambda capturing args).
      - Results come back in input order, regardless of completion order.
      - First exception propagates (asyncio.gather default); other branches'
        threads keep running (Python can't kill threads). For exception
        isolation, wrap each task to return a result-or-error sentinel.
      - Nested `parallel(...)` calls work — each creates its own loop in its
        thread; nested context-copies stack naturally.

    Example:
        results = parallel(
            lambda: natural_llm(m1),
            lambda: natural_llm(m2),
            lambda: react_llm(m3),
        )
        # For very wide batches, opt into a bigger pool:
        results = parallel(*[lambda i=i: natural_llm(qs[i]) for i in range(500)],
                           max_workers=64)
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    if not tasks:
        return []
    if max_workers is None:
        n = min(len(tasks), 256)
    else:
        try:
            n = max(1, min(int(max_workers), 1024))
        except (TypeError, ValueError):
            n = min(len(tasks), 256)

    async def _all():
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(max_workers=n)
        loop.set_default_executor(pool)                # asyncio.to_thread uses this
        try:
            return await asyncio.gather(*(asyncio.to_thread(t) for t in tasks))
        finally:
            pool.shutdown(wait=False)

    return asyncio.run(_all())
