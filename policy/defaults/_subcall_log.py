"""Harness-side per-sub-call logging.

NOT a policy (leading underscore -> skipped by iter_default_policies).

Why this exists: sub-call failure is a VALUE (`None`), not an exception, so a
policy that does not check its result will propagate a wrong answer instead of
crashing. Failure attribution must therefore never be inferred from what a
policy returned. Every sub-call records its own outcome here, and the runner
assigns failure classes by reading this log FIRST.

Records are appended as JSON lines to `$PLM_SUBCALL_LOG`. When that variable is
unset the logger is a no-op, so the harness stays usable outside the experiment.

Fields:
    ts            wall-clock at completion
    primitive     "llm" | "react_auto"
    granted       rounds/attempts the sub-call was given
    used          rounds/attempts it actually consumed
    truncated     True if the budget ran out without a delivered result
    clamped       True if the requested budget exceeded the safety cap
    requested     the budget originally asked for (present only when clamped)
    outcome       "ok" | "truncated" | "constraint_exhausted"
"""

from __future__ import annotations


def record(**fields):
    """Append one sub-call record. Never raises: logging must not be able to
    break a run, and a lost record is preferable to a crashed policy."""
    try:
        import json
        import os
        import time

        path = os.environ.get("PLM_SUBCALL_LOG")
        if not path:
            return
        fields.setdefault("ts", time.time())
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(fields, default=str) + "\n")
            fh.flush()
    except Exception:
        pass
