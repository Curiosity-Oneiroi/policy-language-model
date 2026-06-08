"""Kernel-internal state, set by the PREFIX at boot and read by the KERNEL_LOOP guards via a
fresh re-import each cell.

A cell execs in `__main__`, so the PREFIX-injected names a cell could rebind —
`_REPL_INJECTED` (the snapshot blocklist), `_REPL_INJECTED_CANON` (the values Guard C+
restores), `_repl_reset_buffers` (the per-cell stdout/stderr reset) — are plain `__main__`
globals. Rebinding any of them would silently disable the corresponding kernel mechanism
(ship every helper as a snapshot var / revert nothing / leak stdout across cells). Holding the
canonical values HERE and re-importing this module inside the guards (so a fresh `import ...
as` overwrites any cell-rebound `__main__` alias) closes that accidental door, exactly mirroring
the R5-1 re-import of `_audit_cell`/`_post_cell_guard` from `plm.policy`.

This is NOT a security boundary (cells are parent-trusted): a cell that DELIBERATELY does
`import plm.repl._kernel_state as ks; ks.CANON = {}` — or poisons `sys.modules` — still wins,
the same irreducible residual R5-1 documents. It defends the guard MECHANISM against an
accidental rebind of an obscure internal name, not against intentional self-sabotage.

Re-derived fresh on every boot (the PREFIX recomputes + re-assigns these), so nothing here is
snapshotted or carried across a respawn.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, Optional

# Names the per-cell snapshot must NEVER ship as user vars (kernel-owned helpers/modules).
INJECTED: FrozenSet[str] = frozenset()
# Boot-canonical injected callables/modules Guard C+ restores if a cell rebinds/deletes them.
CANON: Dict[str, Any] = {}
# The kernel's stdout/stderr buffer reset (defined in the bootstrap; set here at boot).
reset_buffers: Optional[Callable[[], None]] = None
