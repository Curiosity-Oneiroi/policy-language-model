"""Persistent-kernel REPL session for PLM.

`PythonReplSession` runs a venv'd Python kernel as a child process; cell
execution is serial and stateful (vars persist round-to-round). The kernel
script is assembled from text constants — `KERNEL_BOOTSTRAP` (opens a
UNIX socket to the parent), `PREFIX` (boilerplate + RETURN +
helpers), and `KERNEL_LOOP` (request/response loop).

Cells return stdout, stderr, and (for RETURN) a return object. The kernel
also ships a vars snapshot every cell — used internally for crash-restart
rehydrate only; never surfaced to the model.
"""

from __future__ import annotations

from .session import PythonReplSession, DEFAULT_PREINSTALL
from .kernel import KERNEL_BOOTSTRAP, KERNEL_LOOP
from .prefix import PREFIX

__all__ = [
    "PythonReplSession",
    "DEFAULT_PREINSTALL",
    "KERNEL_BOOTSTRAP",
    "KERNEL_LOOP",
    "PREFIX",
]
