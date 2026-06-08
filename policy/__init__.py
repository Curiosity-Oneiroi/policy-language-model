"""plm.policy — the `@policy` layer: editable, hot-swappable function & class
policies that live in the PLM REPL kernel.

Loaded into the kernel via `PREFIX` (`from plm.policy import (...)`). PLM drives
everything by writing cells — no new parent API, no new frame types. Pure
stdlib, so it imports cleanly in the venv'd subprocess kernel.

Submodules import each other DIRECTLY (`from plm.policy.edits import ...`), never
via this package, so re-exporting here can't create an import cycle.
"""

from __future__ import annotations

from .decorator import policy
from .registry import (
    _SEALED_POLICIES,
    _PLM_POLICIES,
    _install_policy_source,
    delete_lines,
    delete_policy,
    duplicate_policy,
    edit_policy,
    get_policy,
    insert_into,
    list_policies,
    read_policy,
    rewrite_policy,
)
from .guard import _audit_cell, _post_cell_guard
from .proxy import _FunctionPolicy, _policy_note, PolicyStructuralFallbackWarning

__all__ = [
    # the decorator + the by-name helpers PLM calls from cells
    "policy",
    "list_policies", "get_policy", "read_policy",
    "rewrite_policy", "edit_policy", "insert_into", "delete_lines", "delete_policy",
    # Phase 2: duplicate is a general operation on any policy
    "duplicate_policy",
    # kernel-loop integration surface
    "_PLM_POLICIES", "_audit_cell", "_post_cell_guard",
    # Phase 2: shared replay helper + immutability set (bootstrap loader uses these)
    "_install_policy_source", "_SEALED_POLICIES",
    # useful for tests / introspection
    "_FunctionPolicy", "_policy_note", "PolicyStructuralFallbackWarning",
]
