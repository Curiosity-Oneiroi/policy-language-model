"""Discover metaparam SETS under a root directory.

Each immediate sub-directory that contains a `system_prompt.md` (or `.txt`) is a
metaparam set loadable by `PLMMetaParameters.from_dir(<sub-dir>)`. The experiments
UI lists these so a run can be started against a chosen set.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List


def _policy_stems(d: pathlib.Path) -> List[str]:
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.py") if not p.name.startswith("_"))


def discover_metaparams(root) -> List[Dict[str, Any]]:
    """List metaparam sets under `root` (each a from_dir-loadable sub-folder).

    Returns [{name, path, system_prompt_preview, mutable_policies, sealed_policies}].
    A sub-dir WITHOUT a system_prompt is skipped (it isn't a metaparam set).
    """
    root = pathlib.Path(root)
    out: List[Dict[str, Any]] = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith((".", "_")):
            continue
        sp = next((d / f for f in ("system_prompt.md", "system_prompt.txt")
                   if (d / f).is_file()), None)
        if sp is None:
            continue
        out.append({
            "name": d.name,
            "path": str(d.resolve()),
            "system_prompt_preview": sp.read_text(encoding="utf-8")[:800],
            "mutable_policies": _policy_stems(d / "policies" / "mutable"),
            "sealed_policies": _policy_stems(d / "policies" / "sealed"),
        })
    return out
