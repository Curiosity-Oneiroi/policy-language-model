"""plm.metaparams — static PLM meta-parameters.

For now the meta-parameters (currently just the system prompt) are authored by
hand and read from disk. Per MFH, this is the placeholder the optimizer (φ)
will eventually own: φ optimizes PLM's meta-parameters — primarily the system
prompt — to elicit good meta-reasoning. Until φ exists, the meta-params are the
static files in this directory.
"""

from __future__ import annotations

from pathlib import Path

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "systemprompt.md"


def get_system_prompt() -> str:
    """Return the static PLM meta-reasoner system prompt (reads systemprompt.md)."""
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


__all__ = ["get_system_prompt"]
