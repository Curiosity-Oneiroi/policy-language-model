"""Backend selection for experiment runs, sourced from a `.env`.

API keys live in a `.env` (the app loads it); the experiment-create UI just picks a
backend + model. The actual `ModelBackend` is constructed in the run SUBPROCESS via
`build_backend` (it holds an async client, so it can't be pickled across processes).
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List, Optional

# name -> module / class / env credential key / default model / whether base_url applies
BACKENDS: Dict[str, Dict[str, Any]] = {
    "AnthropicBackend": {
        "module": "plm.model_backend.anthropic_backend", "cls": "AnthropicBackend",
        "env_key": "ANTHROPIC_API_KEY", "default_model": "claude-3-5-sonnet-20241022",
        "base_url": False,
    },
    "OpenAIBackend": {
        "module": "plm.model_backend.openai_backend", "cls": "OpenAIBackend",
        "env_key": "OPENAI_API_KEY", "default_model": "gpt-4o", "base_url": True,
    },
    "SlateBackend": {
        "module": "plm.model_backend.slate_backend", "cls": "SlateBackend",
        # opus-4.8 is the meta-reasoner of choice; SlateBackend auto-selects behavior_mode
        # "subagent" for opus (the agent modes are the only ones that serve it) and drives it
        # via the environmental REPL framing so it adopts the PLM persona. gpt-5.3-codex /
        # claude-haiku-4.5 fall back to "search". Endpoint fixed/`SLATE_URL`-env.
        "env_key": "SLATE_API_KEY", "default_model": "anthropic/claude-opus-4.8",
        "base_url": False,
    },
    "VLLMBackend": {
        "module": "plm.model_backend.vllm_backend", "cls": "VLLMBackend",
        "env_key": None, "default_model": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "base_url": True,
    },
    "FireworksBackend": {
        # Fireworks AI, OpenAI-compatible. Qwen 3.7 Plus: reasoning (reasoning_content,
        # default effort "medium") + tool calling, ~290 tok/s. Key from $FIREWORKS_API_KEY.
        "module": "plm.model_backend.fireworks_backend", "cls": "FireworksBackend",
        "env_key": "FIREWORKS_API_KEY",
        "default_model": "accounts/fireworks/models/qwen3p7-plus",
        "base_url": True,
    },
}


def load_env(dotenv_path: Optional[str] = None) -> None:
    """Load a `.env` into os.environ so backends pick up their API keys. No-op if
    python-dotenv is unavailable or the file is missing."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
    else:
        load_dotenv(override=False)


def available_backends(dotenv_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """List backends + whether their credential is present (for the create-experiment UI)."""
    load_env(dotenv_path)
    out: List[Dict[str, Any]] = []
    for name, b in BACKENDS.items():
        key = b["env_key"]
        out.append({
            "name": name,
            "env_key": key,
            "available": (key is None) or bool(os.environ.get(key)),
            "default_model": b["default_model"],
            "supports_base_url": b["base_url"],
        })
    return out


def build_backend(name: str, *, model: Optional[str] = None,
                  base_url: Optional[str] = None, dotenv_path: Optional[str] = None):
    """Construct a `ModelBackend` (in the run subprocess). The API key is read from the
    environment (loaded from `.env`); `model`/`base_url` come from the experiment config."""
    load_env(dotenv_path)
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; available: {sorted(BACKENDS)}")
    b = BACKENDS[name]
    cls = getattr(importlib.import_module(b["module"]), b["cls"])
    kwargs: Dict[str, Any] = {}
    if model:
        kwargs["model"] = model
    if base_url and b["base_url"]:
        kwargs["base_url"] = base_url
    return cls(**kwargs)
