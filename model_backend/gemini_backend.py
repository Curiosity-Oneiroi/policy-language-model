"""GeminiBackend — Gemini models through Google's OpenAI-compatible endpoint.

Thin subclass of OpenAIBackend: same wire protocol, tool-calling, and spec
round-trip; only the default base_url (and API-key env var) differ. Uses the
`openai` SDK — no new dependency.

Usage:
    GeminiBackend(model="gemini-3-flash", api_key=os.environ["GEMINI_API_KEY"])
"""

from __future__ import annotations

import os

from .openai_backend import OpenAIBackend

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiBackend(OpenAIBackend):
    def __init__(self, *, model: str, api_key: str | None = None,
                 base_url: str | None = None, **kwargs):
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("GEMINI_API_KEY", "EMPTY"),
            base_url=base_url or GEMINI_OPENAI_BASE_URL,
            **kwargs,
        )

    @classmethod
    def from_spec(cls, spec, _rm=None):
        return cls(
            model=spec.get("model"),
            api_key=spec.get("api_key"),
            base_url=spec.get("base_url") or GEMINI_OPENAI_BASE_URL,
        )
