"""Fireworks AI backend — OpenAI-compatible Chat Completions with native reasoning.

Fireworks serves models (e.g. Qwen 3.7 Plus) over the standard OpenAI Chat Completions
API at https://api.fireworks.ai/inference/v1, with two things PLM cares about:

  * **Tool calling** — standard OpenAI `tools` (nested `{"type":"function","function":{…}}`)
    and `tool_calls` in the response, `finish_reason="tool_calls"`. We emit the nested
    shape via the base `_as_chat_completions_tools`.
  * **Reasoning** — a SEPARATE `reasoning_content` field on the response message (no
    `<think>` tags to strip). Thinking is OFF by default on the model; we turn it on with
    `reasoning_effort` (default "medium"). Prior-turn reasoning is DROPPED on the way back
    (like the OpenAI/Slate backends) — `_ALLOWED_MESSAGE_KEYS` excludes `reasoning` — so we
    don't send a field the chat API doesn't model. (Cross-turn reasoning preservation via
    `reasoning_content` + `reasoning_history="preserved"` is a future enhancement.)

Key comes from `$FIREWORKS_API_KEY` (see experiments/config.py).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from .base_model_backend import BaseModelBackend

T = TypeVar("T", bound=BaseModel)

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "accounts/fireworks/models/qwen3p7-plus"
DEFAULT_CONTEXT = 262_144            # Qwen 3.7 Plus on Fireworks
DEFAULT_REASONING_EFFORT = "medium"  # thinking is OFF by default on the model; turn it on
# Fireworks `reasoning_effort` accepts these strings (plus an int token budget).
_VALID_EFFORT = {"none", "low", "medium", "high", "xhigh", "max", "adaptive"}


class FireworksBackend(BaseModelBackend):
    """Fireworks (OpenAI-compatible) backend with Qwen-style reasoning + tool calling."""

    _LOG_PREFIX = "[Fireworks]"

    # Drop prior-turn `reasoning` on the way out (the chat API has no input field for it;
    # it returns reasoning_content but doesn't ingest a `reasoning` key). Matches OpenAI/Slate.
    _ALLOWED_MESSAGE_KEYS = {
        "system": {"role", "content", "name"},
        "user": {"role", "content", "name"},
        "assistant": {"role", "content", "tool_calls", "name"},
        "tool": {"role", "content", "tool_call_id"},
    }

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        reasoning_effort: Optional[str] = DEFAULT_REASONING_EFFORT,
        max_context_override: Optional[int] = None,
    ):
        self.base_url = base_url
        # Key from arg, else env. Kept as a plain attr so PLM's spec builder round-trips
        # it to in-kernel sub-LLMs (the hasattr(api_key/base_url/...) loop).
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY") or "EMPTY"
        self.reasoning_effort = reasoning_effort
        super().__init__(model=model, max_context_length=max_context_override or DEFAULT_CONTEXT)
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)

    @classmethod
    def from_spec(cls, spec: Dict[str, Any], worker_context: Any) -> "FireworksBackend":
        return cls(
            base_url=spec.get("base_url", DEFAULT_BASE_URL),
            model=spec.get("model", DEFAULT_MODEL),
            api_key=spec.get("api_key"),
            reasoning_effort=spec.get("reasoning_effort", DEFAULT_REASONING_EFFORT),
            max_context_override=spec.get("max_context_override") or spec.get("max_context_length"),
        )

    # ---- helpers ----------------------------------------------------------- #
    def _effort(self, kwargs: Dict[str, Any]) -> Optional[Any]:
        """Resolve the reasoning effort for this call (per-call override wins)."""
        eff = kwargs.get("reasoning_effort", self.reasoning_effort)
        if eff in (None, "", False):
            return None
        if isinstance(eff, str) and eff not in _VALID_EFFORT:
            return self.reasoning_effort        # bad override -> fall back to default
        return eff

    @staticmethod
    def _read_reasoning(message: Any) -> Optional[str]:
        """Fireworks returns reasoning in `reasoning_content` (a non-standard field the
        OpenAI SDK stashes on the parsed message or in `model_extra`)."""
        r = getattr(message, "reasoning_content", None)
        if r is None:
            extra = getattr(message, "model_extra", None) or {}
            r = extra.get("reasoning_content")
        return r or None

    def _extract_tool_calls(self, message: Any, name_map: Dict[str, str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tc in (message.tool_calls or []):
            sanitized = tc.function.name or ""
            if not sanitized:
                continue
            out.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": name_map.get(sanitized, sanitized),  # restore the original name
                    "arguments": tc.function.arguments,
                },
            })
        return out

    def _budget_max_tokens(self, messages, sanitized_tools, kwargs) -> int:
        cpt = kwargs.get("cumulative_prompt_tokens")
        if cpt is not None:
            est_in = cpt
        else:
            blob = json.dumps(messages) + (json.dumps(sanitized_tools) if sanitized_tools else "")
            est_in = len(blob) // 4
        margin = 2000
        available = self.max_context_length - est_in - margin
        requested = kwargs.get("max_tokens", 32768)
        if available <= 0:
            raise ValueError(
                f"input (~{est_in:,} tokens + {margin} margin) leaves no output room in the "
                f"{self.max_context_length:,}-token window")
        return max(1, min(requested, available))

    def _completion_kwargs(self, messages, sanitized_tools, kwargs, *, extra=None) -> Dict[str, Any]:
        ck: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.6),   # Qwen3 thinking-recommended
            "top_p": kwargs.get("top_p", 0.95),
            "max_tokens": self._budget_max_tokens(messages, sanitized_tools, kwargs),
        }
        extra_body: Dict[str, Any] = dict(extra or {})
        effort = self._effort(kwargs)
        if effort is not None:
            extra_body["reasoning_effort"] = effort          # turn thinking ON (default medium)
        if extra_body:
            ck["extra_body"] = extra_body
        if sanitized_tools:
            ck["tools"] = self._as_chat_completions_tools(sanitized_tools)   # nested OpenAI shape
            ck["tool_choice"] = kwargs.get("tool_choice", "auto")
        return ck

    # ---- core generate ----------------------------------------------------- #
    async def generate(self, messages: List[Dict], tools: Optional[List[Dict]] = None,
                       **kwargs) -> Dict[str, Any]:
        messages = self._sanitize_messages_for_api(messages)
        sanitized_tools, name_map = self._sanitize_and_cache_tools(tools)
        if sanitized_tools:
            messages = self._sanitize_message_history(messages, name_map)

        ck = self._completion_kwargs(messages, sanitized_tools, kwargs)
        t0 = time.time()
        response = await self.client.chat.completions.create(**ck)
        duration = time.time() - t0

        if not response.choices:
            raise ValueError(f"Fireworks returned no choices for model {self.model}.")
        choice = response.choices[0]
        msg = choice.message
        reasoning = self._read_reasoning(msg)
        tool_calls = self._extract_tool_calls(msg, name_map)
        return {
            "content": msg.content or "",
            "reasoning": reasoning,
            "reasoning_summaries": [reasoning] if reasoning else None,   # OpenAI-backend compat
            "tool_calls": tool_calls or None,
            "usage": self._normalize_usage(getattr(response, "usage", None)),
            "model": response.model,
            "backend": "fireworks",
            "duration": duration,
            "finish_reason": choice.finish_reason,
            "stop_reason": getattr(choice, "stop_reason", None),
            "response_output_types": None,
        }

    # ---- structured output ------------------------------------------------- #
    async def generate_structured(self, messages: List[Dict], response_model: Type[T],
                                  tools: Optional[List[Dict]] = None, tool_choice=None,
                                  **kwargs) -> Dict[str, Any]:
        """Constrain output to `response_model` via Fireworks' json_schema response_format.
        Returns the same shape as generate() with content = the parsed model instance."""
        messages = self._sanitize_messages_for_api(messages)
        sanitized_tools, name_map = self._sanitize_and_cache_tools(tools)
        if sanitized_tools:
            messages = self._sanitize_message_history(messages, name_map)

        schema = response_model.model_json_schema()
        rf = {"type": "json_schema",
              "json_schema": {"name": response_model.__name__, "schema": schema}}
        ck = self._completion_kwargs(messages, sanitized_tools, kwargs)
        ck["response_format"] = rf
        if tool_choice is not None and sanitized_tools:
            ck["tool_choice"] = tool_choice
        t0 = time.time()
        response = await self.client.chat.completions.create(**ck)
        duration = time.time() - t0
        choice = response.choices[0]
        msg = choice.message
        parsed = None
        try:
            parsed = response_model.model_validate_json(msg.content or "")
        except Exception:
            try:
                parsed = response_model.model_validate(json.loads(msg.content or "{}"))
            except Exception:
                parsed = None
        return {
            "content": parsed,
            "raw_content": msg.content or "",
            "reasoning": self._read_reasoning(msg),
            "tool_calls": self._extract_tool_calls(msg, name_map) or None,
            "usage": self._normalize_usage(getattr(response, "usage", None)),
            "model": response.model,
            "backend": "fireworks",
            "duration": duration,
            "finish_reason": choice.finish_reason,
        }
