from .base_model_backend import BaseModelBackend
from pydantic import BaseModel, ValidationError
from typing import Dict, Any, List, Optional, Type, TypeVar
from openai import AsyncOpenAI
import requests
import re
import time
import json



# vLLM's chat template doesn't reference `reasoning`, so the input transform
# merges prior-turn reasoning back into content. Inlined here (not in PLM)
# so this stays a vLLM-only concern; OpenAI/Slate drop reasoning instead.
def _merge_reasoning_into_content(reasoning: Optional[str], content: str) -> str:
    r, c = (reasoning or "").strip(), (content or "").strip()
    if not r:
        return c
    if not c:
        return f"[my prior thinking]\n{r}"
    return f"[my prior thinking]\n{r}\n\n[my reply]\n{c}"

# Internal/agent-only kwargs that should never be forwarded to the vLLM API
T = TypeVar("T", bound=BaseModel)


class StructuredOutputParseError(ValueError):
    """
    Raised when vLLM's response cannot be parsed into the requested Pydantic model.
    status_code=400 tells the ResourceManager classifier this is a client-side error
    so it is NOT retried (retries are for external resource failures, not parse failures).
    """
    status_code: int = 400


class VLLMBackend(BaseModelBackend):
    """
    vLLM backend using OpenAI-compatible endpoint.
    Supports tool-name sanitization, structured output, and reasoning extraction
    for Qwen3-series thinking models and any general model hosted on vLLM.
    """

    _LOG_PREFIX = "[vLLM]"

    # vLLM's input shape is OpenAI-flat. ``reasoning`` is on the canonical
    # assistant message but vLLM's chat template ignores it — the input
    # transform below merges reasoning into content before send.
    _ALLOWED_MESSAGE_KEYS = {
        "system": {"role", "content", "name"},
        "user": {"role", "content", "name"},
        "assistant": {"role", "content", "tool_calls", "name"},
        "tool": {"role", "content", "tool_call_id"},
    }

    def _sanitize_messages_for_api(self, messages: List[Dict]) -> List[Dict]:
        """vLLM input transform: merge any prior-turn ``reasoning`` field into
        ``content`` (since the chat template doesn't reference reasoning), then
        let the base sanitizer strip non-allowed keys."""
        merged: List[Dict] = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                merged.append({
                    **msg,
                    "content": _merge_reasoning_into_content(
                        msg.get("reasoning"), msg.get("content", "")
                    ),
                    "reasoning": None,
                })
            else:
                merged.append(msg)
        return super()._sanitize_messages_for_api(merged)

    def __init__(
        self,
        base_url: str = "http://0.0.0.0:8005/v1",
        model: str = "Qwen/Qwen3-30B-A3B-Thinking-2507",
        api_key: str = "EMPTY",
        max_context_override: Optional[int] = None
    ):
        """
        Args:
            base_url: vLLM server URL
            model: Model name
            api_key: API key (usually "EMPTY" for vLLM)
            max_context_override: Override max context length (if None, fetch from vLLM)
        """
        # base_url is referenced by _fetch_max_context_length below.
        self.base_url = base_url
        # Expose api_key as a plain attr so PLM's spec builder (a hasattr loop over
        # api_key/base_url/...) round-trips an authenticated vLLM key to the kernel
        # sub-LLM. added this to OpenAI/Anthropic/Slate but missed VLLM, so a
        # custom key was silently dropped on the in-kernel from_spec round-trip.
        self.api_key = api_key

        # Initialise base first so self.logger / self.model / tool cache state exist.
        # max_context_length is set to a placeholder; resolved immediately below.
        super().__init__(model=model, max_context_length=max_context_override or 0)

        if max_context_override is not None:
            self.logger.info(
                f"VLLMBackend: Using override max_context_length: {self._max_context_length:,}"
            )
        else:
            self._max_context_length = self._fetch_max_context_length()

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @classmethod
    def from_spec(cls, spec: Dict[str, Any], worker_context: Any) -> "VLLMBackend":
        """Build backend from spec. worker_context provides run_store for hash resolution."""
        base_url = spec.get("base_url", "http://0.0.0.0:8005/v1")
        model = spec.get("model", "Qwen/Qwen3-30B-A3B-Thinking-2507")
        max_context_override = spec.get("max_context_override")
        # PLM's spec builder (plm.py) round-trips the parent's RESOLVED context
        # under the key "max_context_length" — the same key OpenAI/Slate/Anthropic
        # from_spec all read. VLLM was the lone backend reading only the legacy
        # "max_context_override" key the builder never emits, so the parent's value
        # was silently dropped in-kernel and __init__ fell through to a BLOCKING
        # requests.get(base_url + "/models") probe on EVERY in-cell generate (the
        # worker_context cache is None on the kernel path). Fall back to the
        # round-tripped value so the round-trip is lossless and the probe is gone;
        # an explicit "max_context_override" still wins.
        if max_context_override is None:
            max_context_override = spec.get("max_context_length")

        # Consult the worker's shared-compute cache before triggering an HTTP
        # probe. Daemon prewarms ``model_max_context`` on submit, so by the
        # time we're here the value is usually present (either from the
        # static known-models seed delivered at spawn, or a fresh probe
        # broadcast just before this backend's instantiation).
        if max_context_override is None and worker_context is not None:
            getter = getattr(worker_context, "shared_cache_get", None)
            if callable(getter):
                cached = getter("model_max_context", (cls.__name__, base_url, model))
                if cached is None:
                    cached = getter("model_max_context", (cls.__name__, "*", model))
                if isinstance(cached, int) and cached > 0:
                    max_context_override = cached

        return cls(
            base_url=base_url,
            model=model,
            api_key=spec.get("api_key", "EMPTY"),
            max_context_override=max_context_override,
        )

    def _fetch_max_context_length(self) -> int:
        """
        Fetch max context length from vLLM server.

        vLLM's /v1/models endpoint returns model info including max_model_len.
        Falls back to 150k if unable to fetch.
        """
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)

            if response.status_code == 200:
                data = response.json()
                # vLLM returns: {"data": [{"id": "model_name", "max_model_len": 262144, ...}]}
                models = data.get("data", [])

                for model_info in models:
                    model_id = model_info.get("id", "")
                    # Match by exact name or if our model name is in the id
                    if model_id == self.model or self.model in model_id:
                        max_len = model_info.get("max_model_len")
                        if max_len:
                            self.logger.info(f"VLLMBackend: Fetched max_context_length from vLLM: {max_len:,}")
                            return max_len

                self.logger.warning(f"VLLMBackend: Model {self.model} not found in vLLM /models response")
        except Exception as e:
            self.logger.warning(f"VLLMBackend: Failed to fetch max_context_length from vLLM: {e}")

        # Default fallback
        self.logger.info("VLLMBackend: Using default max_context_length: 150,000")
        return 150000

    # -------------------------------------------------------------------------
    # Reasoning extraction
    # -------------------------------------------------------------------------

    def _extract_reasoning(self, message: Any) -> Optional[str]:
        """
        Extract reasoning / thinking content from a vLLM response message.
        Tries multiple field names used by different vLLM versions and model families:
          - message.reasoning          (primary, most vLLM builds)
          - message.model_extra        (older SDK versions that hide extra fields here)
          - message.reasoning_content  (some vLLM forks)
          - message.thinking           (some Qwen variants)
        """
        # Primary field
        reasoning = getattr(message, "reasoning", None)
        if reasoning:
            return reasoning

        # model_extra fallback (older openai SDK)
        model_extra = getattr(message, "model_extra", None)
        if model_extra and isinstance(model_extra, dict):
            for key in ("reasoning", "reasoning_content", "thinking"):
                val = model_extra.get(key)
                if val:
                    return val

        # Direct attribute fallbacks
        for attr in ("reasoning_content", "thinking"):
            val = getattr(message, attr, None)
            if val:
                return val

        return None

    @staticmethod
    def _parse_think_from_content(content: str) -> tuple[Optional[str], str]:
        """
        Fallback: parse <think>…</think> out of raw content when the vLLM server
        was not started with --reasoning-parser and think tokens land in content.

        Handles three cases:
          1. Well-formed:   <think>…</think>remaining
          2. Streaming artifact: content starts mid-think, ends with </think>remaining
          3. No think tags: returns (None, content) unchanged

        Returns (reasoning_or_None, cleaned_content).
        """
        if not content:
            return None, content

        # Case 1: well-formed <think>…</think>
        m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if m:
            reasoning = m.group(1).strip()
            # Strip ALL think blocks, not just the first, then clean up whitespace
            cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return reasoning or None, cleaned

        # Case 2: </think> present but no opening tag (streaming / partial capture)
        if "</think>" in content:
            parts = content.split("</think>", 1)
            reasoning = parts[0].replace("<think>", "").strip()
            cleaned = parts[1].strip()
            return reasoning or None, cleaned

        return None, content

    # -------------------------------------------------------------------------
    # Core generate
    # -------------------------------------------------------------------------

    async def generate(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call vLLM endpoint (OpenAI-compatible Chat Completions).

        Returns a dict including:
            - content: str — assistant response text
            - reasoning: str | None — thinking/reasoning content (Qwen3 etc.)
            - reasoning_summaries: list[str] | None — alias for reasoning wrapped in a list,
              for cross-backend compatibility with OpenAI backend consumers
            - tool_calls: list | None
            - usage: dict with prompt_tokens, completion_tokens, total_tokens
            - model, backend, duration, finish_reason, stop_reason
            - response_output_types: None (vLLM has no Responses API; field kept for compat)
        """
        messages = self._sanitize_messages_for_api(messages)

        # Sanitize tools and message history tool-call names
        sanitized_tools, tool_name_mapping = self._sanitize_and_cache_tools(tools)
        if sanitized_tools:
            messages = self._sanitize_message_history(messages, tool_name_mapping)

        # Token budget calculation
        cumulative_prompt_tokens = kwargs.get("cumulative_prompt_tokens", None)
        if cumulative_prompt_tokens is not None:
            estimated_input_tokens = cumulative_prompt_tokens
        else:
            input_text = json.dumps(messages)
            if sanitized_tools:
                input_text += json.dumps(sanitized_tools)
            estimated_input_tokens = len(input_text) // 4

        safety_margin = 2000
        available_tokens = self.max_context_length - estimated_input_tokens - safety_margin
        requested_max_tokens = kwargs.get("max_tokens", 32768)
        if available_tokens <= 0:                          # input fills/overflows the window
            raise ValueError(
                f"input (~{estimated_input_tokens:,} tokens + {safety_margin} safety margin) leaves "
                f"no output room in the {self.max_context_length:,}-token context window")
        actual_max_tokens = max(1, min(requested_max_tokens, available_tokens))   # cap at REAL available, no 1024 floor

        if actual_max_tokens < requested_max_tokens:
            token_source = "actual" if cumulative_prompt_tokens is not None else "estimated"
            self.logger.warning(
                f"Adjusted max_tokens from {requested_max_tokens} to {actual_max_tokens} "
                f"({token_source} input: {estimated_input_tokens:,} tokens, "
                f"available: {available_tokens:,} tokens)"
            )

        # enable_thinking kwarg (default True for thinking models, passable as False)
        enable_thinking = kwargs.get("enable_thinking", True)
        thinking_budget = kwargs.get("thinking_budget", None)
        chat_template_kwargs: Dict[str, Any] = {"enable_thinking": enable_thinking}
        if thinking_budget is not None:
            chat_template_kwargs["thinking_budget"] = thinking_budget

        completion_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 0.8),
            "max_tokens": actual_max_tokens,
            "extra_body": {
                "repetition_penalty": kwargs.get("repetition_penalty", 1.05),
                "chat_template_kwargs": kwargs.get("chat_template_kwargs", chat_template_kwargs),
            },
        }

        if sanitized_tools:
            completion_kwargs["tools"] = sanitized_tools
            completion_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        call_start = time.time()
        response = await self.client.chat.completions.create(**completion_kwargs)
        call_duration = time.time() - call_start

        if not response.choices:
            raise ValueError(f"vLLM returned no choices for model {self.model}.")
        choice = response.choices[0]

        if choice.finish_reason == "length":
            self.logger.warning(
                f"[vLLM] Response truncated (finish_reason='length'): "
                f"model={self.model}, actual_max_tokens={actual_max_tokens}"
            )

        # Extract tool calls and restore original names
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                sanitized_name = tc.function.name if tc.function.name else ""
                if not sanitized_name:
                    self.logger.warning(f"[vLLM] Skipping tool call with no name: {tc}")
                    continue
                original_name = tool_name_mapping.get(sanitized_name, sanitized_name)
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": original_name,
                        "arguments": tc.function.arguments
                    }
                })

        # Extract reasoning (multi-field fallback for Qwen3 / various vLLM builds)
        reasoning = self._extract_reasoning(choice.message)
        content = choice.message.content or ""

        # Always strip <think>…</think> from content — even when the server already
        # parsed reasoning into message.reasoning, stray think blocks can still appear
        # in content. Append any extracted stray reasoning to the native reasoning.
        if "<think>" in content or "</think>" in content:
            stray_reasoning, content = self._parse_think_from_content(content)
            if stray_reasoning:
                self.logger.debug(
                    "[vLLM] Stripped <think> block from content "
                    "(server missing --reasoning-parser or stray think tokens)"
                )
                reasoning = f"{reasoning}\n{stray_reasoning}".strip() if reasoning else stray_reasoning

        usage = self._normalize_usage(getattr(response, "usage", None))

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls if tool_calls else None,
            "usage": usage,
            "model": response.model,
            "backend": "vllm",
            "duration": call_duration,
            "finish_reason": choice.finish_reason,
            "stop_reason": getattr(choice, "stop_reason", None),
        }

    # -------------------------------------------------------------------------
    # Structured output
    # -------------------------------------------------------------------------

    async def generate_structured(
        self,
        messages: List[Dict],
        response_model: Type[T],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response that conforms to a Pydantic model via JSON schema mode.

        Returns the same shape as generate() with content = parsed model instance.
        Uses vLLM's response_format json_schema (OpenAI-compatible).
        """
        messages = self._sanitize_messages_for_api(messages)

        sanitized_tools, tool_name_mapping = self._sanitize_and_cache_tools(tools)
        if sanitized_tools:
            messages = self._sanitize_message_history(messages, tool_name_mapping)

        schema = response_model.model_json_schema()
        self._schema_with_strict_object(schema)

        # Token budget calculation (mirrors generate())
        cumulative_prompt_tokens = kwargs.get("cumulative_prompt_tokens", None)
        if cumulative_prompt_tokens is not None:
            estimated_input_tokens = cumulative_prompt_tokens
        else:
            input_text = json.dumps(messages) + json.dumps(schema)
            estimated_input_tokens = len(input_text) // 4

        safety_margin = 2000
        available_tokens = self.max_context_length - estimated_input_tokens - safety_margin
        requested_max_tokens = kwargs.get("max_tokens", 16384)
        if available_tokens <= 0:                          # input fills/overflows the window
            raise ValueError(
                f"input (~{estimated_input_tokens:,} tokens + {safety_margin} safety margin) leaves "
                f"no output room in the {self.max_context_length:,}-token context window")
        actual_max_tokens = max(1, min(requested_max_tokens, available_tokens))   # cap at REAL available, no 1024 floor

        if actual_max_tokens < requested_max_tokens:
            token_source = "actual" if cumulative_prompt_tokens is not None else "estimated"
            self.logger.warning(
                f"Adjusted max_tokens from {requested_max_tokens} to {actual_max_tokens} "
                f"({token_source} input: {estimated_input_tokens:,} tokens, "
                f"available: {available_tokens:,} tokens)"
            )

        enable_thinking = kwargs.get("enable_thinking", True)
        thinking_budget = kwargs.get("thinking_budget", None)
        chat_template_kwargs: Dict[str, Any] = {"enable_thinking": enable_thinking}
        if thinking_budget is not None:
            chat_template_kwargs["thinking_budget"] = thinking_budget

        completion_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 0.8),
            "max_tokens": actual_max_tokens,
            "extra_body": {
                "repetition_penalty": kwargs.get("repetition_penalty", 1.05),
                "chat_template_kwargs": kwargs.get("chat_template_kwargs", chat_template_kwargs),
            },
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
        }

        if sanitized_tools:
            completion_kwargs["tools"] = sanitized_tools
            completion_kwargs["tool_choice"] = tool_choice or "auto"

        call_start = time.time()
        response = await self.client.chat.completions.create(**completion_kwargs)
        call_duration = time.time() - call_start

        if not response.choices:
            raise ValueError(
                f"vLLM returned no choices for structured completion into {response_model.__name__}."
            )

        choice = response.choices[0]

        if choice.finish_reason == "length":
            self.logger.warning(
                f"[vLLM] Structured response truncated (finish_reason='length') "
                f"into {response_model.__name__}: model={self.model}, actual_max_tokens={actual_max_tokens}"
            )

        raw = choice.message.content
        if not raw:
            raise ValueError(
                f"Failed to parse response into {response_model.__name__}, output was empty."
            )

        # Fallback: if --reasoning-parser is not set on the server, think tokens land
        # in content before the JSON — extract and strip them so model_validate_json
        # receives clean JSON
        think_fallback: Optional[str] = None
        if "<think>" in raw or "</think>" in raw:
            think_fallback, raw = self._parse_think_from_content(raw)
            if think_fallback:
                self.logger.debug(
                    "[vLLM] Stripped <think> block from structured output content "
                    "(server missing --reasoning-parser)"
                )
            if not raw:
                raise ValueError(
                    f"Failed to parse response into {response_model.__name__}: "
                    f"content was empty after stripping <think> block."
                )

        try:
            parsed = response_model.model_validate_json(raw)
        except ValidationError as e:
            raise StructuredOutputParseError(
                f"Failed to parse vLLM response into {response_model.__name__}: {e}\n"
                f"finish_reason={choice.finish_reason!r}, raw={raw[:500]!r}"
            ) from e

        usage = self._normalize_usage(getattr(response, "usage", None))

        # Native reasoning field takes priority; fall back to think-tag extraction
        reasoning = self._extract_reasoning(choice.message) or think_fallback

        return {
            "content": parsed,
            "reasoning": reasoning,
            "tool_calls": None,
            "usage": usage,
            "model": response.model,
            "backend": "vllm",
            "duration": call_duration,
            "finish_reason": choice.finish_reason,
            "stop_reason": getattr(choice, "stop_reason", None),
        }



