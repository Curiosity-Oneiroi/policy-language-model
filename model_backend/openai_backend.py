from .base_model_backend import BaseModelBackend
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Type, TypeVar
import json
import time
import os
import re


# Reasoning models (o1..o9, gpt-5) use Responses API params and strip unsupported sampling params
_REASONING_MODEL_PATTERN = re.compile(r"^(o[1-9]|gpt-5)")
_UNSUPPORTED_REASONING_PARAMS = {"temperature", "top_p", "presence_penalty", "frequency_penalty"}


def _normalize_model_name(model: Optional[str]) -> str:
    """Strip a `provider/` routing prefix (LiteLLM/OpenRouter/Azure: `openai/o3-mini`,
    `azure/gpt-5`) and lowercase, so reasoning-model detection is robust to casing + prefixes.
    Also unwrap an OpenAI fine-tune id `ft:<base-model>:<org>::<id>` to its BASE model, so a
    fine-tune of a reasoning model is still classified as reasoning (NB3-3)."""
    m = (model or "").split("/")[-1].lower()
    if m.startswith("ft:"):
        parts = m.split(":")
        m = parts[1] if len(parts) > 1 else m
    return m


def _is_reasoning_model_name(model: Optional[str]) -> bool:
    """The SINGLE reasoning-model classifier — used by BOTH the `use_responses_api` auto-detect
    AND the `_is_reasoning_model` property, so they can never disagree (a mismatch sends an
    o-series/gpt-5 model down the wrong API path -> `temperature` 400, or a missing `reasoning`
    dict). Covers o1..o9 + gpt-5, regardless of casing or a `provider/` prefix."""
    return bool(_REASONING_MODEL_PATTERN.match(_normalize_model_name(model)))

# Kwargs we are willing to forward to the OpenAI HTTP APIs. Everything else
# the caller passes (event-logging plumbing, agent-core internals, custom
# trace metadata, etc.) is silently dropped before the request goes out.
#
# Listing what the API accepts is short and stable; listing what callers
# might forward is open-ended and brittle, so we prefer the allowlist.
# Backend-internal control fields (e.g. ``_tool_name_mapping``,
# ``include_reasoning_summary``) are read directly from kwargs where
# needed and intentionally absent from these allowlists.

# Common params accepted by both APIs.
_OPENAI_COMMON_KWARGS = frozenset({
    "temperature", "top_p", "n", "stop", "stream", "stream_options",
    "tools", "tool_choice", "parallel_tool_calls",
    "response_format", "seed", "service_tier", "user", "metadata",
    "extra_body", "extra_headers", "extra_query", "timeout",
})

# Extra params accepted only by /v1/responses.
_RESPONSES_API_KWARGS = _OPENAI_COMMON_KWARGS | frozenset({
    "store", "include", "previous_response_id", "truncation",
    "text",          # structured output on this API (the `response_format` analogue)
})

# Extra params accepted only by /v1/chat/completions.
_CHAT_COMPLETIONS_KWARGS = _OPENAI_COMMON_KWARGS | frozenset({
    "presence_penalty", "frequency_penalty", "logit_bias",
    "logprobs", "top_logprobs",
})

T = TypeVar("T", bound=BaseModel)


def _responses_api_text_chunks(node: Any) -> List[str]:
    """
    Flatten Responses API output blocks (message.content, reasoning.summary, etc.)
    into strings safe for str.join. The API may nest lists or put list-valued .text.
    """
    if node is None:
        return []
    if isinstance(node, str):
        return [node] if node else []
    if isinstance(node, (list, tuple)):
        out: List[str] = []
        for x in node:
            out.extend(_responses_api_text_chunks(x))
        return out
    if isinstance(node, dict):
        if "text" in node:
            return _responses_api_text_chunks(node["text"])
        if "content" in node:
            return _responses_api_text_chunks(node["content"])
        return []
    text = getattr(node, "text", None)
    if text is not None:
        return _responses_api_text_chunks(text)
    value = getattr(node, "value", None)
    if value is not None:
        return _responses_api_text_chunks(value)
    inner = getattr(node, "content", None)
    if inner is not None and inner is not node:
        return _responses_api_text_chunks(inner)
    return []


def _usage_value_to_json_safe(value: Any) -> Any:
    """Recursively turn SDK / Pydantic usage sub-objects into JSON-friendly data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _usage_value_to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_usage_value_to_json_safe(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="python")
        except TypeError:
            dumped = model_dump()
        return _usage_value_to_json_safe(dumped)
    return str(value)


def expand_openai_usage(usage: Any) -> Dict[str, Any]:
    """
    Build a usage dict for callers: always includes prompt_tokens, completion_tokens,
    total_tokens, plus any extra fields the SDK returns (e.g. token details).
    """
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    raw: Dict[str, Any] = {}
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        try:
            raw = dict(model_dump(mode="python"))
        except TypeError:
            raw = dict(model_dump())
    else:
        for key in (
            "input_tokens",
            "output_tokens",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens_details",
            "output_tokens_details",
        ):
            if hasattr(usage, key):
                try:
                    raw[key] = getattr(usage, key)
                except Exception:
                    pass

    for k, v in list(raw.items()):
        raw[k] = _usage_value_to_json_safe(v)

    prompt = raw.get("prompt_tokens")
    if prompt is None:
        prompt = raw.get("input_tokens", 0)
    completion = raw.get("completion_tokens")
    if completion is None:
        completion = raw.get("output_tokens", 0)
    total = raw.get("total_tokens")
    if total is None:
        try:
            total = int(prompt) + int(completion)
        except (TypeError, ValueError):
            total = 0

    raw["prompt_tokens"] = int(prompt) if isinstance(prompt, (int, float)) else 0
    raw["completion_tokens"] = int(completion) if isinstance(completion, (int, float)) else 0
    raw["total_tokens"] = int(total) if isinstance(total, (int, float)) else raw["prompt_tokens"] + raw["completion_tokens"]
    return raw


class OpenAIBackend(BaseModelBackend):
    """OpenAI API backend with support for both Chat Completions and Responses API"""

    _LOG_PREFIX = "[OpenAI]"

    # OpenAI's APIs don't accept ``reasoning`` on prior assistant turns
    # (Responses generates its own reasoning server-side; Chat Completions has
    # no reasoning slot). Override the canonical schema to drop it during
    # _sanitize_messages_for_api so it never reaches the wire.
    _ALLOWED_MESSAGE_KEYS = {
        "system": {"role", "content", "name"},
        "user": {"role", "content", "name"},
        "assistant": {"role", "content", "tool_calls", "name"},
        "tool": {"role", "content", "tool_call_id"},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_context_length: int = 150_000,
        use_responses_api: Optional[bool] = None,
        base_url: Optional[str] = None,
    ):
        """
        Args:
            api_key: OpenAI API key (or from OPENAI_API_KEY env var)
            model: Model name
            max_context_length: Maximum context length (default: 150k conservative)
            use_responses_api: Whether to use Responses API (auto-detected for reasoning models if None)
            base_url: Optional base URL for the API (e.g. proxy or custom endpoint)
        """
        super().__init__(model=model, max_context_length=max_context_length)

        client_kwargs: Dict[str, Any] = {}
        api_key_value = api_key or os.getenv("OPENAI_API_KEY")
        if api_key_value:
            client_kwargs["api_key"] = api_key_value
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client_kwargs = client_kwargs
        self._client: Optional[AsyncOpenAI] = None
        # Expose as plain attrs so PLM's spec builder (a hasattr loop) round-trips
        # them into the kernel sub-LLM's `_PLM_BACKEND_SPEC`. Without this, an
        # in-cell llm/react_auto rebuilt via from_spec loses a custom
        # api_key/base_url and silently falls back to OPENAI_API_KEY + the default
        # api.openai.com endpoint. Mirrors Slate/VLLM which already do this.
        self.api_key = api_key_value
        self.base_url = base_url

        # Auto-detect Responses API from the model name via the SHARED classifier (so it can
        # never disagree with `_is_reasoning_model`, and correctly covers o2/o4/o5+ + prefixes).
        if use_responses_api is None:
            self.use_responses_api = _is_reasoning_model_name(model)
        else:
            self.use_responses_api = use_responses_api

        api_type = "Responses" if self.use_responses_api else "Chat Completions"
        self.logger.info(f"OpenAIBackend: Model={model}, API={api_type}, max_context={self._max_context_length:,}")

    @classmethod
    def from_spec(cls, spec: Dict[str, Any], worker_context: Any) -> "OpenAIBackend":
        """Build backend from spec. worker_context provides run_store for hash resolution."""
        return cls(
            api_key=spec.get("api_key"),
            model=spec.get("model", "gpt-4o"),
            max_context_length=spec.get("max_context_length", 150_000),
            use_responses_api=spec.get("use_responses_api"),
            base_url=spec.get("base_url"),
        )
    
    @property
    def client(self) -> AsyncOpenAI:
        """Lazy-initialized OpenAI client (created on first use)."""
        if self._client is None:
            self._client = AsyncOpenAI(**self._client_kwargs)
        return self._client

    @property
    def _is_reasoning_model(self) -> bool:
        """True if the model is a reasoning model (o1..o9, gpt-5) with special param
        requirements. Shares `_is_reasoning_model_name` with the `use_responses_api`
        auto-detect, so the two can NEVER disagree."""
        return _is_reasoning_model_name(self.model)
    
    def _prepare_responses_api_kwargs(self, kwargs: Dict[str, Any], default_tokens: int = 4096) -> Dict[str, Any]:
        """Build kwargs for the Responses API by allowlisting known-good params.

        Token budget and reasoning controls are handled explicitly (Responses
        uses ``max_output_tokens`` plus a ``reasoning`` dict for o-series and
        gpt-5 models, neither of which accept ``temperature`` / ``top_p`` etc.).
        Everything else is filtered against ``_RESPONSES_API_KWARGS`` so
        agent-core plumbing, event-logger handles, and other non-API kwargs
        get silently dropped before the HTTP call.
        """
        tokens = (
            kwargs.get("max_output_tokens")
            or kwargs.get("max_tokens")
            or kwargs.get("max_completion_tokens")
            or default_tokens
        )
        allowed = _RESPONSES_API_KWARGS
        if self._is_reasoning_model:
            allowed = allowed - _UNSUPPORTED_REASONING_PARAMS
            reasoning_effort = kwargs.get("reasoning_effort", "medium")
            reasoning = kwargs.get("reasoning", {"effort": reasoning_effort})
            out: Dict[str, Any] = {
                "max_output_tokens": tokens,
                "reasoning": reasoning,
            }
        else:
            out = {"max_output_tokens": max(16, tokens)}

        for k in allowed:
            if k in kwargs:
                out[k] = kwargs[k]
        return out
    
    def _prepare_responses_api_payload(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Build instructions + input for the Responses API.

        Accepts the canonical AFW (OpenAI-flat) message shape, including assistant
        messages with structured ``tool_calls`` and ``role='tool'`` messages with
        ``tool_call_id``. Converts internally to the Responses API's native input
        items:

        * ``role='system'`` content -> concatenated into ``instructions``.
        * ``role='assistant'`` with ``tool_calls`` -> a text message (if content
          is non-empty) plus one ``{"type":"function_call",...}`` item per call.
        * ``role='tool'`` -> ``{"type":"function_call_output",...}`` item.
        * Everything else -> ``{"role": role, "content": content}`` message item.

        Why this lives in the backend: keeping it here means callers can pass
        canonical messages with structured tool surfaces and the backend handles
        per-wire-format translation internally — no per-backend pre-conversion
        shims at the call site.
        """
        instructions_parts: List[str] = []
        input_items: List[Dict[str, Any]] = []

        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            tool_calls = m.get("tool_calls")
            tool_call_id = m.get("tool_call_id")

            if role == "system":
                if isinstance(content, str):
                    instructions_parts.append(content)
                else:
                    instructions_parts.append(json.dumps(content) if content is not None else "")
                continue

            if role == "tool":
                if not isinstance(content, str):
                    content = json.dumps(content) if content is not None else ""
                input_items.append({
                    "type": "function_call_output",
                    "call_id": tool_call_id or "",
                    "output": content,
                })
                continue

            if role == "assistant" and tool_calls:
                if content:
                    input_items.append({"role": "assistant", "content": content})
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments", "{}")
                    if not isinstance(args, str):
                        args = json.dumps(args)
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "arguments": args,
                    })
                continue

            input_items.append({"role": role, "content": content})

        payload = {
            "instructions": "\n\n".join(instructions_parts).strip(),
            "input": input_items,
        }

        # Responses API requires at least one non-system item in the input.
        if not payload["input"]:
            raise ValueError(
                "Responses API requires at least one non-system message in input "
                "(only system messages were provided)."
            )

        return payload
    
    async def generate(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send request to OpenAI API (Chat Completions or Responses).

        Returns a dict including:
            - content: str — final assistant-visible message text (no reasoning prefix).
            - tool_calls, usage, model
            - reasoning_summaries: optional list of str from Responses API ``reasoning`` items
              (None if absent or Chat Completions path).
            - response_output_types: optional list of output item type strings (Responses only).
            - content_with_reasoning: present only if ``include_reasoning_summary=True`` and
              summaries exist — legacy merged string for logging or display.
            - reasoning_effort: on reasoning models (Responses path).
        """
        messages = self._sanitize_messages_for_api(messages)

        api_type = "Responses API" if self.use_responses_api else "Chat Completions API"
        self.logger.info(f"[OpenAI] Using {api_type} for model {self.model}")

        if self.use_responses_api:
            # `response_format` is Chat-Completions vocabulary; the Responses API
            # expresses the same thing as `text.format`. Callers (e.g. a policy
            # attaching a Constraint) legitimately pass the former without
            # knowing which API a given model routes to, so translate rather than
            # forwarding an argument the SDK will reject.
            _rf = kwargs.pop("response_format", None)
            if isinstance(_rf, dict) and _rf.get("type") == "json_schema" and "text" not in kwargs:
                _js = _rf.get("json_schema") or {}
                kwargs["text"] = {"format": {"type": "json_schema",
                                             "name": _js.get("name", "answer"),
                                             "strict": _js.get("strict", False),
                                             "schema": _js.get("schema", {})}}
            return await self._generate_with_responses_api(messages, tools, **kwargs)
        else:
            return await self._generate_with_completions_api(messages, tools, **kwargs)
    
    def _normalize_usage(self, usage: Any) -> Dict[str, Any]:
        """
        OpenAI override: returns the standard 3 fields PLUS any extras the SDK
        provides (e.g. input_tokens_details, output_tokens_details for reasoning
        models). Base impl only returns the 3 floor fields.
        """
        return expand_openai_usage(usage)
    
    async def _generate_with_responses_api(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Use Responses API (full instructions + input format, reasoning param handling)."""
        sanitized_tools, tool_name_mapping = self._sanitize_and_cache_tools(tools)
        # Prior turns' tool_calls are validated by OpenAI even when THIS turn has no tools, so
        # sanitize history UNCONDITIONALLY — consistent with the Chat path.
        messages = self._sanitize_message_history(messages, tool_name_mapping)
        if sanitized_tools:
            kwargs["_tool_name_mapping"] = tool_name_mapping

        payload = self._prepare_responses_api_payload(messages)
        prepared_kwargs = self._prepare_responses_api_kwargs(kwargs)
        
        request_params: Dict[str, Any] = {
            "model": self.model,
            **payload,
            **prepared_kwargs,
        }
        if sanitized_tools:
            request_params["tools"] = sanitized_tools
            request_params["tool_choice"] = kwargs.get("tool_choice", "auto")
        
        # Only add store when explicitly set to False (e.g. for stateless / zero retention)
        if "store" in kwargs and kwargs["store"] is False:
            request_params["store"] = False
        # include: e.g. ["reasoning.encrypted_content"] for multi-turn reasoning
        if "include" in kwargs:
            request_params["include"] = kwargs["include"]
        
        reasoning_effort = prepared_kwargs.get("reasoning", {}).get("effort", "medium") if self._is_reasoning_model else kwargs.get("reasoning_effort", "medium")
        if reasoning_effort:
            self.logger.debug(f"Using reasoning_effort: {reasoning_effort}")
        
        try:
            call_start = time.time()
            response = await self.client.responses.create(**request_params)
            call_duration = time.time() - call_start
            
            output_items = response.output if hasattr(response, "output") else []
            content_parts = []
            reasoning_content = []
            tool_calls = []
            output_item_types: List[str] = []
            
            for item in output_items:
                item_type = getattr(item, "type", None)
                if isinstance(item_type, str):
                    output_item_types.append(item_type)
                if item_type == "message":
                    if hasattr(item, "content"):
                        content_parts.extend(_responses_api_text_chunks(item.content))
                elif item_type == "reasoning":
                    if hasattr(item, "summary"):
                        reasoning_content.extend(_responses_api_text_chunks(item.summary))
                elif item_type in ("function_call", "tool_call"):
                    sanitized_name = getattr(item, "name", "")
                    mapping = kwargs.get("_tool_name_mapping", {})
                    original_name = mapping.get(sanitized_name, sanitized_name)
                    call_id = getattr(item, "call_id", None) or getattr(item, "id", "")
                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": original_name,
                            "arguments": getattr(item, "arguments", "{}"),
                        },
                    })
            
            message_content = "\n".join(content_parts) if content_parts else ""
            reasoning_flat: Optional[str] = (
                "\n".join(s for s in reasoning_content if s) if reasoning_content else None
            )

            usage = self._normalize_usage(getattr(response, "usage", None))
            return {
                "content": message_content,
                "reasoning": reasoning_flat,
                "tool_calls": tool_calls if tool_calls else None,
                "usage": usage,
                "model": getattr(response, "model", self.model) or self.model,
                "backend": "openai",
                "duration": call_duration,
                "finish_reason": getattr(response, "status", None),
                "stop_reason": None,
            }

        except Exception:
            # No silent fallback to Chat Completions — Responses errors surface
            # to the caller (model selection is intentional; fallback masks
            # real wire-level issues like 400/429/etc).
            self.logger.exception("Responses API error")
            raise
    
    async def _generate_with_completions_api(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Use traditional Chat Completions API"""
        
        # Use cached sanitized tools (much more efficient than re-sanitizing every call)
        sanitized_tools, tool_name_mapping = self._sanitize_and_cache_tools(tools)
        
        # Sanitize message history using the cached mapping
        # CRITICAL: Tool calls from previous turns are in the messages, and OpenAI validates them too!
        messages = self._sanitize_message_history(messages, tool_name_mapping)
        
        # Store mapping for response parsing
        if sanitized_tools:
            tools = sanitized_tools
            kwargs["_tool_name_mapping"] = tool_name_mapping
        
        completion_kwargs = {
            "model": self.model,
            "messages": messages,
        }
        
        # Token budget: reasoning models use max_completion_tokens; others
        # use max_tokens. Accept either input key from the caller.
        tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 16384
        if self._is_reasoning_model:
            completion_kwargs["temperature"] = 1
            completion_kwargs["max_completion_tokens"] = tokens
            allowed = _CHAT_COMPLETIONS_KWARGS - _UNSUPPORTED_REASONING_PARAMS - {"temperature"}
        else:
            completion_kwargs["temperature"] = kwargs.get("temperature", 0.7)
            completion_kwargs["max_tokens"] = tokens
            allowed = _CHAT_COMPLETIONS_KWARGS - {"temperature"}  # already set

        # Allowlist passthrough for the rest. Anything not in this set
        # (agent-core plumbing, event-logger handles, etc.) is dropped.
        for k in allowed:
            if k in kwargs:
                completion_kwargs[k] = kwargs[k]

        if tools:
            completion_kwargs["tools"] = tools
            completion_kwargs.setdefault("tool_choice", kwargs.get("tool_choice", "auto"))
        
        call_start = time.time()
        response = await self.client.chat.completions.create(**completion_kwargs)
        call_duration = time.time() - call_start
        
        if not response.choices:
            raise ValueError("Model returned no choices for Chat Completions request.")
        choice = response.choices[0]
        
        # Extract tool calls and restore original names
        tool_calls = []
        if choice.message.tool_calls:
            tool_name_mapping = kwargs.get("_tool_name_mapping", {})
            for tc in choice.message.tool_calls:
                sanitized_name = tc.function.name if tc.function.name else ""
                
                if not sanitized_name:
                    self.logger.warning(f"[Chat Completions API] Skipping tool call with no name: {tc}")
                    continue
                
                original_name = tool_name_mapping.get(sanitized_name, sanitized_name)
                
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": original_name,  # Use original name
                        "arguments": tc.function.arguments
                    }
                })
        
        return {
            "content": choice.message.content or "",
            "reasoning": None,
            "tool_calls": tool_calls if tool_calls else None,
            "usage": self._normalize_usage(response.usage),
            "model": response.model,
            "backend": "openai",
            "duration": call_duration,
            "finish_reason": choice.finish_reason,
            "stop_reason": None,
        }
    
    async def generate_structured(
        self,
        messages: List[Dict],
        response_model: Type[T],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response that conforms to a Pydantic model. Returns the same shape as generate
        with content = parsed model instance. OpenAIBackend-only (not on ModelBackend base).
        """
        messages = self._sanitize_messages_for_api(messages)
        schema = response_model.model_json_schema()
        self._schema_with_strict_object(schema)
        
        if self.use_responses_api:
            return await self._generate_structured_responses(messages, response_model, schema, tools, tool_choice, **kwargs)
        return await self._generate_structured_completions(messages, response_model, schema, tools, tool_choice, **kwargs)
    
    async def _generate_structured_responses(
        self,
        messages: List[Dict],
        response_model: Type[T],
        schema: Dict[str, Any],
        tools: Optional[List[Dict]],
        tool_choice: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Structured output via Responses API (text.format json_schema)."""
        try:
            sanitized_tools, tool_name_mapping = self._sanitize_and_cache_tools(tools)
            messages = self._sanitize_message_history(messages, tool_name_mapping)   # always
            if sanitized_tools:
                kwargs["_tool_name_mapping"] = tool_name_mapping
            payload = self._prepare_responses_api_payload(messages)
            prepared_kwargs = self._prepare_responses_api_kwargs(kwargs)
            request_params: Dict[str, Any] = {
                "model": self.model,
                **payload,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": schema,
                    }
                },
                **prepared_kwargs,
            }
            if sanitized_tools:
                request_params["tools"] = sanitized_tools
                request_params["tool_choice"] = tool_choice or "auto"
            if "store" in kwargs and kwargs["store"] is False:
                request_params["store"] = False
            if "include" in kwargs:
                request_params["include"] = kwargs["include"]
            
            response = await self.client.responses.create(**request_params)
            raw_output = getattr(response, "output_text", None)
            if not raw_output and hasattr(response, "output"):
                for item in response.output:
                    if getattr(item, "type", "") == "message" and hasattr(item, "content"):
                        for c in item.content if isinstance(item.content, list) else []:
                            if getattr(c, "type", "") == "output_refusal":
                                raise ValueError(f"Model refused to respond: {getattr(c, 'refusal', 'unknown')}")
            if not raw_output:
                raise ValueError(f"Failed to parse response into {response_model.__name__}, output was empty.")
            parsed = response_model.model_validate_json(raw_output)
            return {
                "content": parsed,
                "tool_calls": None,
                "usage": self._normalize_usage(getattr(response, "usage", None)),
                "model": self.model,
            }
        except Exception:
            # No silent fallback to Chat Completions for structured output either.
            self.logger.exception("Structured Responses API error")
            raise
    
    async def _generate_structured_completions(
        self,
        messages: List[Dict],
        response_model: Type[T],
        schema: Dict[str, Any],
        tools: Optional[List[Dict]],
        tool_choice: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Structured output via Chat Completions API (response_format json_schema)."""
        sanitized_tools, tool_name_mapping = self._sanitize_and_cache_tools(tools)
        messages = self._sanitize_message_history(messages, tool_name_mapping)
        if sanitized_tools:
            kwargs["_tool_name_mapping"] = tool_name_mapping
            tools = sanitized_tools
        completion_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 16384
        if self._is_reasoning_model:
            completion_kwargs["temperature"] = 1
            completion_kwargs["max_completion_tokens"] = tokens
            allowed = _CHAT_COMPLETIONS_KWARGS - _UNSUPPORTED_REASONING_PARAMS - {"temperature", "response_format"}
        else:
            completion_kwargs["temperature"] = kwargs.get("temperature", 0.7)
            completion_kwargs["max_tokens"] = tokens
            allowed = _CHAT_COMPLETIONS_KWARGS - {"temperature", "response_format"}
        for k in allowed:
            if k in kwargs:
                completion_kwargs[k] = kwargs[k]
        if tools:
            completion_kwargs["tools"] = tools
            completion_kwargs.setdefault("tool_choice", tool_choice or "auto")
        
        response = await self.client.chat.completions.create(**completion_kwargs)
        if not response.choices:
            raise ValueError(
                f"Model returned no choices for structured completion into {response_model.__name__}."
            )
        choice = response.choices[0]
        raw = choice.message.content
        if not raw:
            raise ValueError(f"Failed to parse response into {response_model.__name__}, output was empty.")
        parsed = response_model.model_validate_json(raw)
        return {
            "content": parsed,
            "tool_calls": None,
            "usage": self._normalize_usage(response.usage),
            "model": response.model,
        }



