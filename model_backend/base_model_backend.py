"""
BaseModelBackend: shared scaffolding for ModelBackend implementations.

Lifts duplicated helpers (message sanitization, tool-name sanitization & caching,
JSON-schema strict-object enforcement, usage normalization, logger/cache init,
max_context_length property) so concrete backends only own provider-specific
wire-call logic.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional
import hashlib
import json
import re

from .model_backend import ModelBackend
import logging


class BaseModelBackend(ModelBackend):
    """
    Concrete-but-still-abstract base for ModelBackend implementations.

    Subclasses MUST implement:
        - from_spec(spec, worker_context) -> instance
        - generate(messages, tools=None, **kwargs) -> dict

    Subclasses MAY override:
        - _ALLOWED_MESSAGE_KEYS  (e.g. vLLM adds reasoning fields to assistant role)
        - _LOG_PREFIX            (e.g. "[vLLM]", "[OpenAI]") for log readability
        - _sanitize_tool_name(name) (default: replace [^a-zA-Z0-9_-] with '_')
        - _normalize_usage(usage)   (default handles both prompt/completion_tokens
                                     and input/output_tokens shapes)
    """

    # AFW canonical message schema. Every backend transforms IN (canonical ->
    # provider-specific) via ``_convert_messages`` and OUT (provider response ->
    # canonical) via the result dict from ``generate``.
    #
    # ``reasoning`` is first-class on assistant turns: backends that natively
    # carry reasoning preserve it; backends whose API/template can't ingest it
    # drop or transform it in their input layer (vLLM merges into content
    # using PLM's [my prior thinking]/[my reply] markers; OpenAI and Slate
    # drop it, since neither accepts external reasoning chunks on prior turns).
    _ALLOWED_MESSAGE_KEYS: Dict[str, set] = {
        "system": {"role", "content", "name"},
        "user": {"role", "content", "name"},
        "assistant": {"role", "content", "reasoning", "tool_calls", "name"},
        "tool": {"role", "content", "tool_call_id"},
    }

    # Log prefix used by tool-sanitization log lines. Override per subclass.
    _LOG_PREFIX: str = ""

    def __init__(self, model: str, max_context_length: int):
        """
        Initialise common fields. Subclasses must call super().__init__(...)
        before creating their HTTP client / using self.logger.
        """
        self.model = model
        self._max_context_length = max_context_length
        self.logger = logging.getLogger("plm.model_backend")

        # Tool sanitization cache state
        self._sanitized_tools_cache: Optional[List[Dict]] = None
        self._tool_name_mapping_cache: Dict[str, str] = {}
        self._original_tools_signature: Optional[str] = None

    @property
    def max_context_length(self) -> int:
        return self._max_context_length

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        ...

    # -------------------------------------------------------------------------
    # Message sanitization
    # -------------------------------------------------------------------------

    def _sanitize_messages_for_api(self, messages: List[Dict]) -> List[Dict]:
        """Return a new list of message dicts with only API-allowed keys; strips internal fields.

        Messages with role ``"verifier"`` are DROPPED entirely: that role is a PRIVATE
        trajectory channel — a verifier leaves notes there and reads them back off the
        trajectory, but they are INVISIBLE to the model (and the provider APIs don't
        recognize the role), so they must never reach the API."""
        out = []
        for msg in messages:
            role = msg.get("role")
            if role == "verifier":          # private verifier channel — never sent to the model
                continue
            allowed = self._ALLOWED_MESSAGE_KEYS.get(role, {"role", "content"})
            # Drop None-valued OPTIONAL keys (tool_calls/reasoning/name) so a `tool_calls: None`
            # or `reasoning: None` doesn't bloat the wire or trip a strict provider validator;
            # keep role + content even if None (content: None is valid on a tool-call-only
            # assistant turn).
            out.append({k: v for k, v in msg.items()
                        if k in allowed and (v is not None or k in ("role", "content"))})
        return out

    # -------------------------------------------------------------------------
    # Tool sanitization helpers
    # -------------------------------------------------------------------------

    def _sanitize_tool_name(self, name: str) -> str:
        """Replace characters that providers (OpenAI/vLLM/Anthropic) reject in tool names."""
        return re.sub(r"[^a-zA-Z0-9_-]", "_", name)

    def _get_tools_signature(self, tools: List[Dict]) -> str:
        """Hash signature for the tool list to detect changes between calls."""
        tool_names = sorted(
            [t.get("function", {}).get("name", "") for t in tools if t.get("type") == "function"]
        )
        return hashlib.md5(json.dumps(tool_names, sort_keys=True).encode()).hexdigest()

    def _sanitize_and_cache_tools(
        self, tools: Optional[List[Dict]]
    ) -> tuple[Optional[List[Dict]], Dict[str, str]]:
        """
        Sanitize tool names (replace invalid chars with underscores) and cache the
        result so repeated calls with the same tool list don't re-sanitize.
        Returns (sanitized_tools, tool_name_mapping: sanitized -> original).
        """
        if not tools:
            return None, {}

        current_signature = self._get_tools_signature(tools)

        if (
            self._original_tools_signature == current_signature
            and self._sanitized_tools_cache is not None
        ):
            self.logger.debug(f"{self._LOG_PREFIX} Using cached sanitized tools")
            return self._sanitized_tools_cache, self._tool_name_mapping_cache.copy()

        self.logger.info(f"{self._LOG_PREFIX} Sanitizing and caching {len(tools)} tools")
        sanitized_tools: List[Dict] = []
        tool_name_mapping: Dict[str, str] = {}

        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                original_name = tool["function"].get("name")
                if not original_name:
                    self.logger.warning(f"{self._LOG_PREFIX} Skipping tool with no name: {tool}")
                    continue
                sanitized_name = self._sanitize_tool_name(original_name)
                if original_name != sanitized_name:
                    self.logger.info(
                        f"{self._LOG_PREFIX} Sanitized: '{original_name}' -> '{sanitized_name}'"
                    )
                tool_name_mapping[sanitized_name] = original_name
                sanitized_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": sanitized_name,
                            "description": tool["function"].get("description", ""),
                            "parameters": tool["function"].get("parameters", {}),
                        },
                    }
                )
            else:
                sanitized_tools.append(tool)

        self._sanitized_tools_cache = sanitized_tools
        self._tool_name_mapping_cache = tool_name_mapping
        self._original_tools_signature = current_signature

        self.logger.info(
            f"{self._LOG_PREFIX} Cached {len(sanitized_tools)} sanitized tools "
            f"with {len(tool_name_mapping)} mappings"
        )
        return sanitized_tools, tool_name_mapping

    def _sanitize_message_history(
        self, messages: List[Dict], tool_name_mapping: Dict[str, str]
    ) -> List[Dict]:
        """
        Sanitize tool-call names in prior assistant messages so they match the
        sanitized names we are sending in the tools list. Mutates tool_name_mapping
        defensively if it encounters a never-seen name.
        """
        sanitized_messages: List[Dict] = []
        for msg in messages:
            msg_copy = msg.copy()
            if msg.get("tool_calls"):
                sanitized_tool_calls = []
                for tc in msg["tool_calls"]:
                    if not isinstance(tc, dict):          # defense: drop a malformed (non-dict)
                        continue                          # tool_call instead of crashing on .copy()
                    tc_copy = tc.copy()
                    if tc_copy.get("function") and tc_copy["function"].get("name"):
                        original_name = tc_copy["function"]["name"]
                        sanitized_name = self._sanitize_tool_name(original_name)
                        if sanitized_name not in tool_name_mapping:
                            tool_name_mapping[sanitized_name] = original_name
                        tc_copy["function"] = tc_copy["function"].copy()
                        tc_copy["function"]["name"] = sanitized_name
                    sanitized_tool_calls.append(tc_copy)
                msg_copy["tool_calls"] = sanitized_tool_calls
            sanitized_messages.append(msg_copy)
        return sanitized_messages

    # -------------------------------------------------------------------------
    # Structured output helper
    # -------------------------------------------------------------------------

    def _schema_with_strict_object(self, schema: Dict[str, Any]) -> None:
        """Set additionalProperties: False on all object nodes in a JSON schema (in-place)."""
        if isinstance(schema, dict):
            if schema.get("type") == "object" or "properties" in schema:
                schema["additionalProperties"] = False
            for v in schema.values():
                self._schema_with_strict_object(v)
        elif isinstance(schema, list):
            for item in schema:
                self._schema_with_strict_object(item)

    # -------------------------------------------------------------------------
    # Usage normalization
    # -------------------------------------------------------------------------

    def _normalize_usage(self, usage: Any) -> Dict[str, Any]:
        """
        Always returns prompt_tokens, completion_tokens, total_tokens.

        Accepts both OpenAI/vLLM-shape usage objects (prompt_tokens / completion_tokens)
        and Anthropic-shape (input_tokens / output_tokens). Subclasses can override
        to return additional provider-specific fields (e.g. token detail breakdowns).
        """
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        prompt = getattr(usage, "prompt_tokens", None)
        if prompt is None:
            prompt = getattr(usage, "input_tokens", 0)
        completion = getattr(usage, "completion_tokens", None)
        if completion is None:
            completion = getattr(usage, "output_tokens", 0)
        prompt = prompt or 0
        completion = completion or 0

        total = getattr(usage, "total_tokens", None)
        if total is None:
            total = prompt + completion

        return {
            "prompt_tokens": int(prompt),
            "completion_tokens": int(completion),
            "total_tokens": int(total),
        }
