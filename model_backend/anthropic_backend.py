from .base_model_backend import BaseModelBackend
from typing import List, Dict, Optional, Any
import json
import time



class AnthropicBackend(BaseModelBackend):
    """Anthropic API backend"""

    _LOG_PREFIX = "[Anthropic]"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        max_context_length: int = 150000
    ):
        """
        Args:
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env var)
            model: Model name
            max_context_length: Maximum context length (default: 150k conservative)
        """
        super().__init__(model=model, max_context_length=max_context_length)

        import os
        # Expose api_key as a plain attr so PLM's spec builder round-trips it to
        # the kernel sub-LLM (else an explicitly-passed key is dropped, leaving
        # only the ANTHROPIC_API_KEY env fallback). Mirrors Slate/VLLM/OpenAI.
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        self.logger.info(f"AnthropicBackend: Using max_context_length: {self._max_context_length:,}")

    @classmethod
    def from_spec(cls, spec: Dict[str, Any], worker_context: Any) -> "AnthropicBackend":
        """Build backend from spec. worker_context provides run_store for hash resolution."""
        return cls(
            api_key=spec.get("api_key"),
            model=spec.get("model", "claude-3-5-sonnet-20241022"),
            max_context_length=spec.get("max_context_length", 150000),
        )

    async def generate(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send request to Anthropic API"""
        messages = self._sanitize_messages_for_api(messages)
        # Convert OpenAI-style messages to Anthropic format
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"] if not system_message else system_message + "\n\n" + msg["content"]
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                blocks: list = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": fn.get("name", ""),
                        "input": json.loads(args) if isinstance(args, str) else args,
                    })
                anthropic_messages.append({"role": "assistant", "content": blocks})
            elif msg["role"] == "tool":
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
                if (anthropic_messages
                        and anthropic_messages[-1]["role"] == "user"
                        and isinstance(anthropic_messages[-1]["content"], list)
                        and anthropic_messages[-1]["content"]
                        and anthropic_messages[-1]["content"][0].get("type") == "tool_result"):
                    anthropic_messages[-1]["content"].append(tool_result_block)
                else:
                    anthropic_messages.append({
                        "role": "user",
                        "content": [tool_result_block],
                    })
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg.get("content", "")})
        
        request_kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
        }
        
        if system_message:
            request_kwargs["system"] = system_message
        
        if tools:
            request_kwargs["tools"] = self._convert_tools_to_anthropic(tools)
        
        # Event logging: model_call_start
        event_logger = kwargs.get("event_logger")
        session_id = kwargs.get("session_id", "unknown")
        round_num = kwargs.get("round_num")
        
        # Extract last user message for logging
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                if isinstance(user_message, list):
                    # Handle structured content
                    user_message = " ".join([part.get("text", "") for part in user_message if part.get("type") == "text"])
                break
        
        if event_logger:
            event_logger.log_event(
                event_type="model_call_start",
                session_id=session_id,
                round_num=round_num,
                event_data={
                    "model": self.model,
                    "backend": "anthropic",
                    "temperature": request_kwargs["temperature"],
                    "max_tokens": request_kwargs["max_tokens"],
                    "has_tools": bool(tools),
                    "has_system": bool(system_message),
                    "user_message": user_message
                }
            )
        
        call_start = time.time()
        response = await self.client.messages.create(**request_kwargs)
        call_duration = time.time() - call_start
        
        # Convert back to unified format
        tool_calls = []
        content = ""
        
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })
        
        # Event logging: model_call_end
        if event_logger:
            event_logger.log_event(
                event_type="model_call_end",
                session_id=session_id,
                round_num=round_num,
                event_data={
                    "model": response.model,
                    "backend": "anthropic",
                    "duration": call_duration,
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                    "has_tool_calls": bool(tool_calls),
                    "stop_reason": response.stop_reason
                },
                raw_response=response
            )
        
        return {
            "content": content,
            "tool_calls": tool_calls if tool_calls else None,
            "usage": self._normalize_usage(response.usage),
            "model": response.model
        }
    
    def _convert_tools_to_anthropic(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI tool format to Anthropic"""
        return [
            {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"]
            }
            for tool in tools
        ]



