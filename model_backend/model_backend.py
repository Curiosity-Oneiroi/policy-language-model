from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class ModelBackend(ABC):
    """Abstract base class for model backends"""

    @classmethod
    @abstractmethod
    def from_spec(cls, spec: Dict[str, Any], worker_context: Any) -> "ModelBackend":
        """
        Build backend from spec. worker_context provides run_store for hash resolution, etc.
        """
        ...
    
    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """Return maximum context length for this backend"""
        pass
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send chat completion request
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            Dict with at least:
                - content: str
                - tool_calls: List[Dict] or None
                - usage: Dict with token counts (and optionally extra SDK fields)
                - model: str

            Concrete backends may add more keys (e.g. OpenAIBackend adds
            ``reasoning_summaries``, ``response_output_types``). See
            ``docs/API_REFERENCE.md`` (Model Backends) for details.
        """
        pass