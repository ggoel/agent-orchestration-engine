"""Base LLM client interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion from LLM.

        Args:
            prompt: Input prompt
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools for function calling
            **kwargs: Additional model-specific parameters

        Returns:
            LLMResponse with completion and metadata
        """
        pass

    @abstractmethod
    async def complete_with_tools(
        self,
        messages: List[Dict[str, str]],
        model: str,
        tools: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion with tool use.

        Args:
            messages: Conversation messages
            model: Model name
            tools: Available tools
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            LLMResponse with completion and tool calls
        """
        pass

    @abstractmethod
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get information about a model.

        Returns:
            Dictionary with model capabilities and limits
        """
        pass
