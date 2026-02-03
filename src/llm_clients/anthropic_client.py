"""Anthropic Claude client implementation."""

from typing import Any, Dict, List, Optional

from .base import LLMClient, LLMResponse


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(self, api_key: str):
        """Initialize Anthropic client."""
        self.api_key = api_key
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion from Claude."""
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        if tools:
            params["tools"] = tools

        response = await self.client.messages.create(**params)

        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": block.input
                    }
                })

        return LLMResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason,
            tool_calls=tool_calls if tool_calls else None,
            metadata={"response_id": response.id}
        )

    async def complete_with_tools(
        self,
        messages: List[Dict[str, str]],
        model: str,
        tools: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate completion with tool use."""
        params = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        response = await self.client.messages.create(**params)

        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": block.input
                    }
                })

        return LLMResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason,
            tool_calls=tool_calls if tool_calls else None,
            metadata={"response_id": response.id}
        )

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get Claude model information."""
        model_info = {
            "claude-3-opus-20240229": {
                "max_tokens": 200000,
                "supports_tools": True,
                "supports_vision": True,
            },
            "claude-3-sonnet-20240229": {
                "max_tokens": 200000,
                "supports_tools": True,
                "supports_vision": True,
            },
            "claude-3-haiku-20240307": {
                "max_tokens": 200000,
                "supports_tools": True,
                "supports_vision": True,
            },
        }

        return model_info.get(model, {
            "max_tokens": 100000,
            "supports_tools": True,
            "supports_vision": False,
        })
