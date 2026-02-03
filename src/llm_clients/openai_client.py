"""OpenAI client implementation."""

from typing import Any, Dict, List, Optional

from .base import LLMClient, LLMResponse


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self, api_key: str):
        """Initialize OpenAI client."""
        self.api_key = api_key
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
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
        """Generate completion from OpenAI."""
        messages = [{"role": "user", "content": prompt}]

        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        if tools:
            params["tools"] = tools

        response = await self.client.chat.completions.create(**params)

        message = response.choices[0].message
        usage = response.usage

        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]

        return LLMResponse(
            content=message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            finish_reason=response.choices[0].finish_reason,
            tool_calls=tool_calls,
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

        response = await self.client.chat.completions.create(**params)

        message = response.choices[0].message
        usage = response.usage

        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]

        return LLMResponse(
            content=message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            finish_reason=response.choices[0].finish_reason,
            tool_calls=tool_calls,
            metadata={"response_id": response.id}
        )

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get OpenAI model information."""
        model_info = {
            "gpt-4": {
                "max_tokens": 8192,
                "supports_tools": True,
                "supports_vision": False,
            },
            "gpt-4-turbo": {
                "max_tokens": 128000,
                "supports_tools": True,
                "supports_vision": True,
            },
            "gpt-3.5-turbo": {
                "max_tokens": 16385,
                "supports_tools": True,
                "supports_vision": False,
            },
        }

        return model_info.get(model, {
            "max_tokens": 4096,
            "supports_tools": False,
            "supports_vision": False,
        })
