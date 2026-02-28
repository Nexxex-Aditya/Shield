"""
AgentVault — LLM Adapters
Unified interface for any LLM provider.
Supports OpenAI, Anthropic, Google Gemini, and Ollama out of the box.

Gemini integration uses the google-genai SDK (pip install google-genai).
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger("agentvault.adapters")


class BaseLLMAdapter(ABC):
    """
    Base interface for all LLM adapters.
    Extend this to support any LLM API.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            messages: Chat messages [{"role": "user", "content": "..."}]
            tools: Optional tool definitions for function calling
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            dict with keys: content, tool_calls (optional), model, usage
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM is available."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this adapter."""
        ...


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for OpenAI API (GPT-4o, GPT-4, etc.)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        result: dict[str, Any] = {
            "content": message.content or "",
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }

        # Parse tool calls
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in message.tool_calls
            ]

        return result

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(response.choices)
        except Exception as e:
            logger.error("OpenAI health check failed: %s", e)
            return False

    @property
    def name(self) -> str:
        return f"OpenAI ({self._model})"


class AnthropicAdapter(BaseLLMAdapter):
    """Adapter for Anthropic API (Claude)."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        client = self._get_client()

        # Anthropic uses a different format — system message separate
        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_msg:
            kwargs["system"] = system_msg

        if tools:
            # Convert OpenAI tool format to Anthropic format
            anthropic_tools = []
            for tool in tools:
                func = tool.get("function", tool)
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
            kwargs["tools"] = anthropic_tools

        response = await client.messages.create(**kwargs)

        result: dict[str, Any] = {
            "content": "",
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
        }

        # Parse content blocks
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        if tool_calls:
            result["tool_calls"] = tool_calls

        return result

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            response = await client.messages.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(response.content)
        except Exception as e:
            logger.error("Anthropic health check failed: %s", e)
            return False

    @property
    def name(self) -> str:
        return f"Anthropic ({self._model})"


class GeminiAdapter(BaseLLMAdapter):
    """
    Adapter for Google Gemini API (Gemini 2.0 Flash, Gemini 2.5 Pro, etc.).

    Uses the official google-genai SDK.
    Supports function calling with automatic format conversion from
    OpenAI-style tool definitions.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _convert_tools_to_gemini(self, tools: list[dict]) -> list[dict]:
        """
        Convert OpenAI-style tool definitions to Gemini function declarations.

        OpenAI format:
            {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
        Gemini format:
            {"name": ..., "description": ..., "parameters": {...}}
        """
        declarations = []
        for tool in tools:
            func = tool.get("function", tool)
            params = func.get("parameters", {})

            # Gemini doesn't allow empty properties in parameters
            if params and not params.get("properties"):
                params = None

            decl = {
                "name": func["name"],
                "description": func.get("description", ""),
            }
            if params:
                decl["parameters"] = params
            declarations.append(decl)
        return declarations

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        client = self._get_client()
        from google.genai import types

        # Separate system instruction from messages
        system_instruction = None
        chat_contents = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "assistant":
                chat_contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=content)],
                    )
                )
            elif role == "tool":
                # Tool results sent back to the model
                chat_contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(
                            text=f"Tool result: {content}"
                        )],
                    )
                )
            else:
                # user messages
                chat_contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content)],
                    )
                )

        # Build generation config
        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        if system_instruction:
            gen_config.system_instruction = system_instruction

        # Add tools if provided
        if tools:
            declarations = self._convert_tools_to_gemini(tools)
            gen_config.tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(**d) for d in declarations
                    ]
                )
            ]

        # Call the API
        response = await client.aio.models.generate_content(
            model=self._model,
            contents=chat_contents,
            config=gen_config,
        )

        # Parse response into our unified format
        result: dict[str, Any] = {
            "content": "",
            "model": self._model,
            "usage": {
                "prompt_tokens": getattr(
                    getattr(response, "usage_metadata", None),
                    "prompt_token_count", 0
                ) or 0,
                "completion_tokens": getattr(
                    getattr(response, "usage_metadata", None),
                    "candidates_token_count", 0
                ) or 0,
            },
        }

        # Extract text content and tool calls from candidates
        tool_calls = []
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    result["content"] += part.text
                elif part.function_call:
                    fc = part.function_call
                    tool_calls.append({
                        "id": f"gemini_{fc.name}_{len(tool_calls)}",
                        "name": fc.name,
                        "arguments": dict(fc.args) if fc.args else {},
                    })

        if tool_calls:
            result["tool_calls"] = tool_calls

        return result

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            response = await client.aio.models.generate_content(
                model=self._model,
                contents="ping",
            )
            return bool(response.candidates)
        except Exception as e:
            logger.error("Gemini health check failed: %s", e)
            return False

    @property
    def name(self) -> str:
        return f"Gemini ({self._model})"


class OllamaAdapter(BaseLLMAdapter):
    """Adapter for Ollama (local LLM inference)."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        import httpx

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        message = data.get("message", {})

        result: dict[str, Any] = {
            "content": message.get("content", ""),
            "model": data.get("model", self._model),
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        }

        # Parse tool calls from Ollama response
        if message.get("tool_calls"):
            result["tool_calls"] = [
                {
                    "id": f"ollama_{i}",
                    "name": tc["function"]["name"],
                    "arguments": tc["function"].get("arguments", {}),
                }
                for i, tc in enumerate(message["tool_calls"])
            ]

        return result

    async def health_check(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.debug("Ollama health check failed: %s", e)
            return False

    @property
    def name(self) -> str:
        return f"Ollama ({self._model})"

    async def list_models(self) -> list[str]:
        """List available Ollama models."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


def get_adapter(
    provider: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> BaseLLMAdapter:
    """
    Factory function to create an LLM adapter by provider name.

    Args:
        provider: "openai", "anthropic", "gemini"/"google", "ollama"
        api_key: API key (not needed for Ollama)
        model: Model name override
        **kwargs: Additional provider-specific arguments
    """
    provider = provider.lower().strip()

    if provider == "openai":
        if not api_key:
            raise ValueError("OpenAI adapter requires an API key")
        return OpenAIAdapter(
            api_key=api_key,
            model=model or "gpt-4o",
            **kwargs,
        )

    elif provider == "anthropic":
        if not api_key:
            raise ValueError("Anthropic adapter requires an API key")
        return AnthropicAdapter(
            api_key=api_key,
            model=model or "claude-sonnet-4-20250514",
            **kwargs,
        )

    elif provider in ("gemini", "google"):
        if not api_key:
            raise ValueError("Gemini adapter requires an API key")
        return GeminiAdapter(
            api_key=api_key,
            model=model or "gemini-2.0-flash",
            **kwargs,
        )

    elif provider == "ollama":
        return OllamaAdapter(
            model=model or "llama3.2",
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: openai, anthropic, gemini, ollama. "
            f"Or extend BaseLLMAdapter for custom providers."
        )


async def auto_detect_adapter(
    api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
) -> Optional[BaseLLMAdapter]:
    """
    Auto-detect available LLM. Checks Ollama first (local), then falls back
    to cloud providers if API keys are provided.

    Args:
        api_key: OpenAI / Anthropic API key
        gemini_api_key: Google Gemini API key (separate because provider differs)
    """
    import os

    # Try Ollama first (local, no key needed)
    ollama = OllamaAdapter()
    if await ollama.health_check():
        logger.info("Auto-detected Ollama at localhost:11434")
        return ollama

    # Try OpenAI if key provided
    if api_key:
        openai_adapter = OpenAIAdapter(api_key=api_key)
        if await openai_adapter.health_check():
            logger.info("Auto-detected OpenAI API")
            return openai_adapter

    # Try Gemini if key provided (or from env)
    gkey = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gkey:
        gemini_adapter = GeminiAdapter(api_key=gkey)
        if await gemini_adapter.health_check():
            logger.info("Auto-detected Google Gemini API")
            return gemini_adapter

    logger.warning("No LLM adapter auto-detected")
    return None
