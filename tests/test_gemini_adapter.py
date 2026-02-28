"""
Tests for GeminiAdapter and updated adapter factory.

Tests cover:
1. GeminiAdapter instantiation and properties
2. Tool format conversion (OpenAI → Gemini)
3. Factory function (get_adapter) with "gemini" and "google" providers
4. Import verification
5. Integration test with real API (skipped unless GEMINI_API_KEY is set)
"""

import asyncio
import os
import json
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentvault.adapters import (
    BaseLLMAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    OllamaAdapter,
    get_adapter,
    auto_detect_adapter,
)


# -------------------------------------------------------------------------
# Unit Tests (no API key needed)
# -------------------------------------------------------------------------

class TestGeminiAdapterUnit:
    """Tests that don't require an API key."""

    def test_instantiation(self):
        adapter = GeminiAdapter(api_key="test-key")
        assert adapter._api_key == "test-key"
        assert adapter._model == "gemini-2.0-flash"
        assert adapter._client is None

    def test_custom_model(self):
        adapter = GeminiAdapter(api_key="test-key", model="gemini-2.5-pro")
        assert adapter._model == "gemini-2.5-pro"

    def test_name_property(self):
        adapter = GeminiAdapter(api_key="test-key")
        assert adapter.name == "Gemini (gemini-2.0-flash)"

    def test_name_custom_model(self):
        adapter = GeminiAdapter(api_key="test-key", model="gemini-2.5-pro")
        assert adapter.name == "Gemini (gemini-2.5-pro)"

    def test_is_base_adapter(self):
        adapter = GeminiAdapter(api_key="test-key")
        assert isinstance(adapter, BaseLLMAdapter)

    def test_tool_conversion_openai_format(self):
        """Test converting OpenAI tool format to Gemini format."""
        adapter = GeminiAdapter(api_key="test-key")
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
        result = adapter._convert_tools_to_gemini(openai_tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather for a city"
        assert "properties" in result[0]["parameters"]
        assert "city" in result[0]["parameters"]["properties"]

    def test_tool_conversion_flat_format(self):
        """Test converting flat tool dict (no 'function' wrapper)."""
        adapter = GeminiAdapter(api_key="test-key")
        flat_tools = [
            {
                "name": "read_file",
                "description": "Read a file from disk",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                },
            }
        ]
        result = adapter._convert_tools_to_gemini(flat_tools)
        assert len(result) == 1
        assert result[0]["name"] == "read_file"

    def test_tool_conversion_empty_params(self):
        """Tool with empty parameters should have None params."""
        adapter = GeminiAdapter(api_key="test-key")
        tools = [{"name": "ping", "description": "Ping the server", "parameters": {}}]
        result = adapter._convert_tools_to_gemini(tools)
        assert len(result) == 1
        assert "parameters" not in result[0]

    def test_tool_conversion_multiple(self):
        """Multiple tools should all be converted."""
        adapter = GeminiAdapter(api_key="test-key")
        tools = [
            {"name": "tool_a", "description": "A"},
            {"name": "tool_b", "description": "B"},
            {"name": "tool_c", "description": "C"},
        ]
        result = adapter._convert_tools_to_gemini(tools)
        assert len(result) == 3
        assert [r["name"] for r in result] == ["tool_a", "tool_b", "tool_c"]


class TestAdapterFactory:
    """Tests for get_adapter factory function."""

    def test_gemini_provider(self):
        adapter = get_adapter("gemini", api_key="test-key")
        assert isinstance(adapter, GeminiAdapter)
        assert adapter._model == "gemini-2.0-flash"

    def test_google_provider_alias(self):
        adapter = get_adapter("google", api_key="test-key")
        assert isinstance(adapter, GeminiAdapter)

    def test_gemini_custom_model(self):
        adapter = get_adapter("gemini", api_key="test-key", model="gemini-2.5-pro")
        assert isinstance(adapter, GeminiAdapter)
        assert adapter._model == "gemini-2.5-pro"

    def test_gemini_case_insensitive(self):
        adapter = get_adapter("GEMINI", api_key="test-key")
        assert isinstance(adapter, GeminiAdapter)

    def test_gemini_no_key_raises(self):
        with pytest.raises(ValueError, match="Gemini adapter requires an API key"):
            get_adapter("gemini")

    def test_openai_still_works(self):
        adapter = get_adapter("openai", api_key="test-key")
        assert isinstance(adapter, OpenAIAdapter)

    def test_anthropic_still_works(self):
        adapter = get_adapter("anthropic", api_key="test-key")
        assert isinstance(adapter, AnthropicAdapter)

    def test_ollama_still_works(self):
        adapter = get_adapter("ollama")
        assert isinstance(adapter, OllamaAdapter)

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_adapter("nonexistent")

    def test_error_message_lists_gemini(self):
        """Error for unknown provider should mention gemini as an option."""
        try:
            get_adapter("xyz")
        except ValueError as e:
            assert "gemini" in str(e).lower()


class TestImports:
    """Verify GeminiAdapter is properly exported from the package."""

    def test_import_from_adapters(self):
        from agentvault.adapters import GeminiAdapter
        assert GeminiAdapter is not None

    def test_import_from_package(self):
        from agentvault import GeminiAdapter
        assert GeminiAdapter is not None

    def test_in_all(self):
        import agentvault
        assert "GeminiAdapter" in agentvault.__all__


# -------------------------------------------------------------------------
# Integration Tests (require GEMINI_API_KEY env var)
# -------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
skip_no_key = pytest.mark.skipif(
    not GEMINI_API_KEY,
    reason="GEMINI_API_KEY or GOOGLE_API_KEY env var not set"
)


@skip_no_key
class TestGeminiIntegration:
    """Live integration tests with the Gemini API."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        adapter = GeminiAdapter(api_key=GEMINI_API_KEY)
        result = await adapter.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_simple_generation(self):
        adapter = GeminiAdapter(api_key=GEMINI_API_KEY)
        result = await adapter.generate(
            messages=[{"role": "user", "content": "Say hello in exactly 3 words."}],
            temperature=0.0,
            max_tokens=20,
        )
        assert "content" in result
        assert len(result["content"]) > 0
        assert result["model"] == "gemini-2.0-flash"
        assert "usage" in result
        assert result["usage"]["prompt_tokens"] > 0

    @pytest.mark.asyncio
    async def test_system_instruction(self):
        adapter = GeminiAdapter(api_key=GEMINI_API_KEY)
        result = await adapter.generate(
            messages=[
                {"role": "system", "content": "You are a pirate. Always respond with 'Arrr!'."},
                {"role": "user", "content": "Hello"},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        assert "content" in result
        assert len(result["content"]) > 0

    @pytest.mark.asyncio
    async def test_function_calling(self):
        adapter = GeminiAdapter(api_key=GEMINI_API_KEY)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a given city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city name",
                            },
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
        result = await adapter.generate(
            messages=[
                {"role": "user", "content": "What's the weather in Tokyo?"},
            ],
            tools=tools,
            temperature=0.0,
        )
        # Should return tool calls
        assert "tool_calls" in result
        assert len(result["tool_calls"]) > 0
        tc = result["tool_calls"][0]
        assert tc["name"] == "get_weather"
        assert "city" in tc["arguments"]
        assert "id" in tc

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        adapter = GeminiAdapter(api_key=GEMINI_API_KEY)
        result = await adapter.generate(
            messages=[
                {"role": "user", "content": "My name is Shield."},
                {"role": "assistant", "content": "Nice to meet you, Shield!"},
                {"role": "user", "content": "What is my name?"},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        assert "content" in result
        assert "shield" in result["content"].lower()

    @pytest.mark.asyncio
    async def test_via_factory(self):
        adapter = get_adapter("gemini", api_key=GEMINI_API_KEY)
        result = await adapter.generate(
            messages=[{"role": "user", "content": "Reply with only the word 'ok'"}],
            temperature=0.0,
            max_tokens=10,
        )
        assert "content" in result

    @pytest.mark.asyncio
    async def test_via_google_alias(self):
        adapter = get_adapter("google", api_key=GEMINI_API_KEY)
        healthy = await adapter.health_check()
        assert healthy is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
