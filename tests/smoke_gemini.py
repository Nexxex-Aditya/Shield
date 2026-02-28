"""Quick integration smoke test for GeminiAdapter."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentvault.adapters import GeminiAdapter, get_adapter

key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


async def run_tests():
    if not key:
        print("SKIP: No GEMINI_API_KEY or GOOGLE_API_KEY set")
        return

    print(f"API key found (length: {len(key)})")
    adapter = GeminiAdapter(api_key=key)

    # Test 1: Health check
    print("\n--- Test 1: Health Check ---")
    try:
        healthy = await adapter.health_check()
        print(f"Healthy: {healthy}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

    # Test 2: Simple generation
    print("\n--- Test 2: Simple Generation ---")
    try:
        result = await adapter.generate(
            messages=[{"role": "user", "content": "Respond with only the word OK"}],
            temperature=0.0,
            max_tokens=10,
        )
        content = result["content"]
        model = result["model"]
        usage = result["usage"]
        print(f"Content: {content!r}")
        print(f"Model: {model}")
        print(f"Usage: {usage}")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

    # Test 3: System instruction
    print("\n--- Test 3: System Instruction ---")
    try:
        result = await adapter.generate(
            messages=[
                {"role": "system", "content": "Always respond in exactly one word."},
                {"role": "user", "content": "Hello"},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        print(f"Content: {result['content']!r}")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

    # Test 4: Function calling
    print("\n--- Test 4: Function Calling ---")
    try:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
        result = await adapter.generate(
            messages=[{"role": "user", "content": "What is the weather in London?"}],
            tools=tools,
            temperature=0.0,
        )
        if "tool_calls" in result:
            for tc in result["tool_calls"]:
                print(f"Tool call: {tc['name']}({tc['arguments']})")
            print("PASSED")
        else:
            print(f"No tool calls returned. Content: {result['content']!r}")
            print("WARN: Expected tool calls but got text")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

    # Test 5: Factory
    print("\n--- Test 5: Factory (gemini) ---")
    try:
        adapter2 = get_adapter("gemini", api_key=key)
        result = await adapter2.generate(
            messages=[{"role": "user", "content": "Say yes"}],
            max_tokens=5,
        )
        print(f"Content: {result['content']!r}")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

    print("\n=== All integration tests complete ===")


if __name__ == "__main__":
    asyncio.run(run_tests())
