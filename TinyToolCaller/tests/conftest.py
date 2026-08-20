"""Shared fixtures for the TinyToolCaller test suite."""

import sys
from pathlib import Path

import pytest

# Make the package importable when running `pytest` from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeTokenizer:
    """Minimal stand-in for a chat-template tokenizer (pure Python)."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"[{m['role']}]{m['content']}[/{m['role']}]")
        if add_generation_prompt:
            parts.append("[assistant]")
        return "".join(parts)


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def weather_example():
    return {
        "query": "What's the weather in Tokyo?",
        "tools": [{
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }],
        "answers": [{"name": "get_weather",
                     "arguments": {"location": "Tokyo", "unit": "celsius"}}],
    }
