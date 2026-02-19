import os
import pytest
from agents.llm import LLMClient

openai = pytest.importorskip("openai")

@pytest.mark.skipif(os.getenv("OPENAI_API_KEY") is None, reason="OPENAI_API_KEY not set")
def test_openai_provider_smoke():
    client = LLMClient(provider="openai", token_budget=2000)
    res = client.call("Say hello", max_tokens=5)
    assert isinstance(res, dict) and "text" in res and isinstance(res["text"], str)
