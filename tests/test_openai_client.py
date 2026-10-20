"""
Unit tests for OpenAI client configuration and context length auto-recovery.
"""
from unittest.mock import MagicMock
import openai
from config import config
from src.rag.openai_client import OpenAIClient
from src.rag.engine import ConversationalRAGEngine


def test_openai_default_model_is_gpt_4o_mini():
    """Verify that default model is gpt-4o-mini (128k context) across config and clients."""
    assert config.openai_model == "gpt-4o-mini"
    client = OpenAIClient(api_key="test-key")
    assert client.model == "gpt-4o-mini"


def test_rag_engine_openai_defaults():
    """Verify ConversationalRAGEngine defaults to gpt-4o-mini for OpenAI."""
    engine = ConversationalRAGEngine(llm_provider="openai")
    assert engine.openai_model == "gpt-4o-mini"


def test_openai_context_length_exceeded_auto_recovery():
    """Verify that if gpt-3.5-turbo hits context_length_exceeded, it auto-recovers to gpt-4o-mini."""
    client = OpenAIClient(api_key="test-key", model="gpt-3.5-turbo")

    call_count = 0

    class MockDelta:
        content = "성공적인 복구 완료"

    class MockChoice:
        delta = MockDelta()

    class MockChunk:
        choices = [MockChoice()]

    def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        model_used = kwargs.get("model")
        if model_used == "gpt-3.5-turbo":
            # Simulate OpenAI 400 context_length_exceeded error
            raise openai.BadRequestError(
                message="This model's maximum context length is 16385 tokens. However, your messages resulted in 17869 tokens. Please reduce the length of the messages.",
                response=MagicMock(status_code=400),
                body={"error": {"code": "context_length_exceeded"}}
            )
        # Succeeded on gpt-4o-mini
        return [MockChunk()]

    client.client.chat.completions.create = mock_create

    tokens = list(client.chat_stream([{"role": "user", "content": "질문"}]))
    assert "성공적인 복구 완료" in tokens
    assert call_count == 2
    assert client.model == "gpt-4o-mini"


def test_sanitize_api_key():
    """Verify that sanitize_api_key strips prefixes, unicode dashes, quotes, and non-ascii chars."""
    from src.security.secrets_manager import sanitize_api_key

    # 1. Unicode dashes and quotes
    dirty_1 = '  “OPENAI_API_KEY: sk-proj\u201312345678901234567890\u200b”  '
    assert sanitize_api_key(dirty_1) == "sk-proj-12345678901234567890"

    # 2. Variable assignment prefix
    dirty_2 = "OPENAI_API_KEY=sk-proj-abcdef12345678901234"
    assert sanitize_api_key(dirty_2) == "sk-proj-abcdef12345678901234"

    # 3. Accidental Korean chars around key
    dirty_3 = "sk-proj-abcdef12345678901234한글"
    assert sanitize_api_key(dirty_3) == "sk-proj-abcdef12345678901234"

    # 4. Pure Korean input produces empty string (safe from ascii encode error)
    dirty_4 = "오픈에이아이키"
    assert sanitize_api_key(dirty_4) == ""


def test_openai_client_handles_non_ascii_safely():
    """Verify that OpenAIClient never passes non-ascii characters to httpx."""
    client = OpenAIClient(api_key="sk-123456789012345한글키")
    assert client.api_key.isascii()

