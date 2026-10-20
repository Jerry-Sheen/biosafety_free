"""
OpenAI Client Wrapper for Biosafety Free.
Tolerates missing API keys gracefully for Free/Local Open-Source mode.
"""
import os
from typing import Optional, List, Dict, Any
from config import config
from src.security.logger import get_secure_logger
from src.security.secrets_manager import sanitize_api_key

logger = get_secure_logger("openai_client")


def is_context_length_exceeded(err: Exception) -> bool:
    """Detects OpenAI context length exceeded errors across SDK formats and exceptions."""
    s = f"{err} {repr(err)}".lower()
    if any(k in s for k in ("context_length_exceeded", "maximum context length", "reduce the length of the messages")):
        return True
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        err_info = body.get("error")
        if isinstance(err_info, dict) and err_info.get("code") == "context_length_exceeded":
            return True
    return False


class OpenAIClient:
    """Simple OpenAI client wrapper supporting chat completion and embeddings.
    Tolerates missing API key gracefully so Free/Local mode runs with zero configuration.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, embed_model: Optional[str] = None):
        self.api_key = sanitize_api_key(api_key if api_key is not None else config.openai_api_key)
        self.model = model or getattr(config, "openai_model", None) or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.embed_model = embed_model or config.openai_embed_model or "text-embedding-ada-002"
        self.client = None
        self.is_configured = False

        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                self.is_configured = True
            except Exception as e:
                logger.warning(f"OpenAI client initialization notice: {e}")
                self.client = None
                self.is_configured = False

    def set_api_key(self, api_key: str):
        """Updates and sanitizes the API key, re-initializing the underlying client."""
        self.api_key = sanitize_api_key(api_key)
        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                self.is_configured = True
            except Exception as e:
                logger.warning(f"Failed to re-initialize OpenAI client with key: {e}")
                self.client = None
                self.is_configured = False
        else:
            self.client = None
            self.is_configured = False

    def _ensure_client(self):
        if not self.client or not self.is_configured:
            raise ValueError(
                "OpenAI API 키가 설정되지 않았습니다. 좌측 사이드바에서 유효한 OpenAI API 키(sk-...)를 입력해 주세요."
            )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the chat model."""
        self._ensure_client()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if isinstance(e, UnicodeEncodeError) or "codec can't encode" in str(e):
                logger.error(f"OpenAI UnicodeEncodeError: {e}")
                raise ValueError("OpenAI API 키에 유효하지 않은 문자(한글, 특수 유니코드, 보이지 않는 공백 등)가 포함되어 있습니다. 영문/숫자로 된 올바른 API 키를 입력해 주세요.")
            if is_context_length_exceeded(e) and "3.5" in self.model:
                logger.info(f"Context length exceeded for {self.model}. Auto-upgrading to gpt-4o-mini (128k context).")
                self.model = "gpt-4o-mini"
                return self.generate(prompt, **kwargs)
            raise

    def generate_stream(self, prompt: str, **kwargs):
        """Yield tokens from a streaming chat completion for a simple prompt."""
        self._ensure_client()
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **kwargs,
            )
        except Exception as e:
            if isinstance(e, UnicodeEncodeError) or "codec can't encode" in str(e):
                logger.error(f"OpenAI UnicodeEncodeError: {e}")
                raise ValueError("OpenAI API 키에 유효하지 않은 문자(한글, 특수 유니코드, 보이지 않는 공백 등)가 포함되어 있습니다. 영문/숫자로 된 올바른 API 키를 입력해 주세요.")
            if is_context_length_exceeded(e) and "3.5" in self.model:
                logger.info(f"Context length exceeded for {self.model}. Auto-upgrading to gpt-4o-mini (128k context).")
                self.model = "gpt-4o-mini"
                yield from self.generate_stream(prompt, **kwargs)
                return
            raise

        for chunk in stream:
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def chat_stream(self, messages: List[Dict[str, str]], system: Optional[str] = None, **kwargs):
        """Yield tokens from a streaming chat completion given a list of messages."""
        self._ensure_client()
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                stream=True,
                **kwargs,
            )
        except Exception as e:
            if isinstance(e, UnicodeEncodeError) or "codec can't encode" in str(e):
                logger.error(f"OpenAI UnicodeEncodeError: {e}")
                raise ValueError("OpenAI API 키에 유효하지 않은 문자(한글, 특수 유니코드, 보이지 않는 공백 등)가 포함되어 있습니다. 영문/숫자로 된 올바른 API 키를 입력해 주세요.")
            if is_context_length_exceeded(e):
                logger.warning(f"OpenAI context_length_exceeded on model {self.model}: {e}")
                if "3.5" in self.model:
                    logger.info("Automatically upgrading model from gpt-3.5-turbo to gpt-4o-mini (128k window).")
                    self.model = "gpt-4o-mini"
                    yield from self.chat_stream(messages, system=system, **kwargs)
                    return
                if len(chat_messages) > 2:
                    logger.info("Pruning older chat history turns to recover context length...")
                    condensed = [chat_messages[0], chat_messages[-1]]
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=condensed,
                        stream=True,
                        **kwargs,
                    )
                else:
                    raise
            else:
                raise

        for chunk in stream:
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def embed(self, texts: List[str]) -> List[float]:
        """Return embedding vector for a list of texts."""
        self._ensure_client()
        response = self.client.embeddings.create(model=self.embed_model, input=texts)
        return response.data[0].embedding
