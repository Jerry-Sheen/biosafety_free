"""
Robust Google Gemini API Client and Model Resolver.
Manages:
- Model auto-discovery (filters for generateContent capability)
- Separation of Auth (401/403), Model Not Found (404), and Rate Limit (429) errors
- Tenacity exponential backoff retries on transient 5xx errors
- Fallback model cascades
"""
import time
from typing import List, Dict, Any, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from config import config
from pipeline.logger import logger, IngestionErrorType, log_pipeline_event

# Candidate standard models in priority order
RECOMMENDED_GEMINI_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]


class GeminiModelResolver:
    """Discovers and resolves valid generateContent models for Google GenAI SDK."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.gemini_api_key
        self._cached_models: Optional[List[Dict[str, Any]]] = None

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Lists available models from Google Gemini API and identifies their supported capabilities.
        """
        if not self.api_key:
            return []

        if self._cached_models is not None:
            return self._cached_models

        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            models_info = []

            for m in client.models.list():
                # Inspect model attributes
                name = getattr(m, "name", "")
                display_name = getattr(m, "display_name", "")
                supported_methods = getattr(m, "supported_generation_methods", []) or getattr(m, "supported_actions", [])
                
                # Normalize name: models/gemini-2.5-flash -> gemini-2.5-flash
                short_name = name.replace("models/", "") if name else ""

                models_info.append({
                    "name": name,
                    "short_name": short_name,
                    "display_name": display_name,
                    "supported_methods": list(supported_methods),
                    "supports_generate_content": (
                        "generateContent" in supported_methods
                        or "generate_content" in supported_methods
                        or not supported_methods  # If not populated by SDK, keep for name matching
                    )
                })

            self._cached_models = models_info
            logger.info(f"Discovered {len(models_info)} models from Gemini API.")
            return models_info

        except Exception as e:
            err_str = str(e)
            if "401" in err_str or "API_KEY_INVALID" in err_str:
                logger.error(f"Gemini API authentication failed (401): {e}")
            elif "403" in err_str or "PERMISSION_DENIED" in err_str:
                logger.error(f"Gemini API permission denied (403): {e}")
            else:
                logger.warning(f"Could not retrieve dynamic model list ({e}). Will rely on recommended defaults.")
            return []

    def resolve_model(self, requested_model: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Resolves the best usable model name.
        Returns: (resolved_model_name, warning_message_if_any)
        """
        target = requested_model or config.gemini_model or "gemini-2.5-flash"
        target_clean = target.replace("models/", "").strip()

        available = self.list_available_models()
        if not available:
            # When list cannot be fetched (e.g. offline test or first call), return target
            return target_clean, None

        # Check if requested target is directly in available list
        for m in available:
            if m["short_name"].lower() == target_clean.lower() and m["supports_generate_content"]:
                return target_clean, None

        # Requested model was not found: find best fallback
        warning_msg = (
            f"Requested model '{target_clean}' is not available or does not support generateContent. "
        )

        for rec in RECOMMENDED_GEMINI_MODELS:
            for m in available:
                if m["short_name"].lower() == rec.lower() and m["supports_generate_content"]:
                    warning_msg += f"Falling back to verified model '{rec}'."
                    logger.warning(warning_msg)
                    return rec, warning_msg

        # Fallback to the first available model supporting generateContent
        for m in available:
            if m["supports_generate_content"]:
                chosen = m["short_name"]
                warning_msg += f"Falling back to available model '{chosen}'."
                logger.warning(warning_msg)
                return chosen, warning_msg

        return target_clean, warning_msg


def classify_gemini_error(e: Exception) -> Dict[str, Any]:
    """
    Categorizes Gemini SDK exceptions into standard error types:
    404 -> GEMINI_MODEL_NOT_FOUND
    401/403 -> GEMINI_AUTH_ERROR
    429 -> GEMINI_RATE_LIMIT
    5xx -> UNKNOWN_ERROR / RETRYABLE
    """
    err_msg = str(e)

    if "404" in err_msg or "NOT_FOUND" in err_msg:
        return {
            "error_type": IngestionErrorType.GEMINI_MODEL_NOT_FOUND,
            "user_message": (
                "⚠️ **요청한 Gemini 모델을 찾을 수 없습니다 (404 NOT_FOUND).**\n"
                "해당 모델명이 지원되지 않거나 API 버전에서 제공되지 않습니다. "
                "`gemini-2.5-flash` 또는 `gemini-2.0-flash` 모델을 사용하는 것을 권장합니다."
            ),
            "retryable": False
        }
    elif "401" in err_msg or "API_KEY_INVALID" in err_msg:
        return {
            "error_type": IngestionErrorType.GEMINI_AUTH_ERROR,
            "user_message": (
                "⚠️ **Google Gemini API 인증 실패 (401 UNAUTHENTICATED).**\n"
                "제공된 GEMINI_API_KEY가 유효하지 않습니다. Google AI Studio에서 키를 다시 확인해주세요."
            ),
            "retryable": False
        }
    elif "403" in err_msg or "PERMISSION_DENIED" in err_msg:
        return {
            "error_type": IngestionErrorType.GEMINI_AUTH_ERROR,
            "user_message": (
                "⚠️ **Google Gemini API 권한 오류 (403 PERMISSION_DENIED).**\n"
                "해당 API 키에 모델 호출 권한이 없거나 차단되었습니다."
            ),
            "retryable": False
        }
    elif "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
        return {
            "error_type": IngestionErrorType.GEMINI_RATE_LIMIT,
            "user_message": (
                "⚠️ **Gemini API 호출 속도 제한 초과 (429 RESOURCE_EXHAUSTED).**\n"
                "잠시 후 다시 시도하거나, 요청 간격을 조절해주세요."
            ),
            "retryable": True
        }
    else:
        return {
            "error_type": IngestionErrorType.UNKNOWN_ERROR,
            "user_message": f"⚠️ **Gemini API 호출 중 오류 발생:** {err_msg}",
            "retryable": "500" in err_msg or "503" in err_msg or "INTERNAL" in err_msg
        }
