"""
Local High-Performance Open-Source LLM Provider.
Supports two backends:
1. (Primary) llama-cpp-python with 4-bit quantized GGUF models (3-5x faster, AVX-512/AVX2 native C++ kernel, ~1GB RAM).
2. (Fallback) HuggingFace Transformers with CPU threading optimization.
Features lazy importing and singleton caching so application startup is instantaneous (< 1s).
"""

import os
import queue
import threading
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional

from config import config
from src.security.logger import get_secure_logger

logger = get_secure_logger("local_llm")

_LOCAL_LLM_CACHE: Optional["LocalLLMProvider"] = None
_CACHE_LOCK = threading.Lock()


class LocalLLMProvider:
    """Singleton provider for high-efficiency CPU-based open-source language models."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        gguf_repo: Optional[str] = None,
        gguf_file: Optional[str] = None,
        n_ctx: Optional[int] = None
    ):
        self.model_name = model_name or getattr(config, "local_llm_model", "Qwen/Qwen2.5-1.5B-Instruct")
        self.device = device or getattr(config, "local_llm_device", "cpu")
        self.gguf_repo = gguf_repo or getattr(config, "local_llm_gguf_repo", "Qwen/Qwen2.5-1.5B-Instruct-GGUF")
        self.gguf_file = gguf_file or getattr(config, "local_llm_gguf_file", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
        self.n_ctx = n_ctx or getattr(config, "local_llm_n_ctx", 16384)

        # Engine state
        self.engine_type = "uninitialized"  # "llama_cpp" or "transformers"
        self.llm = None           # Llama instance if llama_cpp
        self.tokenizer = None     # AutoTokenizer if transformers
        self.model = None         # AutoModelForCausalLM if transformers
        self._is_ready = False
        self._load_lock = threading.Lock()

    @property
    def model_display_name(self) -> str:
        """Returns descriptive model name including engine and quantization info."""
        if self.engine_type == "llama_cpp":
            return f"Qwen2.5-1.5B-Instruct (GGUF 4-bit · llama.cpp 고속 엔진)"
        return f"{self.model_name} (PyTorch CPU)"

    def _resolve_gguf_path(self) -> Optional[str]:
        """Finds local GGUF model file or downloads it via huggingface_hub."""
        # 1. Check local models/ directory
        local_path = Path(__file__).resolve().parent.parent.parent / "models" / self.gguf_file
        if local_path.is_file():
            logger.info(f"Found local GGUF model file at: {local_path}")
            return str(local_path)

        # 2. Check HuggingFace hub cache or download
        try:
            from huggingface_hub import hf_hub_download
            logger.info(f"Resolving GGUF model via huggingface_hub ({self.gguf_repo}/{self.gguf_file})...")
            cached_path = hf_hub_download(
                repo_id=self.gguf_repo,
                filename=self.gguf_file
            )
            if cached_path and os.path.isfile(cached_path):
                logger.info(f"Resolved GGUF model path from HF cache: {cached_path}")
                return cached_path
        except Exception as e:
            logger.warning(f"Failed to resolve GGUF via huggingface_hub: {e}")

        return None

    def _load_llama_cpp(self) -> bool:
        """Loads GGUF model using high-performance llama-cpp-python."""
        try:
            from llama_cpp import Llama

            model_path = self._resolve_gguf_path()
            if not model_path or not os.path.isfile(model_path):
                logger.warning(f"GGUF model file not found ({self.gguf_file}); cannot initialize llama_cpp.")
                return False

            num_threads = min(4, os.cpu_count() or 4)
            logger.info(f"Initializing llama-cpp-python with model '{model_path}', n_ctx={self.n_ctx}, threads={num_threads}...")

            self.llm = Llama(
                model_path=model_path,
                n_ctx=self.n_ctx,
                n_threads=num_threads,
                n_batch=512,
                verbose=False
            )
            self.engine_type = "llama_cpp"
            self._is_ready = True
            logger.info("[SUCCESS] llama-cpp-python GGUF 4-bit engine successfully initialized!")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize llama-cpp-python engine ({e}). Will attempt transformers fallback.", exc_info=True)
            return False

    def _load_transformers(self) -> bool:
        """Fallback loader using Hugging Face Transformers on CPU."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Fallback: Loading Transformers model '{self.model_name}' on CPU...")
            num_threads = min(4, os.cpu_count() or 4)
            torch.set_num_threads(num_threads)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            torch_dtype = torch.float32 if self.device == "cpu" else (
                torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float32
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cpu" if self.device == "cpu" else None,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.model.eval()
            self.engine_type = "transformers"
            self._is_ready = True
            logger.info("Transformers CPU engine successfully loaded.")
            return True
        except Exception as e:
            logger.error(f"Failed to load Transformers model '{self.model_name}': {e}", exc_info=True)
            return False

    def load_model(self) -> bool:
        """Loads the model using llama-cpp-python (primary) or Transformers (fallback)."""
        if self._is_ready:
            return True

        with self._load_lock:
            if self._is_ready:
                return True

            # Try primary high-speed llama_cpp first
            if self._load_llama_cpp():
                return True

            # Fallback to transformers
            return self._load_transformers()

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """
        Generates tokens in real time with streaming output.
        Automatically utilizes llama_cpp (fast) or transformers (fallback).
        """
        if not self._is_ready:
            success = self.load_model()
            if not success:
                yield "⚠️ 로컬 오픈소스 LLM 모델을 로드하는 중 오류가 발생했습니다. 서버 로그를 확인해 주세요."
                return

        # ------------------ ENGINE 1: llama-cpp-python (Fast GGUF) ------------------
        if self.engine_type == "llama_cpp" and self.llm is not None:
            try:
                # Safety guard: ensure messages fit within n_ctx minus requested generation tokens
                effective_n_ctx = self.llm.n_ctx() if hasattr(self.llm, "n_ctx") else self.n_ctx
                safe_limit = max(1024, effective_n_ctx - max_new_tokens - 128)

                processed_messages = []
                for m in messages:
                    processed_messages.append({"role": m["role"], "content": m["content"]})

                # Check total text volume and trim middle of context if excessively large
                total_chars = sum(len(m.get("content", "")) for m in processed_messages)
                max_allowed_chars = int(safe_limit * 1.8)
                if total_chars > max_allowed_chars:
                    logger.info(f"Input messages ({total_chars} chars) exceed safe limit ({max_allowed_chars} chars). Trimming context...")
                    for m in processed_messages:
                        if m["role"] == "user" and len(m["content"]) > max_allowed_chars:
                            head_part = m["content"][:int(max_allowed_chars * 0.45)]
                            tail_part = m["content"][-int(max_allowed_chars * 0.45):]
                            m["content"] = f"{head_part}\n\n...[중간 규정 컨텍스트 자동 압축: 컨텍스트 한도 초과 방지]...\n\n{tail_part}"

                # OpenAI-compatible streaming chat completion
                response_stream = self.llm.create_chat_completion(
                    messages=processed_messages,
                    max_tokens=max_new_tokens,
                    temperature=max(temperature, 0.01),
                    top_p=top_p,
                    stream=True
                )
                for chunk in response_stream:
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                return
            except Exception as e:
                logger.error(f"Error during llama_cpp streaming: {e}", exc_info=True)
                yield f"\n[오류: GGUF 모델 생성 중 예외 발생 - {str(e)}]"
                return

        # ------------------ ENGINE 2: Transformers Fallback ------------------
        try:
            import torch
            from transformers import TextIteratorStreamer

            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt = ""
                for m in messages:
                    prompt += f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '')}<|im_end|>\n"
                prompt += "<|im_start|>assistant\n"

            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=180.0
            )

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

            worker_exception: List[Exception] = []

            def _generate_worker():
                try:
                    with torch.inference_mode():
                        self.model.generate(**generation_kwargs)
                except Exception as ex:
                    worker_exception.append(ex)

            thread = threading.Thread(target=_generate_worker, daemon=True)
            thread.start()

            for new_text in streamer:
                yield new_text

            thread.join(timeout=2.0)
            if worker_exception:
                raise worker_exception[0]

        except Exception as e:
            if isinstance(e, queue.Empty) or type(e).__name__ == "Empty":
                err_detail = "CPU 추론 타임아웃(180초 초과) - 컨텍스트 길이가 길어 로컬 처리 시간이 초과되었습니다."
            else:
                err_detail = str(e) if str(e) else f"{type(e).__name__} 오류"
            logger.error(f"Error during local LLM streaming generation: {err_detail}", exc_info=True)
            yield f"\n[오류: 로컬 모델 생성 중 예외 발생 - {err_detail}]"


def get_local_llm_provider() -> LocalLLMProvider:
    """Returns the singleton instance of LocalLLMProvider."""
    global _LOCAL_LLM_CACHE
    with _CACHE_LOCK:
        if _LOCAL_LLM_CACHE is None:
            _LOCAL_LLM_CACHE = LocalLLMProvider()
        return _LOCAL_LLM_CACHE
