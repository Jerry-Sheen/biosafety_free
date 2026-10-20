"""
Local CPU-Optimized Open-Source LLM Provider.
Provides memory-efficient, single-instance CPU inference with streaming output.
Features lazy importing so application startup is instantaneous (< 1s).
"""

import os
import queue
import threading
from typing import List, Dict, Any, Generator, Optional

from src.config import config
from src.security.logger import get_secure_logger

logger = get_secure_logger("local_llm")

_LOCAL_LLM_CACHE: Optional["LocalLLMProvider"] = None
_CACHE_LOCK = threading.Lock()


class LocalLLMProvider:
    """Singleton provider for CPU-based open-source language models."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or getattr(config, "local_llm_model", "Qwen/Qwen2.5-1.5B-Instruct")
        self.device = device or getattr(config, "local_llm_device", "cpu")
        self.tokenizer = None
        self.model = None
        self._is_ready = False
        self._load_lock = threading.Lock()

    def load_model(self) -> bool:
        """Loads the tokenizer and model into memory using lazy import and CPU optimization."""
        if self._is_ready:
            return True

        with self._load_lock:
            if self._is_ready:
                return True

            logger.info(f"Loading local open-source LLM '{self.model_name}' on device '{self.device}'...")
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # Tune CPU threading for Intel Core i7 (4 physical cores) to prevent OpenMP thrashing
                if self.device == "cpu":
                    num_threads = min(4, os.cpu_count() or 4)
                    torch.set_num_threads(num_threads)
                    logger.info(f"Configured PyTorch CPU threads: {num_threads}")

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )

                # Use float32 on CPU for optimal native AVX-512 kernel execution on Intel i7-1065G7
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
                self._is_ready = True
                logger.info(f"Local LLM '{self.model_name}' successfully loaded and ready for inference.")
                return True
            except Exception as e:
                logger.error(f"Failed to load local LLM '{self.model_name}': {e}", exc_info=True)
                return False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """
        Generates tokens in real time using TextIteratorStreamer with inference_mode.
        Yields text chunks as they are generated.
        """
        if not self._is_ready:
            success = self.load_model()
            if not success:
                yield "⚠️ 로컬 오픈소스 LLM 모델을 로드하는 중 오류가 발생했습니다. 서버 로그를 확인해 주세요."
                return

        try:
            import torch
            from transformers import TextIteratorStreamer

            # Apply chat template
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
