"""
Application Configuration and Constants.
Centralizes paths, models, thresholds, and execution flags.
Compatible with both legacy src/ and new root modular pipeline.
"""
import os
import shutil
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

# Load .env file
load_dotenv(override=False)

# Force stable ChromaDB Python API to avoid RustBindingsAPI attribute errors

BASE_DIR = Path(__file__).resolve().parent

class AppConfig(BaseModel):
    """Central configuration for RAG ingestion pipeline and applications."""

    # 1. Environment & Network (Dedicated 8600 series ports)
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    app_host: str = Field(default_factory=lambda: os.getenv("APP_HOST", "0.0.0.0"))
    admin_port: int = Field(default_factory=lambda: int(os.getenv("ADMIN_PORT", "8601")))
    user_port: int = Field(default_factory=lambda: int(os.getenv("USER_PORT", "8602")))
    backend_port: int = Field(default_factory=lambda: int(os.getenv("BACKEND_PORT", "8600")))
    backend_url: str = Field(default_factory=lambda: os.getenv("BACKEND_URL", "http://localhost:8600"))

    # 2. PDF & Document State Thresholds
    min_text_chars_per_page: int = Field(default_factory=lambda: int(os.getenv("MIN_TEXT_CHARS_PER_PAGE", "80")))
    scanned_page_ratio: float = Field(default_factory=lambda: float(os.getenv("SCANNED_PAGE_RATIO", "0.7")))
    ocr_min_confidence: float = Field(default_factory=lambda: float(os.getenv("OCR_MIN_CONFIDENCE", "55.0")))

    # 3. Chunking Configuration
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "120")))
    min_chunk_size: int = Field(default_factory=lambda: int(os.getenv("MIN_CHUNK_SIZE", "100")))

    # 4. OCR Engine Configuration
    tesseract_cmd: Optional[str] = Field(default_factory=lambda: os.getenv("TESSERACT_CMD", None))
    ocr_language: str = Field(default_factory=lambda: os.getenv("OCR_LANGUAGE", "kor+eng"))
    ocr_dpi: int = Field(default_factory=lambda: int(os.getenv("OCR_DPI", "300")))
    prefer_rapidocr: bool = Field(default_factory=lambda: os.getenv("PREFER_RAPIDOCR", "true").lower() == "true")

    # 5. LLM & Embedding Settings
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "free"))
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"))
    gemini_embed_model: str = Field(default_factory=lambda: os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004"))

    # Fixed local embedding model (CRITICAL: shared across admin.py, user.py, and backend).
    local_embed_model: str = Field(default_factory=lambda: os.getenv(
        "LOCAL_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))

    # Local CPU Open-Source LLM Settings (Free mode)
    local_llm_model: str = Field(default_factory=lambda: os.getenv(
        "LOCAL_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    local_llm_device: str = Field(default_factory=lambda: os.getenv("LOCAL_LLM_DEVICE", "cpu"))
    local_llm_gguf_repo: str = Field(default_factory=lambda: os.getenv(
        "LOCAL_LLM_GGUF_REPO", "Qwen/Qwen2.5-1.5B-Instruct-GGUF"))
    local_llm_gguf_file: str = Field(default_factory=lambda: os.getenv(
        "LOCAL_LLM_GGUF_FILE", "qwen2.5-1.5b-instruct-q4_k_m.gguf"))
    local_llm_n_ctx: int = Field(default_factory=lambda: int(os.getenv("LOCAL_LLM_N_CTX", "16384")))

    # OpenAI Settings (Paid mode BYOK)
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    openai_embed_model: str = Field(default_factory=lambda: os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002"))

    # 6. Directories & Paths
    chroma_persist_dir: Path = Field(
        default_factory=lambda: BASE_DIR / os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma_db")
    )
    chroma_collection_name: str = Field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
    )
    upload_dir: Path = Field(
        default_factory=lambda: BASE_DIR / os.getenv("UPLOAD_DIRECTORY", "data/uploads")
    )
    data_input_dir: Path = Field(default_factory=lambda: BASE_DIR / "data" / "input")
    data_extracted_dir: Path = Field(default_factory=lambda: BASE_DIR / "data" / "extracted")
    data_failed_dir: Path = Field(default_factory=lambda: BASE_DIR / "data" / "failed")
    logs_dir: Path = Field(default_factory=lambda: BASE_DIR / "logs")
    log_file: Path = Field(default_factory=lambda: BASE_DIR / "logs" / "ingestion.log")

    # Security & Admin Access PIN
    admin_access_pin: str = Field(default_factory=lambda: os.getenv("ADMIN_ACCESS_PIN", "admin1234"))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def init_directories(self):
        """Ensures all necessary data and log directories exist."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.data_input_dir.mkdir(parents=True, exist_ok=True)
        self.data_extracted_dir.mkdir(parents=True, exist_ok=True)
        self.data_failed_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def get_tesseract_path(self) -> Optional[str]:
        """
        Resolves tesseract executable path with prioritized fallback.
        """
        if self.tesseract_cmd and Path(self.tesseract_cmd).is_file():
            return str(Path(self.tesseract_cmd).resolve())

        env_cmd = os.getenv("TESSERACT_CMD")
        if env_cmd and Path(env_cmd).is_file():
            return str(Path(env_cmd).resolve())

        windows_defaults = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        for p in windows_defaults:
            if Path(p).is_file():
                return str(Path(p).resolve())

        which_path = shutil.which("tesseract")
        if which_path:
            return which_path

        return None


config = AppConfig()
config.init_directories()

# Backward compatibility alias for config values
MIN_TEXT_CHARS_PER_PAGE = config.min_text_chars_per_page
SCANNED_PAGE_RATIO = config.scanned_page_ratio
OCR_MIN_CONFIDENCE = config.ocr_min_confidence
CHUNK_SIZE = config.chunk_size
CHUNK_OVERLAP = config.chunk_overlap
MIN_CHUNK_SIZE = config.min_chunk_size
