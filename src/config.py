"""
Application configuration backward compatibility bridge.
Re-exports AppConfig and config instance from root config.py.
"""
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from config import (
    AppConfig,
    config,
    MIN_TEXT_CHARS_PER_PAGE,
    SCANNED_PAGE_RATIO,
    OCR_MIN_CONFIDENCE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    BASE_DIR
)

__all__ = [
    "AppConfig",
    "config",
    "MIN_TEXT_CHARS_PER_PAGE",
    "SCANNED_PAGE_RATIO",
    "OCR_MIN_CONFIDENCE",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "MIN_CHUNK_SIZE",
    "BASE_DIR"
]
