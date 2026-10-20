# src/cli/user.py
"""Command‑line interface for the RAG pipeline.

Provides a convenient entry point for users to process a single document or an
entire directory.  The CLI mirrors the functionality requested by the user and
exposes the progress‑bar implemented in :pymod:`src.pipeline`.

Typical usage::

    python -m src.cli.user --input path/to/file.pdf
    python -m src.cli.user --recursive path/to/folder --force
    python -m src.cli.user --list-models
    python -m src.cli.user --check-gemini
    python -m src.cli.user --check-ocr

All commands print human‑readable messages (in Korean/English) and log JSON
entries via the structured logger.
"""

import argparse
import sys
from pathlib import Path

from src.pipeline import process_file
from src.data.ingest import ingest_directory
from src.rag.gemini_client import GeminiClient
from src.rag.engine import check_ocr_installed  # assumed utility function
from src.utils.structured_logger import get_structured_logger

logger = get_structured_logger(__name__)


def list_models() -> None:
    """Print the list of Gemini models that support `generateContent`."""
    client = GeminiClient()
    models = client.available_generate_content_models()
    if not models:
        print("No generateContent‑compatible models found.")
        return
    print("Available Gemini models (generateContent):")
    for m in models:
        print(f" - {m}")


def check_gemini() -> None:
    """Perform a lightweight health‑check of the Gemini API.

    It attempts to retrieve the model list.  Success indicates that the API key
    is valid and the service is reachable.
    """
    try:
        client = GeminiClient()
        models = client.available_generate_content_models()
        print(f"Gemini API reachable – {len(models)} generateContent models available.")
    except Exception as exc:
        logger.error("Gemini health‑check failed", exc_info=exc)
        print("Gemini health‑check failed – see logs for details.")
        sys.exit(1)


def check_ocr() -> None:
    """Verify that the OCR engine (Tesseract) is installed and callable."""
    try:
        if check_ocr_installed():
            print("OCR engine detected and functional.")
        else:
            print("OCR engine not detected – ensure Tesseract is installed and PATH is set.")
            sys.exit(1)
    except Exception as exc:
        logger.error("OCR check raised an exception", exc_info=exc)
        print("OCR check failed – see logs for details.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG pipeline CLI – process documents with progress bar",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Path to a single document to process")
    group.add_argument("--recursive", type=str, help="Path to a directory – ingest all supported files")
    parser.add_argument("--force", action="store_true", help="Force re‑processing even for duplicate files")
    parser.add_argument("--check-ocr", action="store_true", help="Validate OCR engine installation and exit")
    parser.add_argument("--check-gemini", action="store_true", help="Validate Gemini API connectivity and exit")
    parser.add_argument("--list-models", action="store_true", help="List available Gemini generateContent models and exit")
    args = parser.parse_args()

    if args.check_ocr:
        check_ocr()
        return
    if args.check_gemini:
        check_gemini()
        return
    if args.list_models:
        list_models()
        return

    if args.input:
        file_path = Path(args.input)
        if not file_path.is_file():
            logger.error("Input file not found", extra={"error_type": "FILE_READ_ERROR", "source": str(file_path)})
            print(f"Error: {file_path} does not exist.")
            sys.exit(1)
        process_file(file_path, force=args.force)
    elif args.recursive:
        dir_path = Path(args.recursive)
        if not dir_path.is_dir():
            logger.error("Directory not found", extra={"error_type": "FILE_READ_ERROR", "source": str(dir_path)})
            print(f"Error: {dir_path} is not a directory.")
            sys.exit(1)
        ingest_directory(dir_path, force=args.force)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
