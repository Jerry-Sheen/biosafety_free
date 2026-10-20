"""
CSV Document Parser for Biosafety RAG Pipeline.
Supports multi-encoding detection (UTF-8, CP949, EUC-KR), table & record preservation,
and returns structured DocumentRecord representations.
"""
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from config import config
from parsers import DocumentRecord
from pipeline.logger import logger, IngestionErrorType, log_pipeline_event


class CSVParser:
    """Parses CSV documents with automatic Korean encoding detection and record structure preservation."""

    ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]

    @classmethod
    def parse(cls, file_path: Path, rows_per_chunk: int = 15) -> List[DocumentRecord]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_bytes = file_path.read_bytes()
        doc_hash = hashlib.sha256(file_bytes).hexdigest()
        source_name = file_path.name
        source_path = str(file_path.resolve())

        df = None
        detected_enc = None

        for enc in cls.ENCODINGS_TO_TRY:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                detected_enc = enc
                logger.info(f"Successfully loaded CSV '{source_name}' using encoding '{enc}'.")
                break
            except Exception:
                continue

        if df is None:
            # Fallback with error replacement
            try:
                df = pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace")
                detected_enc = "utf-8_replace"
            except Exception as e:
                log_pipeline_event(
                    filename=source_name,
                    parser="csv_parser",
                    error_type=IngestionErrorType.FILE_READ_ERROR,
                    message=f"Failed to read CSV with all candidate encodings: {e}",
                    status="failed"
                )
                raise

        # Clean NaN rows/columns
        df = df.dropna(how="all")
        columns = [str(c) for c in df.columns]
        total_rows = len(df)

        records: List[DocumentRecord] = []

        # If small CSV, emit single comprehensive markdown table
        if total_rows <= 30:
            md_table = df.to_markdown(index=False)
            table_text = f"[표 시작]\n{md_table}\n[표 종료]"
            meta = {
                "source": source_name,
                "source_path": source_path,
                "page": 1,
                "content_type": "table",
                "table_id": f"{Path(source_name).stem}_table_full",
                "rows": total_rows,
                "columns": len(columns),
                "column_names": columns,
                "encoding": detected_enc,
                "document_hash": doc_hash,
                "extractor": "csv_table"
            }
            records.append(DocumentRecord(text=table_text, raw_text=md_table, metadata=meta))
        else:
            # Group rows by logical batches without breaking individual records
            for start_idx in range(0, total_rows, rows_per_chunk):
                batch_df = df.iloc[start_idx : start_idx + rows_per_chunk]
                md_table = batch_df.to_markdown(index=False)
                table_text = f"[표 시작 (행 {start_idx + 1}~{min(total_rows, start_idx + rows_per_chunk)})]\n{md_table}\n[표 종료]"
                page_idx = (start_idx // rows_per_chunk) + 1
                meta = {
                    "source": source_name,
                    "source_path": source_path,
                    "page": page_idx,
                    "content_type": "table",
                    "table_id": f"{Path(source_name).stem}_rows_{start_idx + 1}_{min(total_rows, start_idx + rows_per_chunk)}",
                    "rows": len(batch_df),
                    "columns": len(columns),
                    "column_names": columns,
                    "encoding": detected_enc,
                    "document_hash": doc_hash,
                    "extractor": "csv_table"
                }
                records.append(DocumentRecord(text=table_text, raw_text=md_table, metadata=meta))

        return records
