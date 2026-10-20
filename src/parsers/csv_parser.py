from .record import DocumentRecord

class CSVParser:
    """Parse CSV files with automatic encoding detection.

    Returns a ``DocumentRecord`` where ``text`` contains the raw CSV rows
    (joined by ``\n``) and ``metadata`` stores encoding, column count, etc.
    """

    def parse(self, file_path):
        encodings = ("utf-8-sig", "utf-8", "cp949", "euc-kr")
        chosen_enc = None
        rows = []
        for enc in encodings:
            try:
                with open(file_path, encoding=enc, newline="") as f:
                    import csv
                    reader = csv.reader(f)
                    rows = list(reader)
                chosen_enc = enc
                break
            except UnicodeDecodeError:
                continue
        if chosen_enc is None:
            # fallback: replace errors
            with open(file_path, encoding="utf-8", errors="replace", newline="") as f:
                import csv
                rows = list(csv.reader(f))
            chosen_enc = "utf-8-replace"

        # Preserve each row as a single line (do not split fields)
        text = "\n".join([",".join(r) for r in rows])
        metadata = {
            "source": file_path.name,
            "content_type": "body",
            "extractor": "csv",
            "encoding": chosen_enc,
            "row_count": len(rows),
            "column_count": max((len(r) for r in rows), default=0),
        }
        return DocumentRecord(text=text, metadata=metadata)
