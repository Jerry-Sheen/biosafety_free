"""
Parsers package for biosafety document ingestion.
Provides unified DocumentRecord data structure and specialized document parsers.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class DocumentRecord:
    """Standard internal data structure returned by all parsers."""
    text: str
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "raw_text": self.raw_text or self.text,
            "metadata": self.metadata
        }
