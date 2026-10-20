from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DocumentRecord:
    """Canonical container for parsed document output.

    Attributes
    ----------
    text: str
        Full extracted textual content (raw or cleaned).
    metadata: Dict[str, Any]
        Arbitrary key/value pairs describing source, parser, page info, etc.
    """
    text: str
    metadata: Dict[str, Any]
