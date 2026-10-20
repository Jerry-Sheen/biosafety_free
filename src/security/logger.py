"""
Secure logging module.
Ensures sensitive tokens (API keys, passwords, credentials) are never written to logs.
"""
import re
import logging
import sys
from typing import List

# Patterns matching sensitive keys like Google API keys, bearer tokens, AWS secrets
SENSITIVE_PATTERNS: List[re.Pattern] = [
    re.compile(r"AIza[0-9A-Za-z\-_]{30,45}"),          # Google API Key
    re.compile(r"bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE), # Bearer tokens
    re.compile(r"AKIA[0-9A-Z]{16}"),                    # AWS Access Key ID
    re.compile(r"secret[_-]?key\s*[:=]\s*['\"]?([^'\"\s]+)", re.IGNORECASE),
    re.compile(r"password\s*[:=]\s*['\"]?([^'\"\s]+)", re.IGNORECASE),
    re.compile(r"api[_-]?key\s*[:=]\s*['\"]?([^'\"\s]+)", re.IGNORECASE),
]

class SensitiveDataFilter(logging.Filter):
    """Logging filter that scrubs sensitive patterns from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.sanitize(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: (self.sanitize(v) if isinstance(v, str) else v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self.sanitize(arg) if isinstance(arg, str) else arg for arg in record.args)
        return True

    @staticmethod
    def sanitize(text: str) -> str:
        sanitized = text
        for pattern in SENSITIVE_PATTERNS:
            sanitized = pattern.sub("[REDACTED_SECRET]", sanitized)
        return sanitized


def get_secure_logger(name: str = "rag_system") -> logging.Logger:
    """Configures and returns a security-hardened logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"))
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger
