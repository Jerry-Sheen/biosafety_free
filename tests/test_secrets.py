"""
Unit tests for Secrets Manager and Sensitive Data Logger.
"""
from src.security.secrets_manager import SecretsManager
from src.security.logger import SensitiveDataFilter


def test_mask_secret():
    sm = SecretsManager()
    assert sm.mask_secret("AIzaSyDummySecretKey1234567890") == "AIzaSy...7890"
    assert sm.mask_secret("short") == "********"
    assert sm.mask_secret(None) == "NOT_CONFIGURED"


def test_sensitive_data_filter():
    filt = SensitiveDataFilter()
    raw_message = "Connecting with key AIzaSyDUMMYKEY12345678901234567890123"
    sanitized = filt.sanitize(raw_message)
    assert "[REDACTED_SECRET]" in sanitized
    assert "AIzaSy" not in sanitized
