"""
Secrets Manager module.
Handles loading secrets securely across Development, Testing, and Production environments.
Supports Docker Secrets, AWS Secrets Manager, and local .env files.
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from src.security.logger import get_secure_logger

logger = get_secure_logger("secrets_manager")


class SecretsManager:
    """Manages secret retrieval according to application environment and least-privilege standards."""

    def __init__(self):
        self.app_env = os.getenv("APP_ENV", "development").lower()
        self._cached_secrets: Dict[str, str] = {}
        self._initialize()

    def _initialize(self):
        """Initializes secrets based on APP_ENV."""
        logger.info(f"Initializing Secrets Manager for environment: {self.app_env}")

        if self.app_env in ("development", "testing"):
            # Load from .env file in dev/test
            env_path = Path(__file__).resolve().parent.parent.parent / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=False)
                logger.info(f"Loaded local environment variables from {env_path.name}")
            else:
                logger.warning(".env file not found. Falling back to environment variables.")
        elif self.app_env == "production":
            # Production: Try Docker Secrets or AWS Secrets Manager first
            self._load_production_secrets()

    def _load_production_secrets(self):
        """Attempts to load secrets from Docker Secrets or Cloud Secrets Manager."""
        # 1. Docker Secrets inspection (/run/secrets/*)
        docker_secrets_dir = Path("/run/secrets")
        if docker_secrets_dir.exists() and docker_secrets_dir.is_dir():
            for secret_file in docker_secrets_dir.iterdir():
                if secret_file.is_file():
                    try:
                        secret_val = secret_file.read_text(encoding="utf-8").strip()
                        self._cached_secrets[secret_file.name] = secret_val
                        logger.info(f"Loaded secret '{secret_file.name}' from Docker Secrets.")
                    except Exception as e:
                        logger.error(f"Failed to read Docker secret {secret_file.name}: {e}")

        # 2. AWS Secrets Manager (if AWS_SECRET_NAME is set and boto3 available)
        aws_secret_name = os.getenv("AWS_SECRET_NAME")
        if aws_secret_name:
            self._load_aws_secrets(aws_secret_name)

    def _load_aws_secrets(self, secret_name: str):
        """Fetches secrets from AWS Secrets Manager using IAM Role / default credentials."""
        try:
            import boto3
            from botocore.exceptions import ClientError

            region_name = os.getenv("AWS_REGION", "ap-northeast-2")
            logger.info(f"Connecting to AWS Secrets Manager in region {region_name} for secret: {secret_name}")

            # Uses IAM Role attached to EC2 / ECS / EKS (No hardcoded credentials)
            client = boto3.client("secretsmanager", region_name=region_name)
            response = client.get_secret_value(SecretId=secret_name)

            if "SecretString" in response:
                secret_data = json.loads(response["SecretString"])
                if isinstance(secret_data, dict):
                    self._cached_secrets.update(secret_data)
                    logger.info("Successfully loaded secrets from AWS Secrets Manager.")
        except ImportError:
            logger.warning("boto3 is not installed; skipping AWS Secrets Manager lookup.")
        except Exception as e:
            logger.error(f"Failed to fetch secrets from AWS Secrets Manager: {e}")

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieves a secret by key.
        Checks cached cloud/docker secrets first, then OS environment variables.
        """
        if key in self._cached_secrets:
            return self._cached_secrets[key]

        val = os.getenv(key, default)
        return val

    def mask_secret(self, secret: Optional[str]) -> str:
        """Returns a masked version of the secret for safe UI display (e.g. AIza...****)."""
        if not secret or len(secret) < 8:
            return "********" if secret else "NOT_CONFIGURED"
        return f"{secret[:6]}...{secret[-4:]}"


def sanitize_api_key(key: Optional[str]) -> str:
    """
    Sanitizes API keys by:
    1. Normalizing unicode characters (NFKC).
    2. Stripping quotes, whitespace, and invisible unicode characters (zero-width spaces, BOM, etc.).
    3. Normalizing unicode dashes/hyphens to ASCII hyphen (-).
    4. Stripping accidentally copied variable prefixes (e.g. 'OPENAI_API_KEY=', 'Bearer ').
    5. Ensuring the key is strictly ASCII to prevent 'ascii' codec can't encode errors in HTTP headers.
    """
    if not key:
        return ""
    import unicodedata
    import re

    # 1. Normalize unicode
    cleaned = unicodedata.normalize("NFKC", str(key)).strip()

    # 2. Strip surrounding quotes & whitespace
    cleaned = cleaned.strip("'\"`“”‘’ \t\r\n")

    # 3. Normalize various unicode dashes to standard ASCII hyphen
    dash_chars = ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212", "\uFE58", "\uFE63", "\uFF0D"]
    for d in dash_chars:
        cleaned = cleaned.replace(d, "-")

    # 4. Remove invisible unicode characters (zero-width spaces, BOM, non-breaking spaces)
    invisible_chars = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u00a0", "\u3000"]
    for inv in invisible_chars:
        cleaned = cleaned.replace(inv, "")

    # 5. Strip accidentally included variable assignment or header prefixes
    prefixes = [
        "openai_api_key=", "openai_api_key:", "openai_api_key ", "openai_api_key",
        "gemini_api_key=", "gemini_api_key:", "gemini_api_key ", "gemini_api_key",
        "api_key=", "api_key:", "api_key ", "bearer ", "키=", "키:", "키 "
    ]
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break

    # 6. Re-strip quotes in case they were inside the prefix
    cleaned = cleaned.strip("'\"`“”‘’ ")

    # 7. If non-ASCII characters remain (e.g. accidentally pasted Korean characters or symbols):
    if not cleaned.isascii():
        # Check if an actual valid key (e.g., sk-...) is embedded inside
        match = re.search(r"sk-[a-zA-Z0-9_\-]{20,}", cleaned)
        if match:
            cleaned = match.group(0)
        else:
            # Otherwise, strip any non-ASCII characters
            cleaned = re.sub(r"[^\x21-\x7E]", "", cleaned)

    return cleaned.strip()


# Global Singleton instance
secrets_manager = SecretsManager()
