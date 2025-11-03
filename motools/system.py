"""System-level utilities for motools."""

import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


def load_dotenv() -> None:
    """Load environment variables from .env file if it exists.

    Searches for .env file in current directory and parent directories.
    This is called automatically when importing motools to ensure
    environment variables are available throughout the package.
    """
    try:
        from dotenv import load_dotenv as _load_dotenv

        # Find .env file in current or parent directories
        current = Path.cwd()
        for directory in [current, *current.parents]:
            env_file = directory / ".env"
            if env_file.exists():
                _load_dotenv(env_file)
                logger.info(f"Loaded environment variables from {env_file}")
                return

        # If no .env found, just try loading from project root directory
        env_file = Path(__file__).parent.parent / ".env"
        _load_dotenv(env_file)
        logger.info(f"Loaded environment variables from {env_file}")

    except ImportError:
        # python-dotenv not installed, skip
        logger.warning("python-dotenv not installed, skipping environment variable loading")


@dataclass
class EnvConfig:
    """Environment variable configuration.

    Attributes:
        required: List of required environment variables
        optional: List of optional environment variables with descriptions
    """

    required: list[str]
    optional: dict[str, str] | None = None  # var_name -> description

    def __post_init__(self) -> None:
        """Initialize optional dict if None."""
        if self.optional is None:
            self.optional = {}


class EnvValidationError(Exception):
    """Raised when required environment variables are missing."""

    pass


def validate_env(config: EnvConfig) -> None:
    """Validate that required environment variables are set.

    Args:
        config: Environment configuration

    Raises:
        EnvValidationError: If required environment variables are missing
    """
    missing = []

    for var_name in config.required:
        if not os.getenv(var_name):
            missing.append(var_name)

    if missing:
        error_msg = "Missing required environment variables:\n"
        for var_name in missing:
            error_msg += f"  - {var_name}\n"

        error_msg += "\nPlease set these variables and try again."

        raise EnvValidationError(error_msg)


# Load environment variables on import
load_dotenv()
