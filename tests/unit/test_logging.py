"""Tests for motools logging configuration."""

import os
import sys
from io import StringIO
from unittest.mock import patch


def test_setup_logging_respects_env_var():
    """Test that setup_logging reads MOTOOLS_LOGGING_LEVEL from environment."""
    # Need to reload motools module to pick up env var changes

    # Set env var before importing
    with patch.dict(os.environ, {"MOTOOLS_LOGGING_LEVEL": "DEBUG"}):
        # Remove module if already imported
        if "motools" in sys.modules:
            del sys.modules["motools"]

        import motools

        # Verify logger is configured
        assert motools.logger is not None

        # Test that debug messages are logged
        stderr_capture = StringIO()
        with patch("sys.stderr", stderr_capture):
            motools.setup_logging()  # Reconfigure with env var
            motools.logger.debug("test debug message")

        output = stderr_capture.getvalue()
        assert "test debug message" in output
        assert "DEBUG" in output


def test_setup_logging_explicit_level_overrides_env():
    """Test that explicit level parameter overrides env var."""

    with patch.dict(os.environ, {"MOTOOLS_LOGGING_LEVEL": "DEBUG"}):
        if "motools" in sys.modules:
            del sys.modules["motools"]

        import motools

        # Explicitly set to ERROR - should override env var
        stderr_capture = StringIO()
        with patch("sys.stderr", stderr_capture):
            motools.setup_logging(level="ERROR")
            motools.logger.debug("debug should not appear")
            motools.logger.info("info should not appear")
            motools.logger.warning("warning should not appear")
            motools.logger.error("error should appear")

        output = stderr_capture.getvalue()
        assert "debug should not appear" not in output
        assert "info should not appear" not in output
        assert "warning should not appear" not in output
        assert "error should appear" in output


def test_setup_logging_default_level():
    """Test that default logging level is INFO when env var not set."""

    # Ensure env var is not set
    with patch.dict(os.environ, {}, clear=False):
        if "MOTOOLS_LOGGING_LEVEL" in os.environ:
            del os.environ["MOTOOLS_LOGGING_LEVEL"]

        if "motools" in sys.modules:
            del sys.modules["motools"]

        import motools

        stderr_capture = StringIO()
        with patch("sys.stderr", stderr_capture):
            motools.setup_logging()
            motools.logger.debug("debug should not appear")
            motools.logger.info("info should appear")

        output = stderr_capture.getvalue()
        assert "debug should not appear" not in output
        assert "info should appear" in output
