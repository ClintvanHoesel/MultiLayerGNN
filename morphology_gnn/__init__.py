# This package handles GNN-based molecular property prediction for morphology analysis.

import logging
import os

# Keep the package silent by default (no "No handlers" warning) until a caller
# opts into INFO/DEBUG via configure_logging or the MGN_LOG_LEVEL env var.
logging.getLogger(__name__).addHandler(logging.NullHandler())

from ._logging import configure_logging, get_logger, log_duration  # noqa: E402

# Opt into logging at import time when an env var is set, so that
# `MGN_LOG_LEVEL=DEBUG` alone is enough (no explicit configure_logging() call).
if os.environ.get("MGN_LOG_LEVEL") or os.environ.get("LOG_LEVEL"):
    configure_logging()

__all__ = ["model", "configure_logging", "get_logger", "log_duration"]
__version__ = "0.1.0"
