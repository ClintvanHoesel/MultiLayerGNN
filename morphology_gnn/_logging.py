"""Small stdlib ``logging`` helpers for the ``morphology_gnn`` library.

The library is **silent by default** (``WARNING`` level): importing it produces
no output, so notebooks and tests keep their current behaviour. Call
:func:`configure_logging` (or set the ``MGN_LOG_LEVEL`` env var) to opt into
``INFO`` / ``DEBUG`` output for debugging.

Conventions:
    * Every module creates its own logger: ``logger = logging.getLogger(__name__)``
      (the ``morphology_gnn.*`` hierarchy is what :func:`configure_logging`
      configures).
    * Log messages use lazy ``%``-style formatting
      (``logger.debug("x=%s", x)``) so the cost is only paid when the level
      is enabled.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from time import perf_counter

# Handlers attach to the package logger, so child loggers (morphology_gnn.*)
# inherit them. `propagate=False` stops them re-emitting through the root.
_PACKAGE_LOGGER_NAME = "morphology_gnn"

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"

ENV_LEVEL = "MGN_LOG_LEVEL"
ENV_FILE = "MGN_LOG_FILE"

_VALID_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger for ``name`` (defaults to the package logger).

    Library modules should use ``logging.getLogger(__name__)`` directly; this
    helper is a convenience for scripts and ad-hoc debugging.
    """
    return logging.getLogger(name or _PACKAGE_LOGGER_NAME)


def _normalize_level(level: str | int | None) -> int:
    if level is None:
        level = os.environ.get(ENV_LEVEL) or os.environ.get("LOG_LEVEL") or "WARNING"
    if isinstance(level, str):
        level = level.strip().upper()
        if level not in _VALID_LEVELS:
            raise ValueError(
                f"invalid log level {level!r}; choose one of "
                + ", ".join(_VALID_LEVELS)
            )
        return getattr(logging, level)
    return int(level)


def configure_logging(
    level: str | int | None = None,
    log_file: str | None = None,
    *,
    stderr: bool = True,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
) -> logging.Logger:
    """Configure the ``morphology_gnn`` logger tree.

    Safe to call multiple times: the level is always (re)applied, but stderr /
    file handlers are only added once, so existing configuration is never
    duplicated.

    Args:
        level: Threshold (name or int). Defaults to the ``MGN_LOG_LEVEL`` (or
            ``LOG_LEVEL``) env var, else ``WARNING``.
        log_file: Optional path; adds a ``FileHandler`` alongside stderr.
            Defaults to the ``MGN_LOG_FILE`` env var.
        stderr: Whether to log to ``sys.stderr`` (default True).
        fmt / datefmt: Formatter used for the handlers.

    Returns:
        The package logger (``logging.getLogger("morphology_gnn")``).
    """
    pkg_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    pkg_logger.setLevel(_normalize_level(level))
    pkg_logger.propagate = False

    formatter = logging.Formatter(fmt, datefmt=datefmt)
    has_stderr = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in pkg_logger.handlers
    )
    has_file = any(isinstance(h, logging.FileHandler) for h in pkg_logger.handlers)

    if stderr and not has_stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        pkg_logger.addHandler(handler)

    file_path = log_file if log_file is not None else os.environ.get(ENV_FILE)
    if file_path and not has_file:
        file_path = os.fspath(file_path)
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setFormatter(formatter)
        pkg_logger.addHandler(fh)

    return pkg_logger


@contextmanager
def log_duration(
    logger: logging.Logger,
    msg: str,
    *args,
    level: int = logging.DEBUG,
):
    """Time a block and log ``msg`` (with optional ``%``-style args) on exit.

    Usage::

        with log_duration(logger, "radius graph for n=%d", n):
            edge_index = radius_graph_pbc(...)
    """
    start = perf_counter()
    try:
        yield
    finally:
        logger.log(level, msg + " (%.3fs)", *args, perf_counter() - start)
