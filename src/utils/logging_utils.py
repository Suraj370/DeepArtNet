"""Logging configuration for DeepArtNet.

Sets up a root logger that writes to both the console (stdout) and an
optional log file.  All other modules in the project obtain their logger
via ``logging.getLogger(__name__)`` and inherit this configuration.
"""

from __future__ import annotations

import logging
import pathlib
import sys
from typing import Optional


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str | pathlib.Path] = None,
) -> logging.Logger:
    """Configure the root logger with a console handler and optional file handler.

    Should be called once at the start of each script (``train.py``,
    ``evaluate.py``, ``inference.py``).  Subsequent calls are idempotent —
    existing handlers are cleared before new ones are added.

    Args:
        log_level: Minimum severity level to capture (default ``logging.INFO``).
        log_file: If provided, log messages are also written to this path.
            Parent directories are created automatically.

    Returns:
        The configured root :class:`logging.Logger`.

    Example::

        logger = setup_logging(log_file="outputs/logs/train.log")
        logger.info("Training started")
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Optional file handler
    if log_file is not None:
        log_path = pathlib.Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (shorthand for ``logging.getLogger(name)``).

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
