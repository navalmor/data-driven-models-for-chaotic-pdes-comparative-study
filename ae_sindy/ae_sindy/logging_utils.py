"""AE-SINDy logging entry point.

This is a thin adapter over the repository-wide logging policy in
``common.logging_setup``. It keeps the AE-SINDy call signature
(``configure_logging(level, log_file, reset=...)``) that the AE-SINDy scripts
already use, while delegating the actual console format, component naming
(``AE-SINDy``) and stdout/stderr policy to the one shared implementation, so no
formatting logic is duplicated across packages.

If ``common`` cannot be imported (for example the package is used in isolation
without the repository root on ``sys.path``), a self-contained fallback keeps the
original behaviour so importing or configuring never fails.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_PACKAGE_LOGGER_NAME = "ae_sindy"
_COMPONENT = "AE-SINDy"


def configure_logging(
    level: str = "INFO", log_file: Optional[str | Path] = None, *, reset: bool = False
) -> logging.Logger:
    """Configure logging for an AE-SINDy run, delegating to the shared policy."""
    package_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)

    try:
        from common.logging_setup import configure_logging as _shared_configure
    except ImportError:
        # Only an unavailable ``common`` package justifies the fallback. Any other
        # exception is a real defect in the canonical implementation and must
        # surface rather than be masked by a reduced second policy.
        return _fallback_configure(level, log_file, reset=reset)

    # The shared policy attaches handlers to the root logger and injects the
    # component via a filter, so the package logger must propagate and must not
    # carry its own duplicate handlers.
    if reset:
        for handler in list(package_logger.handlers):
            package_logger.removeHandler(handler)
            handler.close()
    package_logger.propagate = True
    package_logger.setLevel(logging.NOTSET)

    _shared_configure(level, component=_COMPONENT, log_file=log_file, force=reset)
    return package_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if name is None:
        return logging.getLogger(_PACKAGE_LOGGER_NAME)
    return logging.getLogger(f"{_PACKAGE_LOGGER_NAME}.{name}")


def _fallback_configure(
    level: str = "INFO", log_file: Optional[str | Path] = None, *, reset: bool = False
) -> logging.Logger:
    """Emergency standalone configuration, used only when ``common`` is absent.

    This exists so the AE-SINDy package still logs sensibly when it is used on its
    own, detached from this repository. It mirrors the canonical policy -- the same
    component-aware line shape, structured output on stderr, and an append-mode
    UTF-8 file at full DEBUG detail -- but deliberately stays minimal. Inside this
    repository the canonical implementation is always importable, so this path is
    not exercised.
    """
    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if reset:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.flush()
            handler.close()

    if not logger.handlers:
        # stderr, matching the canonical stream policy: stdout stays presentation-only.
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        console_handler.setFormatter(
            logging.Formatter(
                f"%(asctime)s | %(levelname)-7s | {_COMPONENT:<10} | %(message)s", "%H:%M:%S"
            )
        )
        logger.addHandler(console_handler)

        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter(
                    f"%(asctime)s | %(levelname)-7s | {_COMPONENT:<10} | "
                    "%(name)s:%(lineno)d | %(message)s",
                    "%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(file_handler)

    return logger
