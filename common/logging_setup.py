"""One authoritative logging policy for the whole reproduction pipeline.

Every entry point (the reproduction wrapper, the forecasting runners, the
optimiser searches, the AE-SINDy tools, the KSE simulation and the plotting
scripts) configures terminal output through this single module, so all
components share one format, one component-naming scheme and one stdout/stderr
policy. Library modules keep using ``logging.getLogger(__name__)`` and never
configure logging on import; only entry points call :func:`configure_logging`.

Standard operational log lines look like::

    HH:MM:SS | LEVEL   | COMPONENT  | message

with the standard logging levels only (DEBUG, INFO, WARNING, ERROR, CRITICAL).

The stream policy is deliberately simple: **all** structured operational output
goes to stderr, at every level, while stdout carries only plain presentation --
the interactive menu, ``--list`` and ``--plan`` text, workflow headings, tables
and the formal summaries -- written through :class:`Presentation`. Keeping the
two streams apart means stdout stays usable as presentation (and as pipeable
output) while diagnostics never interleave into it, and no production code needs
a bare ``print``.

The implementation uses only the Python standard library, is safe to call more
than once (it never accumulates duplicate handlers), and never configures global
logging as an import side effect.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "configure_logging",
    "configure_from_args",
    "add_logging_arguments",
    "resolve_logging_settings",
    "get_logger",
    "Presentation",
    "get_presentation",
    "verbosity",
    "is_verbose",
    "is_debug",
    "component_for",
    "LoggingSettings",
    "LEVELS",
]

# --------------------------------------------------------------------------- #
# Formats and constants
# --------------------------------------------------------------------------- #

CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(component)-10s | %(message)s"
CONSOLE_DATEFMT = "%H:%M:%S"

# The file format keeps the same head but adds the logger name and line number,
# which are useful when reading a persisted log after the fact.
FILE_FORMAT = "%(asctime)s | %(levelname)-7s | %(component)-10s | %(name)s:%(lineno)d | %(message)s"
FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
DEFAULT_COMPONENT = "GENERAL"

# Environment variables used to propagate the policy to child processes launched
# by the reproduction wrapper.
ENV_LEVEL = "REPRO_LOG_LEVEL"
ENV_VERBOSE = "REPRO_VERBOSE"
ENV_DEBUG = "REPRO_DEBUG"

# Ordered mapping of logger-name prefix -> short, stable component label. The
# list is scanned in order, so the most specific prefixes must come first. Any
# logger name that matches none of these falls back to the process default
# component (set by the entry point in :func:`configure_logging`).
_COMPONENT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("forecasting.scripts.optimizer_search_ngrc", "OPT-NGRC"),
    ("forecasting.scripts.optimizer_search_rc", "OPT-RC"),
    ("forecasting.scripts.optimizer_search_perc", "OPT-PERC"),
    ("forecasting.scripts.optimizer_search_pinn", "OPT-PINN"),
    ("forecasting.scripts.run_ngrc", "NGRC"),
    ("forecasting.scripts.run_rc", "RC"),
    ("forecasting.scripts.run_perc", "PERC"),
    ("forecasting.scripts.run_pinn", "PINN"),
    ("forecasting.scripts.plot_forecasting_visuals", "PLOTS"),
    ("forecasting.runners.ngrc_runner", "NGRC"),
    ("forecasting.runners.rc_runner", "RC"),
    ("forecasting.runners.perc_runner", "PERC"),
    ("forecasting.runners.pinn_runner", "PINN"),
    ("forecasting.models.ngrc", "NGRC"),
    ("forecasting.models.rc", "RC"),
    ("forecasting.models.perc", "PERC"),
    ("forecasting.models.pinn", "PINN"),
    ("forecasting.visualization", "PLOTS"),
    ("ae_sindy", "AE-SINDy"),
    ("posthoc_sindy", "SINDy"),
    ("kse", "KSE"),
    ("simulation", "KSE"),
    ("repro.validate", "VALIDATE"),
    ("repro", "REPRO"),
    ("py.warnings", "PYTHON"),
)

_PRESENTATION_LOGGER_NAME = "repro.presentation"


# --------------------------------------------------------------------------- #
# Process-wide state (the single, documented logging global)
# --------------------------------------------------------------------------- #


@dataclass
class _State:
    """Process-wide logging state, set once by :func:`configure_logging`.

    ``verbosity`` is 0 for the default view, 1 for ``--verbose`` and 2 for
    ``--debug``. Runners read it to decide how much already-computed detail to
    emit; it never influences any scientific computation.
    """

    verbosity: int = 0
    default_component: str = DEFAULT_COMPONENT
    configured: bool = False


_STATE = _State()


def verbosity() -> int:
    """Return the current verbosity tier (0 normal, 1 verbose, 2 debug)."""
    return _STATE.verbosity


def is_verbose() -> bool:
    """True when ``--verbose`` (or ``--debug``) is in effect."""
    return _STATE.verbosity >= 1


def is_debug() -> bool:
    """True when ``--debug`` is in effect."""
    return _STATE.verbosity >= 2


def component_for(name: str) -> Optional[str]:
    """Return the component label for a logger *name*, or ``None`` if unmapped."""
    for prefix, comp in _COMPONENT_PREFIXES:
        if name == prefix or name.startswith(prefix + "."):
            return comp
    return None


# --------------------------------------------------------------------------- #
# Filters and handlers
# --------------------------------------------------------------------------- #


class _ComponentFilter(logging.Filter):
    """Attach a ``component`` attribute to every record before formatting.

    Priority: an explicit ``component`` (passed via ``extra=``) wins; otherwise
    the logger name is mapped through :data:`_COMPONENT_PREFIXES`; otherwise the
    process default component is used. The filter never drops a record.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if not getattr(record, "component", None):
            record.component = component_for(record.name) or _STATE.default_component
        return True


class _DynamicStreamHandler(logging.StreamHandler):
    """A StreamHandler that always resolves ``sys.stdout``/``sys.stderr`` live.

    Binding to the current attribute on :mod:`sys` (rather than capturing the
    stream at construction time) keeps the handler correct under output
    redirection -- both in production (pipes, Slurm) and in tests that use
    ``contextlib.redirect_stdout``.
    """

    def __init__(self, stream_name: str) -> None:
        self._stream_name = stream_name
        super().__init__()

    @property
    def stream(self):  # type: ignore[override]
        return getattr(sys, self._stream_name)

    @stream.setter
    def stream(self, value) -> None:  # noqa: D401 - StreamHandler assigns in __init__
        # The stream is derived from sys.<name> on every access, so assignments
        # (including the one in StreamHandler.__init__) are intentionally ignored.
        pass


def _clear_managed_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, "_repro_managed", False):
            logger.removeHandler(handler)
            handler.flush()
            handler.close()


def _managed(handler: logging.Handler) -> logging.Handler:
    handler._repro_managed = True  # type: ignore[attr-defined]
    return handler


def _managed_file_handlers(logger: logging.Logger) -> list[logging.FileHandler]:
    """Return the file handlers this module owns on *logger*."""
    return [
        h
        for h in logger.handlers
        if isinstance(h, logging.FileHandler) and getattr(h, "_repro_managed", False)
    ]


def _add_file_handler(root: logging.Logger, log_file: Union[str, Path]) -> logging.FileHandler:
    """Attach one managed DEBUG-level UTF-8 file handler in append mode.

    The parent directory is created here (and only here), because reaching this
    point means a log file was explicitly requested.
    """
    path = Path(log_file)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.addFilter(_ComponentFilter())
    handler.setFormatter(logging.Formatter(FILE_FORMAT, FILE_DATEFMT))
    root.addHandler(_managed(handler))
    return handler


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def configure_logging(
    level: str = "INFO",
    *,
    component: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    debug: bool = False,
    force: bool = False,
) -> logging.Logger:
    """Configure root logging for one process. Safe to call more than once.

    Args:
        level: Standard level name; ``debug=True`` forces ``"DEBUG"``.
        component: Default component label for loggers that do not match a known
            prefix (typically the entry point's own component).
        log_file: Optional path to a persistent UTF-8 log file (append mode). Its
            parent directory is created only when a path is given. When set, the
            file captures full DEBUG detail regardless of the console level.
            On a non-forced reconfiguration a requested file is attached if none
            is present and reused if it is the same path; requesting a *different*
            path raises ValueError rather than silently dropping it or adding a
            second destination.
        verbose: Enable the verbose tier (extra INFO-level detail).
        debug: Enable the debug tier (DEBUG level plus tracebacks).
        force: Recreate handlers even if logging was already configured. Replaced
            handlers are flushed and closed.

    Returns:
        The root logger.

    Raises:
        ValueError: for an unknown *level*, or for a conflicting *log_file* on a
            non-forced reconfiguration.
    """
    level = (level or "INFO").upper()
    if debug:
        level = "DEBUG"
    if level not in LEVELS:
        raise ValueError(f"level must be one of {list(LEVELS)}, got {level!r}.")

    numeric = getattr(logging, level)

    if component:
        _STATE.default_component = component
    _STATE.verbosity = 2 if debug else (1 if verbose else 0)

    root = logging.getLogger()

    _publish_verbosity(root)

    if _STATE.configured and not force:
        # Update the console level in place without stacking new handlers.
        for handler in root.handlers:
            if getattr(handler, "_repro_managed", False) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(numeric)
        if log_file is not None:
            # A requested log file must never be silently dropped. Either attach
            # it, recognise it as the one already attached, or say why not.
            existing = _managed_file_handlers(root)
            wanted = os.path.abspath(str(Path(log_file)))
            if not existing:
                _add_file_handler(root, log_file)
            elif not any(h.baseFilename == wanted for h in existing):
                raise ValueError(
                    f"Logging is already configured to write {existing[0].baseFilename!r}; "
                    f"cannot also add {wanted!r} without replacing the configuration. "
                    "Call configure_logging(..., force=True) to replace it."
                )
        _apply_root_level(root, numeric, log_file, debug)
        _STATE.configured = True
        return root

    _clear_managed_handlers(root)
    _apply_root_level(root, numeric, log_file, debug)

    comp_filter = _ComponentFilter()
    console_formatter = logging.Formatter(CONSOLE_FORMAT, CONSOLE_DATEFMT)

    # ONE structured console handler, on stderr, carrying every level. stdout is
    # reserved entirely for plain presentation, so diagnostics never interleave
    # with menus, tables or the formal summaries.
    stderr_handler = _DynamicStreamHandler("stderr")
    stderr_handler.setLevel(numeric)
    stderr_handler.addFilter(comp_filter)
    stderr_handler.setFormatter(console_formatter)
    root.addHandler(_managed(stderr_handler))

    if log_file is not None:
        _add_file_handler(root, log_file)

    # Route Python warnings through logging (component PYTHON).
    logging.captureWarnings(True)

    _STATE.configured = True
    return root


def _publish_verbosity(root: logging.Logger) -> None:
    """Expose the verbosity tier on the root logger as ``repro_verbosity``.

    Library modules that need the tier (currently only the PINN training-loop
    reporting cadence) can read it with nothing but ``import logging``, so no
    package outside this one has to import :mod:`common` or touch ``sys.path``.
    """
    root.repro_verbosity = _STATE.verbosity  # type: ignore[attr-defined]


def _apply_root_level(
    root: logging.Logger, numeric: int, log_file: Optional[Union[str, Path]], debug: bool
) -> None:
    # A file handler (or --debug) needs DEBUG records to reach the handlers; the
    # per-handler levels then decide what each destination actually shows.
    root.setLevel(logging.DEBUG if (log_file is not None or debug) else numeric)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a standard logger; ``None`` returns the root logger."""
    return logging.getLogger(name) if name else logging.getLogger()


# --------------------------------------------------------------------------- #
# Presentation channel (plain, unprefixed stdout)
# --------------------------------------------------------------------------- #


class Presentation:
    """Plain stdout channel for human-facing, unprefixed presentation output.

    Used for the interactive menu, ``--list``/``--plan`` text, workflow
    headings, tables and the final summaries. Output is emitted through a
    dedicated logging handler (never a bare ``print``) and is independent of the
    configured log level, so ``--list`` still prints under ``--log-level
    WARNING``.
    """

    def __init__(self) -> None:
        self._logger = _presentation_logger()

    def line(self, text: str = "") -> None:
        """Emit one line of presentation text (like ``print(text)``)."""
        self._logger.info("%s", text)

    def blank(self) -> None:
        """Emit a single blank line."""
        self._logger.info("")


def _presentation_logger() -> logging.Logger:
    logger = logging.getLogger(_PRESENTATION_LOGGER_NAME)
    if not getattr(logger, "_repro_presentation_ready", False):
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        handler = _DynamicStreamHandler("stdout")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(_managed(handler))
        logger._repro_presentation_ready = True  # type: ignore[attr-defined]
    return logger


_PRESENTATION_SINGLETON: Optional[Presentation] = None


def get_presentation() -> Presentation:
    """Return the shared :class:`Presentation` instance."""
    global _PRESENTATION_SINGLETON
    if _PRESENTATION_SINGLETON is None:
        _PRESENTATION_SINGLETON = Presentation()
    return _PRESENTATION_SINGLETON


# --------------------------------------------------------------------------- #
# CLI integration
# --------------------------------------------------------------------------- #


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared logging options to an argparse parser.

    ``--verbose``, ``--debug`` and ``--log-level`` are mutually exclusive (a
    conflicting combination makes argparse exit with code 2). ``--log-file`` is
    independent.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--verbose",
        action="store_true",
        help="More frequent INFO-level progress (per-epoch or per-trial updates where "
             "applicable). Split and internal shapes are DEBUG detail; use --debug for those.",
    )
    group.add_argument(
        "--debug",
        action="store_true",
        help="Full DEBUG diagnostics: paths, seeds, shapes, commands and tracebacks.",
    )
    group.add_argument(
        "--log-level",
        type=lambda s: str(s).upper(),
        choices=list(LEVELS),
        default=None,
        metavar="{DEBUG,INFO,WARNING,ERROR,CRITICAL}",
        help="Explicit terminal logging level (case-insensitive).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to a persistent UTF-8 log file (append mode).",
    )


@dataclass
class LoggingSettings:
    """Resolved logging settings, ready to hand to :func:`configure_logging`."""

    level: str = "INFO"
    verbose: bool = False
    debug: bool = False
    log_file: Optional[Path] = None

    def as_child_env(self) -> dict[str, str]:
        """Return env vars that propagate this policy to child processes.

        The log *file* is intentionally not propagated: children write their own
        per-command logs under the run directory, so they never share one file.
        """
        env = {ENV_LEVEL: self.level}
        if self.verbose:
            env[ENV_VERBOSE] = "1"
        if self.debug:
            env[ENV_DEBUG] = "1"
        return env


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_logging_settings(
    args: Optional[argparse.Namespace] = None,
    *,
    config_level: Optional[str] = None,
    default_level: str = "INFO",
) -> LoggingSettings:
    """Resolve the effective logging settings from CLI args, env and config.

    Precedence (highest first): ``--debug`` / ``--verbose`` / ``--log-level``
    from *args*; the ``REPRO_DEBUG`` / ``REPRO_VERBOSE`` / ``REPRO_LOG_LEVEL``
    environment variables (used by the wrapper to propagate to children);
    *config_level* (e.g. a runner's ``logging.level``); then *default_level*.
    """
    debug = bool(getattr(args, "debug", False)) or _env_flag(ENV_DEBUG)
    verbose = bool(getattr(args, "verbose", False)) or _env_flag(ENV_VERBOSE)
    cli_level = getattr(args, "log_level", None)
    log_file = getattr(args, "log_file", None)

    if cli_level:
        level = str(cli_level).upper()
    elif debug:
        level = "DEBUG"
    elif verbose:
        level = "INFO"
    elif os.environ.get(ENV_LEVEL):
        level = os.environ[ENV_LEVEL].upper()
    elif config_level:
        level = str(config_level).upper()
    else:
        level = default_level.upper()

    if level not in LEVELS:
        raise ValueError(f"level must be one of {list(LEVELS)}, got {level!r}.")

    return LoggingSettings(
        level=level,
        verbose=verbose,
        debug=debug,
        log_file=Path(log_file) if log_file is not None else None,
    )


def configure_from_args(
    args: Optional[argparse.Namespace],
    *,
    component: str,
    config_level: Optional[str] = None,
    default_level: str = "INFO",
) -> LoggingSettings:
    """Resolve settings from *args*/env/config and configure logging in one step.

    Returns the resolved :class:`LoggingSettings` so an entry point can, for
    example, propagate them to child processes.
    """
    settings = resolve_logging_settings(
        args, config_level=config_level, default_level=default_level
    )
    configure_logging(
        settings.level,
        component=component,
        log_file=settings.log_file,
        verbose=settings.verbose,
        debug=settings.debug,
        force=True,
    )
    return settings


# The iterative-progress cadence lives with the code that needs it
# (``forecasting.progress``); it reads the tier published by
# :func:`_publish_verbosity`, so no library module imports this package.
