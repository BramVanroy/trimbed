"""Logging helpers shared by the library and the scripts.

`configure_logging()` should be called once at the start of a script to set up the package logger,
e.g. to set it to quiet mode.
"""

from __future__ import annotations

import logging


LOGGER_NAME = "trimbed"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger, or a child of it.

    Args:
        name: Optional dotted suffix, typically `__name__`.

    Returns:
        The `trimbed` logger or the named child logger.
    """
    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    suffix = name.removeprefix(f"{LOGGER_NAME}.")
    return logging.getLogger(f"{LOGGER_NAME}.{suffix}")


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Attach a single stderr handler to the package logger.

    Repeated calls replace the previous handler rather than stacking.

    Args:
        verbose: Emit DEBUG-level records.
        quiet: Emit only warnings and above. Ignored when `verbose` is set.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)-8s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO)
    logger.propagate = False
