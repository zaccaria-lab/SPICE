"""Logging set-up for SPICE.

initialize_logger : creates the shared 'SPICE' logger, writing to stderr and, if there is an output directory, also to
                    'spice.log' in it. With `verbose`, it shows TRACE messages and above, otherwise INFO and above.
ColourFormatter   : makes each line on stderr bold and coloured by its level (`LEVEL_COLOURS`), on a terminal or in
                    Jupyter (`use_colour`), unless the NO_COLOR environment variable is set. 'spice.log' is never
                    coloured.
close_handlers    : removes and closes the logger's handlers, so that a log file is not left open.
addLoggingLevel   : adds a custom logging level and its logger method. SPICE uses it for TRACE (between DEBUG and
                    INFO, for progress messages, `logger.trace(...)`) and STEP (between INFO and WARNING, for the
                    start of each step, `logger.step(...)`).
"""

import logging
import sys
import os
from typing import Optional

LOG_FORMAT = '%(name)s %(asctime)s | %(levelname)s | %(filename)s >>> %(message)s'
# ANSI codes (bold and a colour) for whole log lines on a terminal or in Jupyter
LEVEL_COLOURS = {'TRACE': '\033[1;36m', 'DEBUG': '\033[1;34m', 'INFO': '\033[1;32m', 'STEP': '\033[1;35m',
                 'WARNING': '\033[1;91m', 'ERROR': '\033[1;31m', 'CRITICAL': '\033[1;31m'}


class ColourFormatter(logging.Formatter):
    """Log formatter that makes each whole line bold and coloured by its level (`LEVEL_COLOURS`), for output to a
    terminal or Jupyter."""

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        if record.levelname in LEVEL_COLOURS:
            return f'{LEVEL_COLOURS[record.levelname]}{line}\033[0m'
        return line


def use_colour(stream) -> bool:
    """Return True if log output to `stream` should be coloured: a terminal or a Jupyter cell, unless NO_COLOR is
    set."""
    if os.environ.get('NO_COLOR'):
        return False
    return stream.isatty() or type(stream).__module__.startswith('ipykernel')


def close_handlers(logger: logging.Logger) -> None:
    """Remove the logger's handlers and close them, so that a log file is not left open."""
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def initialize_logger(verbose: bool, output_dir: Optional[str],
                      mode: str = 'w') -> tuple[logging.Logger, logging.Handler]:
    """Set up the shared 'SPICE' logger, writing to stderr and, if `output_dir` is given, to 'spice.log' in it.

    Any handlers from an earlier call are removed and closed first, so each call replaces where the log goes. Lines
    on stderr are bold and coloured by level on a terminal or in Jupyter, unless the NO_COLOR environment variable is
    set. The TRACE and STEP levels are added with `addLoggingLevel` if they don't exist yet.

    Parameters
    ----------
    verbose : bool
        If True, log TRACE messages and above; otherwise INFO and above.
    output_dir : str, optional
        Directory to also write 'spice.log' to (never coloured). If None, log to stderr only.
    mode : str, default 'w'
        File mode for 'spice.log': 'w' overwrites it, 'a' appends to it. Not used when logging to stderr.

    Returns
    -------
    logger : logging.Logger
        The 'SPICE' logger.
    logfile : logging.Handler
        The 'spice.log' handler (a `logging.FileHandler`) if `output_dir` is given, otherwise the stderr handler (a
        `logging.StreamHandler`); some callers remove it again with `logger.removeHandler(logfile)`.
    """
    logger = logging.getLogger('SPICE')
    close_handlers(logger)
    formatter = ColourFormatter if use_colour(sys.stderr) else logging.Formatter
    logfile = logging.StreamHandler(sys.stderr)
    logfile.setFormatter(formatter(LOG_FORMAT))
    logger.addHandler(logfile)
    if output_dir is not None:
        logfile = logging.FileHandler(os.path.join(output_dir, 'spice.log'), mode=mode)
        logfile.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(logfile)
    addLoggingLevel("TRACE", logging.DEBUG + 5)
    addLoggingLevel("STEP", logging.INFO + 5)
    if verbose:
        logger.setLevel(logging.TRACE)
    else:
        logger.setLevel(logging.INFO)
    return logger, logfile


def addLoggingLevel(levelName: str, levelNum: int, methodName: Optional[str] = None) -> None:
    """Add a logging level, with a logger method and a `logging` function that log at it.

    Does nothing if `logging` already has an attribute `levelName` or `methodName`, or the logger class already has a
    method `methodName`, so repeated calls (one per `initialize_logger` call) are harmless.

    Parameters
    ----------
    levelName : str
        Name of the level, e.g. 'TRACE'. Also set as `logging.<levelName>`, holding `levelNum`.
    levelNum : int
        Numeric value of the level, e.g. `logging.DEBUG + 5`.
    methodName : str, optional
        Name of the logger method and `logging` function that log at this level. If None or empty,
        `levelName.lower()`.
    """
    if not methodName:
        methodName = levelName.lower()

    if ((hasattr(logging, levelName)) or (hasattr(logging, methodName))
            or (hasattr(logging.getLoggerClass(), methodName))):
        pass
    else:
        def logForLevel(self, message, *args, **kwargs):
            if self.isEnabledFor(levelNum):
                self._log(levelNum, message, args, **kwargs)

        def logToRoot(message, *args, **kwargs):
            logging.log(levelNum, message, *args, **kwargs)

        logging.addLevelName(levelNum, levelName)
        setattr(logging, levelName, levelNum)
        setattr(logging.getLoggerClass(), methodName, logForLevel)
        setattr(logging, methodName, logToRoot)
