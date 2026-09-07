#%% modules setup

import logging
import os
import colorlog
from colorlog import ColoredFormatter

#%% Custom TRACE level (below DEBUG), registered once for the whole project

def addLoggingLevel(levelName, levelNum, methodName=None):
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError("{} already defined in logging module".format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError("{} already defined in logging module".format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError("{} already defined in Logger class".format(methodName))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


if not hasattr(logging, 'TRACE'):
    addLoggingLevel("TRACE", logging.DEBUG - 5)


#%% Centralised handler/level setup
#
# Every module should just do "loger_x = logging.getLogger(__name__)" and
# nothing else here -- no setLevel, no handler. A logger with no level of its
# own inherits the effective level of its nearest ancestor, all the way up to
# the root logger, and propagates records up to the root's handlers by
# default. So configuring the root once, from whichever entry-point script
# calls setup_logging(), controls every module's verbosity and formatting in
# one place instead of each file setting its own.

DEFAULT_LEVEL = logging.INFO

LOG_COLORS = {
    'TRACE': 'white',
    'DEBUG': 'purple',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
}

LOG_FORMAT = (
    '%(white)s%(asctime) -5s| %(blue)s%(name) -28s %(black)s| %(cyan)s %(funcName) '
    '-28s %(black)s|''%(log_color)s%(levelname) -10s | %(message)s'
)

# Attaching the handler to the root logger means every logger without its own
# level -- including third-party libraries -- inherits whatever level we pick.
# That is fine at INFO, but at DEBUG/TRACE it floods the output with library
# internals (matplotlib's LaTeX pipeline is a good example). These are capped
# at WARNING regardless of the chosen project level; add to this list if
# another dependency turns out to be noisy
QUIET_LOGGERS = ['matplotlib', 'PIL', 'h5py']


def setup_logging(level=None):
    """
    Configure the root logger once, near the top of an entry-point script.

    Input:
    level -> int, str or None: explicit level (e.g. logging.DEBUG or 'DEBUG').
             If None, falls back to the LOCAL2D_LOGLEVEL environment variable,
             then to DEFAULT_LEVEL. This is the one place to change verbosity:
             pass a level explicitly, set the environment variable, or edit
             DEFAULT_LEVEL above -- never a per-module setLevel() call.
    """
    if level is None:
        level = os.environ.get('LOCAL2D_LOGLEVEL', DEFAULT_LEVEL)
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    handler = colorlog.StreamHandler()
    handler.setFormatter(ColoredFormatter(
        LOG_FORMAT, datefmt=None, reset=True, log_colors=LOG_COLORS,
        secondary_log_colors={}, style='%'
    ))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
