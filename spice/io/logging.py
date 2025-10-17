import numpy as np
import logging
import sys
import os

def initialize_logger(verbose, output_dir, mode='w'):
    logger = logging.getLogger('SPICE')
    logger.handlers.clear()
    if output_dir is None:
        logfile = logging.StreamHandler(sys.stderr)
    else:
        logfile = logging.FileHandler(os.path.join(output_dir, 'spice.log'), mode=mode)
    fmt = logging.Formatter("%(name)s %(asctime)s | %(levelname)s | %(filename)s >>> %(message)s")
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    addLoggingLevel("TRACE", logging.DEBUG + 5)
    if verbose:
        logger.setLevel(logging.TRACE)
    else:
        logger.setLevel(logging.INFO)
    return logger, logfile

def addLoggingLevel(levelName, levelNum, methodName=None):
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName) or hasattr(logging, methodName) or hasattr(logging.getLoggerClass(), methodName):
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