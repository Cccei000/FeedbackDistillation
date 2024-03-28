# encoding=utf-8
import logging
from enum import Enum
from typing import Optional, Union

from .rank import is_rank_0


logger: logging.Logger = None

LOGGER_NAME = "Log"

LOGGING_FORMAT = {
    "debug": "[%(asctime)s] %(levelname)s: %(message)s",
    "info": "[%(asctime)s] %(levelname)s: %(message)s"
}
GET_LOGGING_LEVEL = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}
class LoggingLevel(Enum):
    """Available Logging Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    """    
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"



def logging_initialize(fname: Optional[str] = None, fmode: str = "a", level: str = "info"):
    """initialize logger

    Args:
        fname (Optional[str], optional): Specifies that a FileHandler be created, using the specified filename. Defaults to None.
        fmode (str, optional): Specifies the mode to open the file, if filename is specified. Defaults to "a".
        level (str, optional): logging level. Should be 'debug' or 'info'. Defaults to "info".
    """
    global logger
    
    if is_rank_0() and logger is None:
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(GET_LOGGING_LEVEL["debug"])
        handler = logging.StreamHandler()
        handler.setLevel(GET_LOGGING_LEVEL.get(level, "info"))
        formatter = logging.Formatter(LOGGING_FORMAT.get(level, LOGGING_FORMAT["info"]))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        if fname is not None:
            f_handler = logging.FileHandler(filename=fname, encoding="utf-8")
            f_handler.setLevel(GET_LOGGING_LEVEL.get(level, "info"))
            f_handler.setFormatter(formatter)
            logger.addHandler(f_handler)
            logger.info(f"Initialize logger, save to '{fname}'.")
        else:
            logger.info(f"Initialize logger!")
    return

def logging_rank_0(msg: str, level:Union[LoggingLevel,str]=LoggingLevel.INFO):
    """global rank 0 logging

    Args:
        msg (str): message
        level (str, optional): logging level Should be 'debug', 'info', 'warning', 'error' or 'critical'. Defaults to "info".
    """
    global logger
    if isinstance(level, LoggingLevel):
        level = level.value
    
    if is_rank_0() and logger is not None:
        if level == "info":
            logger.info(msg=msg)
        elif level == "warning":
            logger.warning(msg=msg)
        elif level == "error":
            logger.error(msg=msg)
        elif level == "critical":
            logger.critical(msg=msg)
        else:
            logger.debug(msg=msg)
    return
