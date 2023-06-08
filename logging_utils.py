import logging
import logging.handlers
import os
import sys
from datetime import datetime


def __create_log_path(log_name):
    """Create a log path from a log directory and log name.
    Parameters
    ----------
    log_name : str
        The name of the log file.
    Returns
    -------
    str
        The path to the log file.
    """

    # Path in the working directoy:
    path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "logs",
        datetime.now().strftime("%Y-%m-%d"),
        f"{log_name}_{datetime.now().strftime('%H-%M-%S')}.log",
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def setup_custom_logger(name, level=logging.INFO):
    # logger settings
    log_format = "%(asctime)s [%(levelname)s] %(filename)s/%(funcName)s:%(lineno)s >> %(message)s"

    # setup logger with TimedRotatingFileHandler
    handler = logging.handlers.TimedRotatingFileHandler(
        __create_log_path(name),
        when="midnight",
        backupCount=3,
        encoding="utf-8",
        delay=False,
        utc=False,
        atTime=datetime.now().time(),
    )
    handler.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # print log messages to console
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(consoleHandler)

    return logger


# logger = setup_custom_logger("ecallisto")
# source: https://docs.python.org/2/howto/logging.html https://stackoverflow.com/questions/37958568/how-to-implement-a-global-python-logger
# logger.debug("")      // Detailed information, typically of interest only when diagnosing problems.
# logger.info("")       // Confirmation that things are working as expected.
# logger.warning("")    // An indication that something unexpected happened, or indicative of some problem in the near future
# logger.error("")      // Due to a more serious problem, the software has not been able to perform some function.
# logger.critical("")   // A serious error, indicating that the program itself may be unable to continue running.


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
