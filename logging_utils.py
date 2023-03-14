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
    path = os.path.join(
        os.path.abspath(os.sep),
        "var",
        "log",
        "ecallisto",
        datetime.now().strftime("%Y-%m-%d"),
        f"{log_name}_{datetime.now().strftime('%H-%M-%S')}.log",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def setup_custom_logger(name, level=logging.INFO):
    # logger settings
    log_file = __create_log_path(name)
    log_file_max_size = 1024 * 1024 * 20  # megabytes
    log_num_backups = 3
    log_format = "%(asctime)s [%(levelname)s] %(filename)s/%(funcName)s:%(lineno)s >> %(message)s"
    log_filemode = "w"  # w: overwrite; a: append

    # setup logger
    logging.basicConfig(
        filename=log_file,
        format=log_format,
        filemode=log_filemode,
        level=level,
    )
    rotate_file = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=log_file_max_size, backupCount=log_num_backups
    )
    logger = logging.getLogger(name)
    logger.addHandler(rotate_file)

    # print log messages to console
    consoleHandler = logging.StreamHandler()
    logFormatter = logging.Formatter(log_format)
    consoleHandler.setFormatter(logFormatter)
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
