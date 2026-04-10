import logging
import sys

class ColorFormatter(logging.Formatter):
    """
    Custom Formatter to add colors based on logging levels and maintain clickable links.
    """
    # ANSI Escape Codes for Colors
    GREY = "\x1b[38;20m"
    CYAN = "\x1b[36;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    # Base format string with the clickable link pattern
    BASE_FORMAT = '[%(asctime)s] [%(levelname)s] File "%(pathname)s", line %(lineno)d: %(message)s'

    FORMATS = {
        logging.DEBUG: GREY + BASE_FORMAT + RESET,
        logging.INFO: CYAN + BASE_FORMAT + RESET,
        logging.WARNING: YELLOW + BASE_FORMAT + RESET,
        logging.ERROR: RED + BASE_FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + BASE_FORMAT + RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logger(name):
    """
    Configures an advanced colored logger that provides clickable links to the source code.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if logger already has handlers to prevent duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)

    return logger
