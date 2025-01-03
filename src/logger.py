import colorlog
import logging
import os
from datetime import datetime

# Constants for log formatting
LOG_FORMAT = "[%(asctime)s] %(levelname)8s %(funcName)25.25s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red,bold",
    "CRITICAL": "red,bg_white,bold",
}


def setup_logger(name, log_level=logging.INFO):
    """
    Set up a logger with both console and file output.

    Args:
        name (str): Logger name
        log_level (int): Logging level (default: logging.INFO)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Configure console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            f"%(log_color)s{LOG_FORMAT}",
            datefmt=DATE_FORMAT,
            log_colors=LOG_COLORS,
            style="%",
        )
    )

    # Configure file handler with timestamp in filename
    log_file = os.path.join(log_dir, f"rag_app_{datetime.now():%Y%m%d}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    # Set up logger
    logger = colorlog.getLogger(name)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(log_level)
    logger.propagate = False

    return logger
