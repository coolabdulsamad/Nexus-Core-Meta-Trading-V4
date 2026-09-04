import logging
import os
import sys
from pythonjsonlogger import jsonlogger


def setup_logger(name: str, log_file: str = "logs/app.log"):
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:                    # idempotent: re-imports don't double-log
        return logger
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
