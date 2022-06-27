import os
import logging


def get_logger(name, log_path, level=logging.INFO):
    formatter = logging.Formatter(
        fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S"
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger