import logging
from pathlib import Path


def setup_logger(log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rldiff2cad")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(Path(log_dir) / "train.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger
