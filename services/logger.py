import logging
from typing import Optional

try:
    from colorama import Fore, Style, init as colorama_init

    colorama_init(autoreset=True)

    _COLOR_MAP = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }
except ImportError:  # pragma: no cover - colorama might not be installed in all environments
    Fore = Style = None  # type: ignore
    _COLOR_MAP = {
        logging.DEBUG: "",
        logging.INFO: "",
        logging.WARNING: "",
        logging.ERROR: "",
        logging.CRITICAL: "",
    }


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = _COLOR_MAP.get(record.levelno, "")
        if Fore is None or not color:
            return message
        return f"{color}{message}{Style.RESET_ALL}"


def _configure_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("vector_benchmark")
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = _ColorFormatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a shared logger with colored output."""
    base_logger = _configure_logger()
    if not name:
        return base_logger

    return base_logger.getChild(name)

