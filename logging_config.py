"""
Logging configuration for APEX — colourised console output
with structured file logging.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class APEXFormatter(logging.Formatter):
    """Custom formatter with colour-coded log levels."""

    GREY = "\033[38;5;245m"
    BLUE = "\033[38;5;75m"
    YELLOW = "\033[38;5;220m"
    RED = "\033[38;5;196m"
    BOLD_RED = "\033[1;38;5;196m"
    PURPLE = "\033[38;5;141m"
    RESET = "\033[0m"

    FORMATS = {
        logging.DEBUG: GREY + "%(asctime)s | DEBUG    | %(name)-30s | %(message)s" + RESET,
        logging.INFO: BLUE + "%(asctime)s | " + PURPLE + "INFO" + BLUE + "     | %(name)-30s | %(message)s" + RESET,
        logging.WARNING: YELLOW + "%(asctime)s | WARNING  | %(name)-30s | %(message)s" + RESET,
        logging.ERROR: RED + "%(asctime)s | ERROR    | %(name)-30s | %(message)s" + RESET,
        logging.CRITICAL: BOLD_RED + "%(asctime)s | CRITICAL | %(name)-30s | %(message)s" + RESET,
    }

    def format(self, record):
        fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure APEX-wide logging."""
    root = logging.getLogger("apex")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not root.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(APEXFormatter())
        root.addHandler(console)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)
