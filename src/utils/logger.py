"""
Structured logging configuration using loguru.
Import `logger` from this module across the entire application.
"""

import sys
from loguru import logger

# Remove default handler
logger.remove()

# Console handler — colored, structured
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    level="DEBUG",
    colorize=True,
)

# File handler — rotating log files
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
)

__all__ = ["logger"]
