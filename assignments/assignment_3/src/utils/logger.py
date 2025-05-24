"""Simple console logging module."""

import logging
import sys

from ..config.settings import settings

# Create logger
logger = logging.getLogger("ngram-model")
logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

# Create formatter
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)8s | %(message)s", datefmt="%H:%M:%S"
)

# Add formatter to handler
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)


def log_section(title: str, emoji: str | None = None) -> None:
    """Log a section header with optional emoji."""
    if emoji:
        title = f"{emoji}  {title}"

    logger.info("\n" + "=" * 80)
    logger.info(title)
    logger.info("=" * 80)
