"""Structured runtime logging configuration for the application."""

import logging
import os
from logging.handlers import RotatingFileHandler

from cv_scribble_diffusion.config import LoggingConfig


_CONFIGURED = False


def configure_logging(cfg: LoggingConfig):
    """Configure root logging once for file + console output."""
    global _CONFIGURED
    if _CONFIGURED or not cfg.enabled:
        return

    os.makedirs(cfg.dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(_parse_level(cfg.level, logging.DEBUG))
    root.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        os.path.join(cfg.dir, cfg.file_name),
        maxBytes=cfg.max_bytes,
        backupCount=cfg.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(_parse_level(cfg.level, logging.DEBUG))
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(_parse_level(cfg.console_level, logging.INFO))
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a module logger."""
    return logging.getLogger(name)


def _parse_level(name: str, default: int) -> int:
    level = getattr(logging, name.upper(), None)
    return level if isinstance(level, int) else default
