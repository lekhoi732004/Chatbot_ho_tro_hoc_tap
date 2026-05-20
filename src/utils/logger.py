"""
Centralized logger for edu-agent.
Provides coloured console output + rotating file handler.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


_loggers: dict[str, logging.Logger] = {}


def get_logger(
    name: str = "edu_agent",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_console: bool = True,
) -> logging.Logger:
    """
    Return a named logger, creating it on first call.
    Subsequent calls with the same name return the cached instance.
    """
    if name in _loggers:
        return _loggers[name]

    # Resolve config defaults
    try:
        from src.utils.config_loader import get_config
        cfg = get_config()
        level = level or cfg.get("logging.level", "INFO")
        log_file = log_file or cfg.get("logging.log_file", "logs/edu_agent.log")
        enable_console = cfg.get("logging.enable_console", True)
    except Exception:
        level = level or "INFO"
        log_file = log_file or "logs/edu_agent.log"

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if enable_console and not any(
        isinstance(h, logging.StreamHandler) for h in logger.handlers
    ):
        ch = logging.StreamHandler()
        ch.setLevel(numeric_level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fh.setLevel(numeric_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    _loggers[name] = logger
    return logger
