"""Structured logging configuration.

Enterprise standard: JSON logs in production (for log aggregators like ELK, Datadog),
pretty console logs in development. NEVER use print().
"""
import logging
import sys

import structlog

from smartdesk.config import get_config


def configure_logging() -> None:
    """Configure structlog + stdlib logging. Call once at app startup."""
    config = get_config()
    level = getattr(logging, config.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if config.environment == "dev":
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger. Use module name: logger = get_logger(__name__)."""
    return structlog.get_logger(name)
