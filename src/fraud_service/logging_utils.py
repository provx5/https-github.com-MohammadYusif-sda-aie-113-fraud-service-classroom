import logging

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structured JSON logging for local and container execution."""
    logging.basicConfig(level=getattr(logging, str(level).upper(), logging.INFO), force=True)
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
