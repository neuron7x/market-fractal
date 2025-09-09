import json
import logging
import os
import sys
import time
import uuid


logger = logging.getLogger(__name__)
_LOGGING_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Format log records as JSON with standard metadata.

    Each record is serialized to JSON with the timestamp (``ts``),
    log level, message, and optional ``run_id``, ``mode`` and
    ``scenario`` fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        rec = {
            "ts": int(time.time() * 1000),
            "level": record.levelname.lower(),
            "msg": record.getMessage(),
            "run_id": getattr(record, "run_id", None),
            "mode": getattr(record, "mode", None),
            "scenario": getattr(record, "scenario", None),
        }
        return json.dumps(rec, ensure_ascii=False)


def configure_logging() -> None:
    """Configure root logger to emit JSON-formatted logs to ``stderr``.

    The log level is taken from the ``LOG_LEVEL`` environment variable,
    defaulting to ``INFO`` if not provided.
    """

    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        logger.debug(
            "Uvicorn is not installed; skipping Uvicorn-specific logging configuration."
        )

    _LOGGING_CONFIGURED = True


def new_run_id() -> str:
    """Return a unique identifier for the current run."""

    return uuid.uuid4().hex
