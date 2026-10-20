import json
import logging
import sys
from datetime import UTC, datetime

from src.glucotwin.config.settings import settings


class StructuredJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno,
            "project": settings.PROJECT_NAME,
            "environment": settings.ENVIRONMENT,
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging() -> logging.Logger:
    app_logger = logging.getLogger(settings.PROJECT_NAME)
    app_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    if not app_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJsonFormatter())
        app_logger.addHandler(handler)

    app_logger.propagate = False
    return app_logger


logger = setup_logging()
