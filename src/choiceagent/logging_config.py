"""Structured application logging without prompts, credentials, or user secrets."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("trace_id", "method", "path", "status_code", "duration_ms", "error_code"):
            if hasattr(record, key):
                data[key] = getattr(record, key)
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    from logging.handlers import RotatingFileHandler

    handler = RotatingFileHandler(log_dir / "app.jsonl", maxBytes=10_000_000, backupCount=3)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("choiceagent")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(handler)
