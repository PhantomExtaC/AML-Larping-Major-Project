import logging
import json
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Dict

# Context variable to track trace IDs across async boundaries
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)

class JSONFormatter(logging.Formatter):
    """Structured JSON formatter including trace IDs."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        trace_id = trace_id_var.get()
        if trace_id:
            log_obj["trace_id"] = trace_id
            
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

def setup_logging(level: int = logging.INFO) -> None:
    """Initialize global structured logging."""
    logger = logging.getLogger("aml")
    logger.setLevel(level)
    
    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
    # Silence third-party noise
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)