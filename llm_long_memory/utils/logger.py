"""Config-driven logger that writes to both console and file."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from llm_long_memory.utils.helpers import load_config, resolve_project_path


class Logger:
    """Simple logger with timestamped console and file output."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize logger from config and ensure log directory exists."""
        self._config = config
        logging_cfg = self._config["logging"]
        log_file = str(logging_cfg["log_file"])
        self._console_enabled = bool(logging_cfg["console_enabled"])
        self._min_level = str(logging_cfg["level"]).strip().upper()
        self._level_rank = {"ERROR": 40, "WARN": 30, "INFO": 20, "DEBUG": 10}
        if self._min_level not in self._level_rank:
            self._min_level = "INFO"
        self._log_path = self._resolve_path(log_file)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._file = self._log_path.open("a", encoding="utf-8", buffering=1)

    @staticmethod
    def _resolve_path(path: str) -> Path:
        return resolve_project_path(path)

    @staticmethod
    def _stamp(level: str, msg: str) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{now}][{level}] {msg}"

    def _write(self, level: str, msg: str) -> None:
        if self._level_rank.get(level, 0) < self._level_rank[self._min_level]:
            return
        line = self._stamp(level, msg)
        with self._lock:
            if self._console_enabled:
                print(line)
            self._file.write(line + "\n")
            self._file.flush()

    def info(self, msg: str) -> None:
        """Log an informational message."""
        self._write("INFO", msg)

    def debug(self, msg: str) -> None:
        """Log a debug message."""
        self._write("DEBUG", msg)

    def warn(self, msg: str) -> None:
        """Log a warning message."""
        self._write("WARN", msg)

    def error(self, msg: str) -> None:
        """Log an error message."""
        self._write("ERROR", msg)

    def close(self) -> None:
        """Close the file handle if opened."""
        try:
            self._file.close()
        except OSError:
            return


class _LoggerProxy:
    """Lazy logger proxy to avoid import-time config side effects."""

    def __init__(self) -> None:
        self._impl: Optional[Logger] = None

    def _get(self) -> Logger:
        if self._impl is None:
            self._impl = Logger(load_config())
        return self._impl

    def info(self, msg: str) -> None:
        self._get().info(msg)

    def debug(self, msg: str) -> None:
        self._get().debug(msg)

    def warn(self, msg: str) -> None:
        self._get().warn(msg)

    def error(self, msg: str) -> None:
        self._get().error(msg)


logger = _LoggerProxy()
