"""Centralized logging setup. Writes a verbose log file and a concise stderr stream."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def setup_logging(
    log_dir: Path | None = None,
    level: str = "INFO",
    file_level: str = "DEBUG",
) -> Path:
    """Configure root logging. Returns the log file path.

    - File handler at `file_level` (default DEBUG) → captures every event.
    - Stderr handler at `level` (default INFO) → readable summary while running.
    - Environment override: BLENDERING_LOG_LEVEL=DEBUG etc.
    """
    level = os.environ.get("BLENDERING_LOG_LEVEL", level).upper()
    file_level = os.environ.get("BLENDERING_FILE_LOG_LEVEL", file_level).upper()

    log_dir = log_dir or Path("./.blendering")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "blendering.log"

    fmt = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    # Reset on repeated calls so tests/CLI invocations don't accumulate handlers.
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(getattr(logging, file_level, logging.DEBUG))
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(getattr(logging, level, logging.INFO))
    sh.setFormatter(logging.Formatter("%(levelname)-7s %(name)s | %(message)s"))
    root.addHandler(sh)

    # Quiet down the noisiest libraries on stderr; full detail stays in the file.
    for noisy in ("httpx", "httpcore", "LiteLLM", "litellm", "mcp.server.lowlevel.server"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root.info(f"logging initialised → {log_path}")
    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
