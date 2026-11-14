"""Logging setup and per-run output directories (runs/<timestamp>_<label>/)."""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

LOGGER_NAME = "ir_agent"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the package logger (``ir_agent.<name>``)."""
    return logging.getLogger(LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}")


def setup_logging(level: str | int = "INFO", logfile: Path | None = None) -> logging.Logger:
    """Configure the package logger once: console handler plus optional file handler."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level if isinstance(level, int) else level.upper())
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    if not any(getattr(h, "_ir_agent_console", False) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        sh._ir_agent_console = True  # type: ignore[attr-defined]
        logger.addHandler(sh)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def slugify(text: str) -> str:
    """File-system-safe label: ``CO on Pt(111) / xtb`` -> ``CO_on_Pt111_xtb``."""
    text = text.replace("+", "plus")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_.") or "run"


@dataclass
class RunDir:
    """A run's output directory with conventional sub-paths."""

    root: Path

    @classmethod
    def create(cls, output_root: str | Path, label: str, timestamp: str | None = None) -> RunDir:
        ts = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
        base = Path(output_root) / f"{ts}_{slugify(label)}"
        base.parent.mkdir(parents=True, exist_ok=True)
        root, suffix = base, 1
        while True:  # atomic: two processes started in the same second get distinct dirs
            try:
                root.mkdir()
                break
            except FileExistsError:
                suffix += 1
                root = base.with_name(f"{base.name}-{suffix}")
        for sub in ("structures", "calc", "plots"):
            (root / sub).mkdir()
        return cls(root)

    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.path(name)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=_json_default)
        return p


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating | np.integer):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "model_dump"):
        return o.model_dump()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
