"""Optional MLflow tracking (local ./mlruns). A no-op when mlflow is missing or disabled."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ir_agent.config import TrackingConfig
from ir_agent.runtime import get_logger

log = get_logger("tracking")


class _NullRun:
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float]) -> None: ...
    def log_artifact(self, path: Path) -> None: ...
    def set_tags(self, tags: dict[str, Any]) -> None: ...


class _MlflowRun(_NullRun):
    def __init__(self, mlflow: Any):
        self.mlflow = mlflow

    def log_params(self, params: dict[str, Any]) -> None:
        self.mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        self.mlflow.log_metrics(clean)

    def log_artifact(self, path: Path) -> None:
        if Path(path).exists():
            self.mlflow.log_artifact(str(path))

    def set_tags(self, tags: dict[str, Any]) -> None:
        self.mlflow.set_tags({k: str(v) for k, v in tags.items()})


@contextlib.contextmanager
def tracked_run(cfg: TrackingConfig, run_name: str) -> Iterator[_NullRun]:
    if not cfg.mlflow:
        yield _NullRun()
        return
    try:
        import mlflow
    except ImportError:
        log.debug("mlflow not installed; tracking disabled")
        yield _NullRun()
        return
    uri = cfg.tracking_uri
    artifact_location = None
    if "://" not in uri:
        # A plain directory (default ./mlruns): MLflow >= 3 put the file store in maintenance mode,
        # so keep everything local in <dir>/mlflow.db (SQLite) + <dir>/artifacts.
        root = Path(uri).resolve()
        root.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
        artifact_location = (root / "artifacts").as_uri()
    try:
        mlflow.set_tracking_uri(uri)
        if mlflow.get_experiment_by_name(cfg.experiment) is None:
            mlflow.create_experiment(cfg.experiment, artifact_location=artifact_location)
        mlflow.set_experiment(cfg.experiment)
        ctx = mlflow.start_run(run_name=run_name)
    except Exception as exc:  # tracking must never break a scientific run
        log.warning("MLflow tracking disabled: %s", exc)
        yield _NullRun()
        return
    with ctx:
        yield _MlflowRun(mlflow)
