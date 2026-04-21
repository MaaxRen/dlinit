from __future__ import annotations

import functools
import json
import time
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

from .config import resolve_timezone, timezone_name

RUNS_DIR = Path("training_summary/runs")
JSONL_NAME = "runs.jsonl"
EXCLUDE_NAMES = {"TOKENIZER", "DEVICE", "PROJECT_ROOT"}
P = ParamSpec("P")
R = TypeVar("R")


def _to_jsonable_value(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v

    if isinstance(v, Path):
        return str(v)

    if isinstance(v, datetime):
        return v.isoformat()

    try:
        import numpy as np

        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass

    try:
        import torch

        if isinstance(v, torch.device):
            return str(v)
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                return v.item()
            return {
                "type": "tensor",
                "shape": list(v.shape),
                "dtype": str(v.dtype),
            }
    except Exception:
        pass

    if isinstance(v, dict):
        return {str(k): _to_jsonable_value(val) for k, val in v.items()}
    if isinstance(v, set):
        return [_to_jsonable_value(val) for val in sorted(v, key=repr)]
    if isinstance(v, (list, tuple)):
        return [_to_jsonable_value(val) for val in v]

    try:
        json.dumps(v)
        return v
    except TypeError:
        return repr(v)


def collect_training_metadata(
    namespace: dict[str, Any],
    *,
    tz: str | tzinfo | None = None,
) -> dict[str, Any]:
    """
    Collect ALL-CAPS globals into a JSON-safe metadata dict.

    The intended usage is:

    ```python
    metadata = collect_training_metadata(globals())
    ```
    """

    resolved_tz = resolve_timezone(tz)
    now = datetime.now(resolved_tz)
    timestamp = now.isoformat()
    if resolved_tz == timezone.utc:
        timestamp = timestamp.replace("+00:00", "Z")

    meta: dict[str, Any] = {
        "run_id": now.strftime("%Y-%m-%d_%H%M%S"),
        "date": now.strftime("%Y-%m-%d"),
        "timestamp": timestamp,
        "timezone": timezone_name(resolved_tz),
    }

    for name, value in namespace.items():
        if name.startswith("__") and name.endswith("__"):
            continue
        if name != name.upper():
            continue
        if name in EXCLUDE_NAMES:
            continue
        if callable(value):
            continue

        meta[name.lower()] = _to_jsonable_value(value)

    note = namespace.get("TRAINING_NOTE")
    if isinstance(note, str) and note.strip():
        meta["training_note"] = note.strip()

    return meta


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / "training_summary").is_dir() or (parent / "src").is_dir():
            return parent
    return current


def metadata_output_dir(project_root: Path | None = None) -> Path:
    root = find_project_root(project_root)
    out_dir = root / RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_training_metadata(
    namespace: dict[str, Any],
    *,
    project_root: Path | None = None,
    filename: str | None = None,
    extra: dict[str, Any] | None = None,
    tz: str | tzinfo | None = None,
) -> dict[str, Path]:
    """
    Persist metadata in two forms:
    - `<run_id>.json` for one-file-per-run inspection
    - `runs.jsonl` for append-only aggregation
    """

    metadata = collect_training_metadata(namespace, tz=tz)

    if extra:
        for key, value in extra.items():
            metadata[str(key)] = _to_jsonable_value(value)

    out_dir = metadata_output_dir(project_root)
    run_name = filename or metadata["run_id"]
    json_path = out_dir / f"{run_name}.json"
    jsonl_path = out_dir / JSONL_NAME

    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(metadata) + "\n")

    return {"json_path": json_path, "jsonl_path": jsonl_path}


def log_training_run(
    func: Callable[P, R] | None = None,
    *,
    project_root: Path | None = None,
    filename: str | None = None,
    extra: dict[str, Any] | None = None,
    capture_result: bool = True,
    tz: str | tzinfo | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    """
    Decorator for training entrypoints.

    Usage:

    ```python
    @log_training_run
    def train():
        ...

    @log_training_run(filename="baseline")
    def train():
        ...
    ```
    """

    def decorator(inner_func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(inner_func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            started_at = time.perf_counter()

            try:
                result = inner_func(*args, **kwargs)
            except Exception as exc:
                failure_extra: dict[str, Any] = {
                    "status": "failed",
                    "function_name": inner_func.__name__,
                    "duration_sec": round(time.perf_counter() - started_at, 6),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                if extra:
                    failure_extra.update(extra)

                save_training_metadata(
                    inner_func.__globals__,
                    project_root=project_root,
                    filename=filename,
                    extra=failure_extra,
                    tz=tz,
                )
                raise

            success_extra: dict[str, Any] = {
                "status": "completed",
                "function_name": inner_func.__name__,
                "duration_sec": round(time.perf_counter() - started_at, 6),
            }
            if capture_result:
                success_extra["result"] = result
            if extra:
                success_extra.update(extra)

            save_training_metadata(
                inner_func.__globals__,
                project_root=project_root,
                filename=filename,
                extra=success_extra,
                tz=tz,
            )
            return result

        return wrapper

    if func is not None:
        return decorator(func)

    return decorator
