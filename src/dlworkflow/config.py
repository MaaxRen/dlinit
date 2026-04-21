from __future__ import annotations

import json
import os
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

CONFIG_FILE_NAME = "config.json"
ALLOWED_PROFILES = ("NLP", "CV", "ML", "STAT")


def config_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "dlworkflow"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "dlworkflow"
    return Path.home() / ".config" / "dlworkflow"


def config_path() -> Path:
    return config_dir() / CONFIG_FILE_NAME


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_config(data: dict[str, Any]) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def update_config(**updates: Any) -> tuple[dict[str, Any], Path]:
    config = load_config()
    for key, value in updates.items():
        if value is not None:
            config[key] = value
    return config, save_config(config)


def clear_config_keys(*keys: str) -> tuple[dict[str, Any], Path]:
    config = load_config()
    for key in keys:
        config.pop(key, None)
    return config, save_config(config)


def resolve_timezone(name: str | tzinfo | None = None) -> tzinfo:
    tz_name = name
    if tz_name is None:
        tz_name = load_config().get("timezone")
    if tz_name is None:
        return timezone.utc
    if isinstance(tz_name, tzinfo):
        return tz_name
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown time zone: {tz_name}") from exc


def timezone_name(name: str | tzinfo | None = None) -> str:
    tz = resolve_timezone(name)
    return getattr(tz, "key", None) or str(tz)


def now_in_timezone(name: str | tzinfo | None = None) -> datetime:
    return datetime.now(resolve_timezone(name))


def normalize_profile(value: str) -> str:
    cleaned = value.strip().upper()
    if cleaned not in ALLOWED_PROFILES:
        allowed = ", ".join(ALLOWED_PROFILES)
        raise ValueError(f"Unknown profile: {value}. Expected one of: {allowed}")
    return cleaned
