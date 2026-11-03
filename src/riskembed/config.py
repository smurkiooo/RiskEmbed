"""Загрузка YAML-конфига с доступом через точку."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Cfg(dict):
    """dict с доступом через точку: cfg.amex.work_dir."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Cfg(value) if isinstance(value, dict) else value


def load_config(path: str | Path | None = None, **overrides: Any) -> Cfg:
    path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    with path.open() as f:
        cfg = yaml.safe_load(f)
    for dotted, value in overrides.items():
        node = cfg
        *parents, leaf = dotted.split("__")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return Cfg(cfg)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
