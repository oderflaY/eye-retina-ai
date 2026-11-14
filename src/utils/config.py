"""Carga de configuración YAML con overrides desde la línea de comandos."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Lee un YAML y aplica overrides del tipo ``train.epochs=5``.

    Args:
        path: ruta al archivo YAML.
        overrides: lista de cadenas ``clave.anidada=valor``; el valor se
            interpreta como YAML (números, booleanos, null, listas).

    Returns:
        Diccionario de configuración.
    """
    with open(path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Override inválido (falta '='): {item}")
        set_nested(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Asigna ``value`` en ``cfg`` siguiendo una clave con puntos."""
    node = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def resolve_path(path: str | Path) -> Path:
    """Convierte rutas relativas del YAML en rutas absolutas del proyecto."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def deep_copy(cfg: dict[str, Any]) -> dict[str, Any]:
    """Copia profunda de la configuración (para no mutar el original)."""
    return copy.deepcopy(cfg)
