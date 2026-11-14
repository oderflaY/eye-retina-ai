"""Configuración global del proyecto: rutas, hiperparámetros por defecto y semillas.

Las rutas se pueden sobrescribir con variables de entorno o con un archivo
``.env`` en la raíz del proyecto (ver ``.env.example``).

Ejemplo:
    >>> from src.config import RAW_DIR, IMG_SIZE_EYE, set_global_seed
    >>> set_global_seed(42)
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

try:  # python-dotenv es opcional: si no está, se usan las variables del sistema
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _path_from_env(var: str, default: Path) -> Path:
    """Lee una ruta de una variable de entorno (relativa a la raíz del proyecto)."""
    value = os.getenv(var)
    if not value:
        return default
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


# --- Rutas -----------------------------------------------------------------
DATA_DIR: Path = _path_from_env("DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR: Path = _path_from_env("RAW_DIR", DATA_DIR / "raw")
PROCESSED_DIR: Path = _path_from_env("PROCESSED_DIR", DATA_DIR / "processed")
AUGMENTED_DIR: Path = _path_from_env("AUGMENTED_DIR", DATA_DIR / "augmented")
MODELS_DIR: Path = _path_from_env("MODELS_DIR", PROJECT_ROOT / "models")
LOGS_DIR: Path = _path_from_env("LOGS_DIR", PROJECT_ROOT / "logs")
REPORTS_DIR: Path = _path_from_env("REPORTS_DIR", PROJECT_ROOT / "reports")
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"

# --- Parámetros globales -----------------------------------------------------
IMG_SIZE_EYE: int = int(os.getenv("IMG_SIZE_EYE", 224))
IMG_SIZE_RETINA: int = int(os.getenv("IMG_SIZE_RETINA", 512))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 32))
EPOCHS: int = int(os.getenv("EPOCHS", 50))
SEED: int = int(os.getenv("SEED", 42))
NUM_LANDMARKS: int = 6


def set_global_seed(seed: int = SEED, deterministic: bool = False) -> None:
    """Fija las semillas de ``random``, NumPy y TensorFlow.

    Args:
        seed: semilla global.
        deterministic: si es True activa ``tf.config.experimental.enable_op_determinism``
            (resultados idénticos en GPU a costa de velocidad).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        if deterministic:
            tf.config.experimental.enable_op_determinism()
    except ImportError:  # pragma: no cover - permite usar el módulo sin TF
        pass


def enable_mixed_precision_if_gpu() -> bool:
    """Activa ``mixed_float16`` solo si hay GPU, y crecimiento de memoria.

    Returns:
        True si se activó mixed precision.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # la GPU ya estaba inicializada
    policy = "mixed_float16" if gpus else "float32"
    tf.keras.mixed_precision.set_global_policy(policy)
    return bool(gpus)


def ensure_dirs() -> None:
    """Crea las carpetas de salida si no existen."""
    for d in (PROCESSED_DIR, AUGMENTED_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    for name in ("PROJECT_ROOT", "RAW_DIR", "PROCESSED_DIR", "MODELS_DIR", "LOGS_DIR", "REPORTS_DIR"):
        print(f"{name:14s} = {globals()[name]}")
    print(f"IMG_SIZE_EYE={IMG_SIZE_EYE} BATCH_SIZE={BATCH_SIZE} EPOCHS={EPOCHS} SEED={SEED}")
