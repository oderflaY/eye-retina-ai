"""Reproducibilidad: semillas fijas para Python, NumPy y TensorFlow."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Fija todas las semillas aleatorias.

    Args:
        seed: semilla global (42 por defecto).
        deterministic: si es True activa ``enable_op_determinism`` de TF;
            garantiza resultados idénticos en GPU a costa de velocidad.
    """
    import tensorflow as tf  # import diferido: los módulos de datos no lo necesitan

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def configure_gpu(mixed_precision: bool) -> bool:
    """Activa crecimiento de memoria en GPU y, opcionalmente, mixed precision.

    Returns:
        True si se activó ``mixed_float16``.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # la GPU ya estaba inicializada
    if mixed_precision and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        return True
    tf.keras.mixed_precision.set_global_policy("float32")
    return False
