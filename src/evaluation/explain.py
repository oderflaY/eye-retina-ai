"""Explicabilidad: Grad-CAM y Grad-CAM++.

Sirve para el detector (objetivo = logit de objetividad más alto, o de una
celda concreta) y para los clasificadores de la Fase 2 (objetivo = logit de
una clase).
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import tensorflow as tf

from src.models.eye_grid_detector import HEAD_FEATURE_LAYER

TargetFn = Callable[[tf.Tensor], tf.Tensor]


def max_objectness(pred: tf.Tensor) -> tf.Tensor:
    """Objetivo por defecto del detector: el logit de objetividad máximo."""
    return tf.reduce_max(pred[..., 0], axis=[1, 2])


def cell_objectness(row: int, col: int) -> TargetFn:
    """Objetivo: logit de objetividad de una celda (explica una detección concreta)."""
    return lambda pred: pred[:, row, col, 0]


def _grads(
    model: tf.keras.Model, images: np.ndarray, layer: str, target: TargetFn
) -> tuple[tf.Tensor, tf.Tensor]:
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer).output, model.output])
    x = tf.convert_to_tensor(images, tf.float32)
    with tf.GradientTape() as tape:
        feats, pred = grad_model(x, training=False)
        score = tf.cast(target(pred), tf.float32)
    grads = tape.gradient(score, feats)  # respecto a la salida real (mixed precision)
    return tf.cast(feats, tf.float32), tf.cast(grads, tf.float32)


def _normalize(cam: tf.Tensor, size: tuple[int, int]) -> np.ndarray:
    cam = tf.nn.relu(cam).numpy()
    out = []
    for c in cam:
        c = c / (c.max() + 1e-8)
        out.append(cv2.resize(c, size, interpolation=cv2.INTER_LINEAR))
    return np.stack(out)


def grad_cam(
    model: tf.keras.Model,
    images: np.ndarray,
    target: TargetFn = max_objectness,
    layer: str = HEAD_FEATURE_LAYER,
) -> np.ndarray:
    """Grad-CAM (Selvaraju et al., 2017). Devuelve mapas (B, H, W) en [0, 1]."""
    feats, grads = _grads(model, images, layer, target)
    weights = tf.reduce_mean(grads, axis=[1, 2], keepdims=True)
    cam = tf.reduce_sum(weights * feats, axis=-1)
    return _normalize(cam, images.shape[2:0:-1])


def grad_cam_pp(
    model: tf.keras.Model,
    images: np.ndarray,
    target: TargetFn = max_objectness,
    layer: str = HEAD_FEATURE_LAYER,
) -> np.ndarray:
    """Grad-CAM++ (Chattopadhyay et al., 2018): mejor con varios objetos (dos ojos)."""
    feats, grads = _grads(model, images, layer, target)
    g2, g3 = tf.square(grads), tf.pow(grads, 3)
    sum_feats = tf.reduce_sum(feats, axis=[1, 2], keepdims=True)
    denom = 2.0 * g2 + sum_feats * g3
    alpha = g2 / tf.where(tf.abs(denom) > 1e-8, denom, tf.ones_like(denom))
    weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=[1, 2], keepdims=True)
    cam = tf.reduce_sum(weights * feats, axis=-1)
    return _normalize(cam, images.shape[2:0:-1])


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpone el mapa de calor (JET) sobre la imagen RGB uint8."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (image.astype(np.float32) * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)
