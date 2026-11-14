"""Cargador de datos para el detector de ojos de regresión directa.

``EyeDataLoader`` escanea ``data/raw/eyes``, detecta el formato de las etiquetas
y construye un ``tf.data.Dataset`` listo para ``model.fit``.

Formatos aceptados (autodetección):

* **Etiquetas con caja**: CSV, COCO JSON, YOLO TXT o Pascal VOC
  (ver ``docs/datasets.md``). Pueden traer los 6 landmarks.
* **Carpetas por clase**: ``eye/`` (u ``ojo/``) y ``no_eye/`` (``no_ojo/``,
  ``negative/``, ``background/``) sin cajas: solo se entrena la cabeza de
  clasificación ojo/no-ojo.

El modelo predice **una** caja por imagen. Para imágenes con varios ojos
(rostros completos) se crea una fila por ojo con un recorte cuadrado alrededor
de él (2.5 × su tamaño, con un desplazamiento aleatorio reproducible). Las
imágenes sin ojos se usan como negativos (``eye_prob = 0``).

Cada elemento del dataset es ``(imagen, objetivos, pesos)``:

* ``imagen``: float32 (H, W, 3) en [0, 1].
* ``objetivos["bbox"]``: (4,) x, y, w, h normalizados al recorte.
* ``objetivos["landmarks"]``: (18,) = 12 coordenadas normalizadas + 6 máscaras.
* ``objetivos["eye_prob"]``: (1,) 1 = hay ojo.
* ``pesos``: 0 en bbox/landmarks cuando la fila no tiene caja.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # permite `python src/data/loader_eyes.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import BATCH_SIZE, IMG_SIZE_EYE, NUM_LANDMARKS, RAW_DIR, SEED
from src.data.eye_labels import (
    AmbiguousLabelFormat,
    LabelFormatError,
    load_eye_samples,
)
from src.data.io import image_size, list_images, read_image_rgb
from src.data.splits import stratified_split

AUTOTUNE = tf.data.AUTOTUNE
POSITIVE_DIRS = {"eye", "eyes", "ojo", "ojos", "positive", "positivo"}
NEGATIVE_DIRS = {
    "no_eye",
    "no_eyes",
    "noeye",
    "no_ojo",
    "no_ojos",
    "negative",
    "negativo",
    "background",
    "fondo",
    "other",
    "otro",
}
LM_COLS = [f"lm{i}_{a}" for i in range(NUM_LANDMARKS) for a in ("x", "y")]
MASK_COLS = [f"lm{i}_m" for i in range(NUM_LANDMARKS)]
BOX_COLS = ["bx", "by", "bw", "bh"]
CROP_COLS = ["crop_x", "crop_y", "crop_w", "crop_h"]
TARGET_DIM = 4 + 2 * NUM_LANDMARKS + NUM_LANDMARKS + 2  # bbox, lms, máscaras, has_box, clase


class EyeDataLoader:
    """Escanea, divide y convierte a ``tf.data`` el dataset de ojos.

    Args:
        data_dir: carpeta con las imágenes y etiquetas (``data/raw/eyes``).
        img_size: (alto, ancho) de entrada del modelo.
        batch_size: tamaño de lote.
        label_format: ``auto``, ``csv``, ``coco``, ``yolo``, ``voc`` o ``folders``.
        seed: semilla para recortes y divisiones.
        crop_scale: lado del recorte alrededor de cada ojo (múltiplo de su tamaño).
        augment_pipeline: pipeline de Albumentations para ``augment=True``
            (por defecto ``get_eye_augmentation_pipeline()``).
    """

    def __init__(
        self,
        data_dir: Path = RAW_DIR / "eyes",
        img_size: tuple[int, int] = (IMG_SIZE_EYE, IMG_SIZE_EYE),
        batch_size: int = BATCH_SIZE,
        label_format: str = "auto",
        seed: int = SEED,
        crop_scale: float = 2.5,
        augment_pipeline: object | None = None,
        group_prefix: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.batch_size = int(batch_size)
        self.label_format = label_format
        self.seed = int(seed)
        self.crop_scale = float(crop_scale)
        self._augment_pipeline = augment_pipeline
        # True: los archivos «<cara>_<algo>.jpg» de una misma cara van juntos al dividir
        self.group_prefix = bool(group_prefix)

    # ------------------------------------------------------------------ #
    # Metadatos
    # ------------------------------------------------------------------ #
    def load_metadata(self) -> pd.DataFrame:
        """Escanea ``data_dir`` y devuelve una fila por ejemplo de entrenamiento.

        Returns:
            DataFrame con columnas ``image_path``, ``image_id``, ``crop_*``,
            ``eye_prob``, ``has_box``, ``bx…bh``, ``lm*_x/y``, ``lm*_m``,
            ``n_eyes`` (ojos en la imagen original) y ``stratum``.

        Raises:
            LabelFormatError: no hay imágenes o etiquetas reconocibles.
            AmbiguousLabelFormat: hay más de un formato; confirmar ``label_format``.
        """
        if self.label_format == "folders":
            return self._metadata_from_folders()
        try:
            samples = load_eye_samples(self.data_dir, self.label_format)
        except AmbiguousLabelFormat:
            raise
        except LabelFormatError:
            if self.label_format == "auto" and self._looks_like_folders():
                return self._metadata_from_folders()
            raise
        rng = np.random.default_rng(self.seed)
        rows: list[dict[str, object]] = []
        for s in samples:
            if s.num_eyes == 0:
                rows.append(self._row(s.image_path, (0, 0, s.width, s.height), None, None, s.num_eyes))
            elif s.num_eyes == 1:
                rows.append(self._row(s.image_path, (0, 0, s.width, s.height), s.boxes[0], s.landmarks[0], 1))
            else:
                for box, lm in zip(s.boxes, s.landmarks):
                    crop = self._crop_around(box, s.width, s.height, rng)
                    rows.append(self._row(s.image_path, crop, box, lm, s.num_eyes))
        return pd.DataFrame(rows)

    def _looks_like_folders(self) -> bool:
        subdirs = (
            {p.name.lower() for p in self.data_dir.iterdir() if p.is_dir()}
            if self.data_dir.exists()
            else set()
        )
        return bool(subdirs & POSITIVE_DIRS) and bool(subdirs & NEGATIVE_DIRS)

    def _metadata_from_folders(self) -> pd.DataFrame:
        """Formato de carpetas por clase: solo etiqueta ojo / no-ojo."""
        rows = []
        for sub in sorted(p for p in self.data_dir.iterdir() if p.is_dir()):
            name = sub.name.lower()
            if name in POSITIVE_DIRS:
                cls = 1.0
            elif name in NEGATIVE_DIRS:
                cls = 0.0
            else:
                raise AmbiguousLabelFormat(
                    f"No sé si la carpeta '{sub.name}' es de ojos o de negativos. "
                    f"Usa nombres como {sorted(POSITIVE_DIRS)[:3]} / {sorted(NEGATIVE_DIRS)[:3]}."
                )
            for img in list_images(sub):
                w, h = image_size(img)
                row = self._row(img.resolve().as_posix(), (0, 0, w, h), None, None, int(cls))
                row["eye_prob"] = cls
                rows.append(row)
        if not rows:
            raise LabelFormatError(f"No hay imágenes en las carpetas de {self.data_dir}")
        return pd.DataFrame(rows)

    def _crop_around(
        self, box: np.ndarray, w_img: int, h_img: int, rng: np.random.Generator
    ) -> tuple[int, int, int, int]:
        """Recorte cuadrado alrededor de un ojo, desplazado al azar y dentro de la imagen."""
        x, y, w, h = (float(v) for v in box)
        side = self.crop_scale * max(w, h)
        side = min(side, float(w_img), float(h_img))
        jitter = rng.uniform(-0.15, 0.15, 2) * side
        cx, cy = x + w / 2 + jitter[0], y + h / 2 + jitter[1]
        x0 = int(np.clip(cx - side / 2, 0, w_img - side))
        y0 = int(np.clip(cy - side / 2, 0, h_img - side))
        # el ojo debe quedar completo dentro del recorte
        x0 = int(np.clip(x0, max(0, x + w - side), min(x, w_img - side)))
        y0 = int(np.clip(y0, max(0, y + h - side), min(y, h_img - side)))
        s = int(round(side))
        return x0, y0, min(s, w_img - x0), min(s, h_img - y0)

    def _row(
        self,
        path: str,
        crop: tuple[int, int, int, int],
        box: np.ndarray | None,
        lm: np.ndarray | None,
        n_eyes: int,
    ) -> dict[str, object]:
        """Construye una fila con objetivos normalizados al recorte."""
        cx, cy, cw, ch = (int(v) for v in crop)
        row: dict[str, object] = {
            "image_path": path,
            "image_id": Path(path).name,
            "crop_x": cx,
            "crop_y": cy,
            "crop_w": cw,
            "crop_h": ch,
            "n_eyes": n_eyes,
            "stratum": str(min(n_eyes, 3)),
        }
        has_box = box is not None
        row["eye_prob"] = 1.0 if has_box else 0.0
        row["has_box"] = 1.0 if has_box else 0.0
        if has_box:
            x, y, w, h = (float(v) for v in box)
            row.update(bx=(x - cx) / cw, by=(y - cy) / ch, bw=w / cw, bh=h / ch)
        else:
            row.update(bx=0.0, by=0.0, bw=0.0, bh=0.0)
        for i in range(NUM_LANDMARKS):
            pt = lm[i] if lm is not None else np.array([np.nan, np.nan])
            ok = has_box and not np.isnan(pt).any()
            row[f"lm{i}_x"] = float((pt[0] - cx) / cw) if ok else 0.0
            row[f"lm{i}_y"] = float((pt[1] - cy) / ch) if ok else 0.0
            row[f"lm{i}_m"] = 1.0 if ok else 0.0
        return row

    # ------------------------------------------------------------------ #
    # División
    # ------------------------------------------------------------------ #
    def split(
        self, df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """División estratificada train/val/test **por imagen**.

        Todas las filas de una misma imagen (varios ojos) quedan en el mismo
        subconjunto para evitar fuga de información. El estrato es el número de
        ojos de la imagen (0, 1, 2, 3+).

        Args:
            df: salida de :meth:`load_metadata`.
            test_size: fracción de imágenes para test.
            val_size: fracción de imágenes para validación.

        Returns:
            (train, val, test) con índices reiniciados.
        """
        grupo = (df["image_path"].map(lambda p: Path(p).name.split("_")[0]) if self.group_prefix
                 else df["image_path"])
        images = df.groupby(grupo, sort=True)["stratum"].first()
        sp = stratified_split(images.tolist(), 1 - test_size - val_size, val_size, test_size, self.seed)
        names = images.index.to_numpy()
        parts = []
        for key in ("train", "val", "test"):
            keep = set(names[sp[key]])
            parts.append(df[grupo.isin(keep)].reset_index(drop=True))
        return parts[0], parts[1], parts[2]

    # ------------------------------------------------------------------ #
    # tf.data
    # ------------------------------------------------------------------ #
    @staticmethod
    def _targets_array(df: pd.DataFrame) -> np.ndarray:
        """Empaqueta los objetivos en una matriz (N, 24)."""
        return df[BOX_COLS + LM_COLS + MASK_COLS + ["has_box", "eye_prob"]].to_numpy(np.float32)

    def _decode(self, path: tf.Tensor) -> tf.Tensor:
        """Lee y decodifica JPG/PNG/BMP/GIF con TF y TIFF con Pillow, siempre RGB uint8."""

        def tf_decode() -> tf.Tensor:
            raw = tf.io.read_file(path)
            # decode_image cubre JPEG/PNG/BMP/GIF; channels=3 convierte gris -> RGB
            return tf.io.decode_image(raw, channels=3, expand_animations=False)

        def tiff_decode() -> tf.Tensor:
            img = tf.numpy_function(lambda p: read_image_rgb(p.decode()), [path], tf.uint8)
            img.set_shape([None, None, 3])
            return img

        is_tiff = tf.strings.regex_full_match(tf.strings.lower(path), r".*\.tiff?$")
        return tf.cond(is_tiff, tiff_decode, tf_decode)

    def build_tfdataset(
        self, df: pd.DataFrame, augment: bool = False, shuffle: bool = False, cache: bool = True
    ) -> tf.data.Dataset:
        """Construye el pipeline ``tf.data``.

        ``leer → recortar → resize bilineal → cache → shuffle → batch →
        aumentar (opcional) → normalizar [0, 1] → prefetch``.

        Args:
            df: subconjunto de :meth:`load_metadata`.
            augment: aplica Albumentations por lote (rotación, flip, brillo, zoom…).
            shuffle: baraja en cada época.
            cache: guarda en memoria las imágenes ya redimensionadas.

        Returns:
            Dataset de ``(imagen, objetivos, pesos)`` por lotes.
        """
        if df.empty:
            raise ValueError("El DataFrame está vacío")
        h, w = self.img_size
        paths = df["image_path"].astype(str).to_numpy()
        crops = df[CROP_COLS].to_numpy(np.int32)
        targets = self._targets_array(df)

        def load(path: tf.Tensor, crop: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            img = self._decode(path)
            img = tf.image.crop_to_bounding_box(img, crop[1], crop[0], crop[3], crop[2])
            img = tf.image.resize(img, (h, w), method="bilinear")
            img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
            return img, target

        ds = tf.data.Dataset.from_tensor_slices((paths, crops, targets))
        if shuffle and not cache:
            # Se barajan las RUTAS antes de leer: barajar imágenes ya decodificadas con un buffer
            # del tamaño del dataset ocupa GB de RAM (14 k × 224² × 3 float32 ≈ 8.6 GB).
            ds = ds.shuffle(len(df), seed=self.seed, reshuffle_each_iteration=True)
        ds = ds.map(load, num_parallel_calls=AUTOTUNE)
        if cache:
            ds = ds.cache()
            if shuffle:
                ds = ds.shuffle(len(df), seed=self.seed, reshuffle_each_iteration=True)
        ds = ds.batch(self.batch_size)
        if augment:
            from src.data.augment import augment_batch, get_eye_augmentation_pipeline

            pipeline = self._augment_pipeline or get_eye_augmentation_pipeline()
            ds = ds.map(lambda x, y: augment_batch(x, y, pipeline), num_parallel_calls=AUTOTUNE)
        ds = ds.map(split_targets, num_parallel_calls=AUTOTUNE)
        return ds.prefetch(AUTOTUNE)


def split_targets(
    images: tf.Tensor, packed: tf.Tensor
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Separa la matriz de objetivos en el diccionario que espera el modelo.

    Args:
        images: (B, H, W, 3) uint8.
        packed: (B, 24) con bbox, landmarks, máscaras, has_box y clase.

    Returns:
        (imágenes float32 en [0, 1], objetivos, pesos por salida).
    """
    k = NUM_LANDMARKS
    x = tf.cast(images, tf.float32) / 255.0
    has_box = packed[:, 4 + 3 * k]
    y = {
        "bbox": packed[:, :4],
        "landmarks": packed[:, 4 : 4 + 3 * k],  # 12 coordenadas + 6 máscaras
        "eye_prob": packed[:, -1:],
    }
    sw = {"bbox": has_box, "landmarks": has_box, "eye_prob": tf.ones_like(has_box)}
    return x, y, sw


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Resumen del dataset de ojos")
    ap.add_argument("--data-dir", default=str(RAW_DIR / "eyes"))
    args = ap.parse_args()
    loader = EyeDataLoader(Path(args.data_dir))
    meta = loader.load_metadata()
    tr, va, te = loader.split(meta)
    print(meta.describe(include="all").T.head(12))
    print(f"filas: {len(meta)} | train {len(tr)} · val {len(va)} · test {len(te)}")
