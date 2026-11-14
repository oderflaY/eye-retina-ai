"""Detección automática y lectura de etiquetas de ojos.

Formatos soportados (bounding box + hasta 6 landmarks por ojo):

* **csv**  – una fila por ojo: ``image, x, y, w, h`` (o ``xmin, ymin, xmax, ymax``)
  y opcionalmente ``<landmark>_x, <landmark>_y`` o ``lm0_x … lm5_y``.
* **coco** – JSON con ``images``, ``annotations`` (``bbox`` + ``keypoints``).
* **yolo** – un ``.txt`` por imagen: ``cls cx cy w h [px py (v)] x6`` normalizado.
* **voc**  – un ``.xml`` Pascal VOC por imagen, objetos cuyo nombre contiene "eye"/"ojo".

Todas las coordenadas se convierten a píxeles de la imagen original. Un
landmark ausente se guarda como ``NaN``.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.io import IMAGE_EXTS, image_size, list_images

LANDMARK_NAMES: tuple[str, ...] = (
    "inner_corner",  # esquina interna (lagrimal)
    "outer_corner",  # esquina externa
    "upper_lid",  # punto medio del párpado superior
    "lower_lid",  # punto medio del párpado inferior
    "pupil",  # centro de la pupila
    "iris",  # borde lateral del iris
)
NUM_LANDMARKS = len(LANDMARK_NAMES)
FORMATS = ("csv", "coco", "yolo", "voc")
_EYE_WORDS = ("eye", "ojo")


class LabelFormatError(ValueError):
    """No se encontró un formato de etiquetas reconocible."""


class AmbiguousLabelFormat(LabelFormatError):
    """Hay más de una interpretación posible: se debe confirmar con el usuario."""


@dataclass
class EyeSample:
    """Una imagen con sus ojos anotados (coordenadas en píxeles originales)."""

    image_path: str
    width: int
    height: int
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros((0, NUM_LANDMARKS, 2), np.float32))

    @property
    def num_eyes(self) -> int:
        """Número de ojos anotados."""
        return int(self.boxes.shape[0])

    def to_dict(self) -> dict[str, Any]:
        """Serializa a JSON (NaN -> null)."""
        lms = [[[None if np.isnan(v) else float(v) for v in pt] for pt in eye] for eye in self.landmarks]
        return {
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "boxes": self.boxes.astype(float).round(2).tolist(),
            "landmarks": lms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EyeSample":
        """Reconstruye una muestra desde su forma JSON."""
        boxes = np.asarray(d["boxes"], np.float32).reshape(-1, 4)
        lms = np.asarray(
            [[[np.nan if v is None else v for v in pt] for pt in eye] for eye in d["landmarks"]],
            np.float32,
        ).reshape(-1, NUM_LANDMARKS, 2)
        return cls(d["image_path"], int(d["width"]), int(d["height"]), boxes, lms)


# --------------------------------------------------------------------------- #
# Detección del formato
# --------------------------------------------------------------------------- #
_IMAGE_COLS = ("image", "filename", "file", "image_path", "path", "img", "image_name")
_XYWH = ("x", "y", "w", "h")
_XYXY_VARIANTS = (
    ("xmin", "ymin", "xmax", "ymax"),
    ("x_min", "y_min", "x_max", "y_max"),
    ("x1", "y1", "x2", "y2"),
)


def _csv_is_eye_labels(path: Path) -> bool:
    try:
        cols = {c.strip().lower() for c in pd.read_csv(path, nrows=1).columns}
    except Exception:  # noqa: BLE001
        return False
    has_img = any(c in cols for c in _IMAGE_COLS)
    has_box = set(_XYWH) <= cols or any(set(v) <= cols for v in _XYXY_VARIANTS)
    return has_img and has_box


def _json_is_coco(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(data, dict) and "images" in data and "annotations" in data


def _txt_is_yolo(path: Path) -> bool:
    if path.name.lower() in {"classes.txt", "readme.txt", "license.txt", "requirements.txt"}:
        return False
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                toks = line.split()
                return len(toks) >= 5 and all(_is_number(t) for t in toks)
        return True  # un .txt vacío = imagen sin ojos (válido en YOLO)
    except Exception:  # noqa: BLE001
        return False


def _xml_is_voc(path: Path) -> bool:
    try:
        return ET.parse(path).getroot().tag == "annotation"
    except Exception:  # noqa: BLE001
        return False


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def find_label_files(raw_dir: str | Path) -> dict[str, list[Path]]:
    """Agrupa los archivos de etiquetas encontrados por formato."""
    raw_dir = Path(raw_dir)
    found: dict[str, list[Path]] = {f: [] for f in FORMATS}
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext == ".csv" and _csv_is_eye_labels(p):
            found["csv"].append(p)
        elif ext == ".json" and _json_is_coco(p):
            found["coco"].append(p)
        elif ext == ".txt" and _txt_is_yolo(p):
            found["yolo"].append(p)
        elif ext == ".xml" and _xml_is_voc(p):
            found["voc"].append(p)
    return found


def detect_format(raw_dir: str | Path) -> str:
    """Detecta el formato de etiquetas presente en ``raw_dir``.

    Raises:
        LabelFormatError: si no hay imágenes o no hay etiquetas reconocibles.
        AmbiguousLabelFormat: si hay etiquetas en más de un formato.
    """
    raw_dir = Path(raw_dir)
    images = list_images(raw_dir)
    if not images:
        raise LabelFormatError(
            f"No hay imágenes en {raw_dir}. Coloca ahí las imágenes de ojos/rostros "
            "con sus etiquetas (ver docs/FORMATO_ETIQUETAS.md)."
        )
    found = {k: v for k, v in find_label_files(raw_dir).items() if v}
    if len(found) == 1:
        return next(iter(found))
    if not found:
        mask_like = [p for p in images if "mask" in p.as_posix().lower()]
        hint = (
            " Se encontraron máscaras: parece un dataset de segmentación (retina), "
            "no de detección de ojos."
            if mask_like
            else ""
        )
        raise LabelFormatError(
            f"Hay {len(images)} imágenes en {raw_dir} pero no se reconoció ningún "
            f"archivo de etiquetas (CSV, COCO JSON, YOLO TXT, VOC XML).{hint} "
            "Confirma el formato de las etiquetas antes de continuar."
        )
    detail = ", ".join(f"{k} ({len(v)} archivos)" for k, v in found.items())
    raise AmbiguousLabelFormat(
        f"Se encontraron etiquetas en varios formatos: {detail}. Confirma cuál usar "
        "con `data.label_format` en el YAML (p. ej. --override data.label_format=coco)."
    )


# --------------------------------------------------------------------------- #
# Lectores
# --------------------------------------------------------------------------- #
class _ImageIndex:
    """Resuelve nombres de archivo de las etiquetas a rutas reales."""

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.by_rel: dict[str, Path] = {}
        self.by_name: dict[str, list[Path]] = {}
        self.by_stem: dict[str, list[Path]] = {}
        for p in list_images(raw_dir):
            self.by_rel[p.relative_to(raw_dir).as_posix()] = p
            self.by_name.setdefault(p.name, []).append(p)
            self.by_stem.setdefault(p.stem, []).append(p)
        self._sizes: dict[Path, tuple[int, int]] = {}

    def resolve(self, ref: str, base: Path | None = None) -> Path | None:
        ref = str(ref).replace("\\", "/").strip()
        cands = [Path(ref)]
        if base is not None:
            cands.append(base / ref)
        cands.append(self.raw_dir / ref)
        for c in cands:
            if c.is_file() and c.suffix.lower() in IMAGE_EXTS:
                return c
        if ref in self.by_rel:
            return self.by_rel[ref]
        name = Path(ref).name
        hits = self.by_name.get(name) or self.by_stem.get(Path(ref).stem) or []
        if len(hits) > 1:
            raise AmbiguousLabelFormat(
                f"La etiqueta '{ref}' coincide con varias imágenes: "
                f"{[h.as_posix() for h in hits[:3]]}. Usa rutas relativas en las etiquetas."
            )
        return hits[0] if hits else None

    def size(self, path: Path) -> tuple[int, int]:
        if path not in self._sizes:
            self._sizes[path] = image_size(path)
        return self._sizes[path]


def _empty_landmarks(n: int) -> np.ndarray:
    return np.full((n, NUM_LANDMARKS, 2), np.nan, np.float32)


def _landmark_columns(cols: list[str]) -> list[tuple[str, str] | None]:
    """Busca las columnas x/y de cada landmark (por nombre o por índice)."""
    out: list[tuple[str, str] | None] = []
    for i, name in enumerate(LANDMARK_NAMES):
        pair = None
        for stem in (name, f"lm{i}", f"kp{i}", f"landmark{i}", f"p{i}"):
            if f"{stem}_x" in cols and f"{stem}_y" in cols:
                pair = (f"{stem}_x", f"{stem}_y")
                break
        out.append(pair)
    return out


def parse_csv(paths: list[Path], index: _ImageIndex) -> list[EyeSample]:
    """Lee uno o varios CSV con una fila por ojo."""
    groups: dict[Path, list[tuple[np.ndarray, np.ndarray]]] = {}
    for csv_path in paths:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        cols = list(df.columns)
        img_col = next(c for c in _IMAGE_COLS if c in cols)
        cls_col = next((c for c in ("class", "label", "category") if c in cols), None)
        if cls_col is not None:
            values = df[cls_col].dropna().astype(str).str.lower().unique()
            eye_vals = [v for v in values if any(w in v for w in _EYE_WORDS)]
            if len(values) > 1 and not eye_vals:
                raise AmbiguousLabelFormat(
                    f"{csv_path.name}: la columna '{cls_col}' tiene clases {list(values)[:6]} "
                    "y ninguna parece 'eye'/'ojo'. Indica cuál corresponde a ojos."
                )
            if len(values) > 1:
                keep = df[cls_col].astype(str).str.lower().isin(eye_vals) | df[cls_col].isna()
                df = df[keep]
        xyxy = next((v for v in _XYXY_VARIANTS if set(v) <= set(cols)), None)
        lm_cols = _landmark_columns(cols)
        box_cols = list(_XYWH) if set(_XYWH) <= set(cols) else list(xyxy or ())
        normalized = bool(df[box_cols].max(skipna=True).max() <= 1.0)
        for _, row in df.iterrows():
            img = index.resolve(row[img_col], csv_path.parent)
            if img is None:
                raise LabelFormatError(f"{csv_path.name}: no se encontró la imagen '{row[img_col]}'")
            entry = groups.setdefault(img, [])
            vals = row[box_cols].to_numpy(dtype=np.float64)
            if np.isnan(vals).any():
                continue  # fila sin caja = imagen negativa (sin ojos)
            w_img, h_img = index.size(img)
            sx, sy = (w_img, h_img) if normalized else (1.0, 1.0)
            if xyxy is not None and box_cols == list(xyxy):
                x0, y0, x1, y1 = vals
                box = np.array([x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy])
            else:
                box = vals * np.array([sx, sy, sx, sy])
            lm = np.full((NUM_LANDMARKS, 2), np.nan)
            for k, pair in enumerate(lm_cols):
                if pair is not None:
                    lm[k] = [float(row[pair[0]]) * sx, float(row[pair[1]]) * sy]
            entry.append((box, lm))
    return [_make_sample(img, eyes, index) for img, eyes in groups.items()]


def parse_coco(paths: list[Path], index: _ImageIndex) -> list[EyeSample]:
    """Lee anotaciones COCO (``bbox`` en píxeles, ``keypoints`` x, y, v)."""
    samples: list[EyeSample] = []
    for json_path in paths:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        cats = {c["id"]: str(c.get("name", "")).lower() for c in data.get("categories", [])}
        eye_ids = {cid for cid, n in cats.items() if any(w in n for w in _EYE_WORDS)}
        if len(cats) > 1 and not eye_ids:
            raise AmbiguousLabelFormat(
                f"{json_path.name}: categorías {list(cats.values())[:6]} y ninguna es 'eye'/'ojo'. "
                "Indica qué categoría corresponde a ojos."
            )
        keep_ids = eye_ids or set(cats) or None
        anns: dict[int, list[dict[str, Any]]] = {}
        for a in data["annotations"]:
            if keep_ids is None or a.get("category_id") in keep_ids:
                anns.setdefault(a["image_id"], []).append(a)
        for im in data["images"]:
            img = index.resolve(im["file_name"], json_path.parent)
            if img is None:
                raise LabelFormatError(f"{json_path.name}: no se encontró '{im['file_name']}'")
            eyes = []
            for a in anns.get(im["id"], []):
                box = np.asarray(a["bbox"], np.float64)
                lm = np.full((NUM_LANDMARKS, 2), np.nan)
                kps = a.get("keypoints") or []
                if len(kps) >= 3 * NUM_LANDMARKS:
                    k = np.asarray(kps[: 3 * NUM_LANDMARKS], np.float64).reshape(-1, 3)
                    vis = k[:, 2] > 0
                    lm[vis] = k[vis, :2]
                elif len(kps) >= 2 * NUM_LANDMARKS:
                    lm = np.asarray(kps[: 2 * NUM_LANDMARKS], np.float64).reshape(-1, 2)
                eyes.append((box, lm))
            samples.append(_make_sample(img, eyes, index))
    return samples


def _yolo_label_for(img: Path, raw_dir: Path) -> Path | None:
    rel = img.relative_to(raw_dir)
    parts = list(rel.parts)
    cands = [img.with_suffix(".txt")]
    if "images" in parts:
        i = parts.index("images")
        cands.append(raw_dir.joinpath(*parts[:i], "labels", *parts[i + 1 :]).with_suffix(".txt"))
    cands.append(raw_dir / "labels" / (img.stem + ".txt"))
    return next((c for c in cands if c.is_file()), None)


def parse_yolo(raw_dir: Path, index: _ImageIndex) -> list[EyeSample]:
    """Lee etiquetas YOLO (y YOLO-pose) normalizadas."""
    samples: list[EyeSample] = []
    class_ids: set[int] = set()
    rows: list[tuple[Path, list[list[float]]]] = []
    for img in list_images(raw_dir):
        lbl = _yolo_label_for(img, raw_dir)
        if lbl is None:
            continue
        lines = [ln.split() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        vals = [[float(t) for t in ln] for ln in lines]
        class_ids.update(int(v[0]) for v in vals)
        rows.append((img, vals))
    eye_cls = _yolo_eye_class(raw_dir, class_ids)
    for img, vals in rows:
        w_img, h_img = index.size(img)
        eyes = []
        for v in vals:
            if eye_cls is not None and int(v[0]) != eye_cls:
                continue
            cx, cy, bw, bh = v[1:5]
            box = np.array([(cx - bw / 2) * w_img, (cy - bh / 2) * h_img, bw * w_img, bh * h_img])
            rest = np.asarray(v[5:], np.float64)
            lm = np.full((NUM_LANDMARKS, 2), np.nan)
            if rest.size >= 3 * NUM_LANDMARKS:
                k = rest[: 3 * NUM_LANDMARKS].reshape(-1, 3)
                vis = k[:, 2] > 0
                lm[vis] = k[vis, :2] * [w_img, h_img]
            elif rest.size >= 2 * NUM_LANDMARKS:
                lm = rest[: 2 * NUM_LANDMARKS].reshape(-1, 2) * [w_img, h_img]
            eyes.append((box, lm))
        samples.append(_make_sample(img, eyes, index))
    return samples


def _yolo_eye_class(raw_dir: Path, class_ids: set[int]) -> int | None:
    """Elige el id de clase 'ojo' cuando hay varias clases."""
    if len(class_ids) <= 1:
        return None
    names_file = next(iter(raw_dir.rglob("classes.txt")), None)
    if names_file is not None:
        names = [n.strip().lower() for n in names_file.read_text(encoding="utf-8").splitlines()]
        for i, n in enumerate(names):
            if any(w in n for w in _EYE_WORDS):
                return i
    raise AmbiguousLabelFormat(
        f"Las etiquetas YOLO tienen varias clases {sorted(class_ids)} y no hay un "
        "classes.txt que indique cuál es 'eye'. Confirma el id de la clase ojo."
    )


def parse_voc(paths: list[Path], index: _ImageIndex) -> list[EyeSample]:
    """Lee XML Pascal VOC; solo objetos cuyo nombre contiene 'eye'/'ojo'."""
    samples: list[EyeSample] = []
    for xml_path in paths:
        root = ET.parse(xml_path).getroot()
        fname = root.findtext("filename") or xml_path.with_suffix(".jpg").name
        img = index.resolve(fname, xml_path.parent)
        if img is None:
            raise LabelFormatError(f"{xml_path.name}: no se encontró '{fname}'")
        objs = root.findall("object")
        names = {(o.findtext("name") or "").lower() for o in objs}
        eyes = []
        for o in objs:
            name = (o.findtext("name") or "").lower()
            if len(names) > 1 and not any(w in name for w in _EYE_WORDS):
                continue
            bb = o.find("bndbox")
            if bb is None:
                continue
            x0, y0, x1, y1 = (float(bb.findtext(k) or 0) for k in ("xmin", "ymin", "xmax", "ymax"))
            eyes.append((np.array([x0, y0, x1 - x0, y1 - y0]), np.full((NUM_LANDMARKS, 2), np.nan)))
        samples.append(_make_sample(img, eyes, index))
    return samples


def _make_sample(img: Path, eyes: list[tuple[np.ndarray, np.ndarray]], index: _ImageIndex) -> EyeSample:
    """Crea una muestra recortando cajas al borde de la imagen y quitando degeneradas."""
    w_img, h_img = index.size(img)
    boxes, lms = [], []
    for box, lm in eyes:
        x, y, w, h = box
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(w_img), x + w), min(float(h_img), y + h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        boxes.append([x0, y0, x1 - x0, y1 - y0])
        lms.append(lm)
    return EyeSample(
        image_path=img.resolve().as_posix(),
        width=w_img,
        height=h_img,
        boxes=np.asarray(boxes, np.float32).reshape(-1, 4),
        landmarks=np.asarray(lms, np.float32).reshape(-1, NUM_LANDMARKS, 2) if lms else _empty_landmarks(0),
    )


def load_eye_samples(raw_dir: str | Path, label_format: str = "auto") -> list[EyeSample]:
    """Punto de entrada: detecta el formato (si es ``auto``) y lee todas las muestras.

    Raises:
        LabelFormatError / AmbiguousLabelFormat: ver :func:`detect_format`.
    """
    raw_dir = Path(raw_dir)
    fmt = detect_format(raw_dir) if label_format == "auto" else label_format
    if fmt not in FORMATS:
        raise LabelFormatError(f"Formato desconocido '{fmt}'. Opciones: {FORMATS}")
    index = _ImageIndex(raw_dir)
    files = find_label_files(raw_dir)[fmt] if fmt != "yolo" else []
    if fmt == "csv":
        samples = parse_csv(files, index)
    elif fmt == "coco":
        samples = parse_coco(files, index)
    elif fmt == "yolo":
        samples = parse_yolo(raw_dir, index)
    else:
        samples = parse_voc(files, index)
    if not samples:
        raise LabelFormatError(f"No se pudo leer ninguna muestra con el formato '{fmt}'.")
    return sorted(samples, key=lambda s: s.image_path)
