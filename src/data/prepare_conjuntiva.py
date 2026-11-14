"""Prepara la segmentación de la conjuntiva palpebral (para medir palidez → anemia).

Fuente: ``bharath56858/Anemia-seg`` (HuggingFace, Apache-2.0): fotos del ojo con el párpado
inferior evertido y la conjuntiva marcada como polígono YOLO. Se respeta su división
train/val/test (los nombres no permiten agrupar las aumentaciones de una misma foto).

Salida con el mismo formato que ``prepare_retina`` para reutilizar ``train_retina``:
``data/processed/conjuntiva/{images,vessels,fov}/*.png`` + ``manifest.csv``
(la columna ``vessel`` es la máscara de conjuntiva; ``fov`` es toda la imagen útil).

    python -m src.data.prepare_conjuntiva --raw ~/TemrariosSpace/datasets/Anemia/seg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

LADO = 512


def letterbox(img: np.ndarray, lado: int, interp: int, relleno: int | tuple = 0) -> tuple[np.ndarray, np.ndarray]:
    """Escala sin deformar a un cuadrado; devuelve la imagen y la máscara del área útil."""
    h, w = img.shape[:2]
    s = lado / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = cv2.resize(img, (nw, nh), interpolation=interp)
    t, l = (lado - nh) // 2, (lado - nw) // 2
    out = cv2.copyMakeBorder(img, t, lado - nh - t, l, lado - nw - l, cv2.BORDER_CONSTANT, value=relleno)
    util = np.zeros((lado, lado), np.uint8)
    util[t : t + nh, l : l + nw] = 255
    return out, util


def poligonos(txt: Path, w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    if txt.exists():
        for linea in txt.read_text().splitlines():
            v = linea.split()
            if len(v) < 7:
                continue
            pts = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).round().astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


def prepare(raw: Path, out: Path, lado: int = LADO) -> pd.DataFrame:
    for d in ("images", "vessels", "fov"):
        (out / d).mkdir(parents=True, exist_ok=True)
    filas = []
    for split_src, split in (("train", "train"), ("val", "val"), ("test", "test")):
        for p in sorted((raw / split_src / "images").glob("*.jpg")):
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            m = poligonos(raw / split_src / "labels" / f"{p.stem}.txt", w, h)
            if m.sum() == 0:
                continue
            im2, util = letterbox(img, lado, cv2.INTER_AREA)
            m2, _ = letterbox(m, lado, cv2.INTER_NEAREST)
            n = p.stem
            cv2.imwrite(str(out / "images" / f"{n}.png"), im2)
            cv2.imwrite(str(out / "vessels" / f"{n}.png"), (m2 > 127).astype(np.uint8) * 255)
            cv2.imwrite(str(out / "fov" / f"{n}.png"), util)
            filas.append({"id": n, "image": f"images/{n}.png", "vessel": f"vessels/{n}.png", "fov": f"fov/{n}.png",
                          "split": split, "dataset": "anemia_seg", "category": "conjuntiva",
                          "frac_conjuntiva": round(float((m2 > 127).mean()), 4)})
    df = pd.DataFrame(filas)
    df.to_csv(out / "manifest.csv", index=False)
    (out / "prepare_config.json").write_text(json.dumps({"size": lado, "clahe": False, "fuente": str(raw)}, indent=2))
    print(df.split.value_counts().to_dict(), "conjuntiva media:", round(df.frac_conjuntiva.mean(), 3))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path.home() / "TemrariosSpace/datasets/Anemia/seg")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/processed/conjuntiva")
    a = ap.parse_args()
    prepare(a.raw.expanduser(), a.out)


if __name__ == "__main__":
    main()
