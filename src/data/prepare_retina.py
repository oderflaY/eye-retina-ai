"""Prepara los datasets de vasos de retina (FIVES y HRF) para entrenar la U-Net.

Uso::

    python -m src.data.prepare_retina                       # 1024 px, CLAHE
    python -m src.data.prepare_retina --size 768 --no-clahe --limit 20

Entradas (``data/raw/retina``, enlaces a ``../datasets``; se aceptan las
carpetas ``extraido/`` que deja el descompresor):

* FIVES: ``{train,test}/{Original,Ground truth}/<n>_<A|D|G|N>.png``
  (A = DMAE, D = retinopatía diabética, G = glaucoma, N = normal).
* HRF:  ``all/{images,manual1,mask}`` con ``<n>_<h|g|dr>``.

Salidas en ``data/processed/retina/``:

* ``images/<id>.png`` (RGB), ``vessels/<id>.png`` y ``fov/<id>.png`` ({0, 255}).
* ``manifest.csv``: ``id, dataset, category, split, image, vessel, fov``.
* ``prepare_config.json``: tamaño y CLAHE (los lee la inferencia).

División: FIVES train → train/val estratificado por categoría (15 % val),
FIVES test → ``test``, HRF completo → ``external`` (validación externa).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, RAW_DIR, SEED
from src.data.io import read_image_rgb
from src.data.retina_preprocess import preprocess_fundus

FIVES_CATEGORY = {"A": "dmae", "D": "retinopatia_diabetica", "G": "glaucoma", "N": "normal"}
HRF_CATEGORY = {"h": "normal", "g": "glaucoma", "dr": "retinopatia_diabetica"}


def _find(root: Path, pattern: str) -> Path | None:
    """Primera carpeta bajo ``root`` (incluye enlaces) que coincide con ``pattern``."""
    hits = sorted(p for p in root.glob(pattern) if p.is_dir())
    return hits[0] if hits else None


def scan_fives(root: Path) -> list[dict]:
    """Pares imagen/máscara de FIVES."""
    rows = []
    for split in ("train", "test"):
        orig = _find(root, f"**/{split}/Original")
        gt = _find(root, f"**/{split}/Ground truth")
        if orig is None or gt is None:
            continue
        for img in sorted(orig.glob("*.png")):
            mask = gt / img.name
            if not mask.exists():
                continue
            cat = FIVES_CATEGORY.get(img.stem.split("_")[-1].upper(), "desconocida")
            rows.append(
                {"id": f"fives_{img.stem}", "dataset": "FIVES", "category": cat, "orig_split": split,
                 "src_image": str(img), "src_vessel": str(mask), "src_fov": ""}
            )
    return rows


def scan_hrf(root: Path) -> list[dict]:
    """Ternas imagen/vasos/FOV de HRF."""
    images = _find(root, "**/all/images")
    if images is None:
        return []
    base = images.parent
    rows = []
    for img in sorted(p for p in images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif")):
        vessel = base / "manual1" / f"{img.stem}.tif"
        fov = base / "mask" / f"{img.stem}_mask.tif"
        if not vessel.exists():
            continue
        rows.append(
            {"id": f"hrf_{img.stem}", "dataset": "HRF", "category": HRF_CATEGORY.get(img.stem.split("_")[-1], "?"),
             "orig_split": "external", "src_image": str(img), "src_vessel": str(vessel),
             "src_fov": str(fov) if fov.exists() else ""}
        )
    return rows


def assign_splits(df: pd.DataFrame, val_size: float, seed: int) -> pd.DataFrame:
    """train/val estratificado sobre FIVES-train; el resto conserva su división."""
    df = df.copy()
    df["split"] = df["orig_split"]
    tr = df[(df.dataset == "FIVES") & (df.orig_split == "train")]
    if len(tr) >= 10 and val_size > 0:
        counts = tr["category"].value_counts()
        n_val = int(np.ceil(val_size * len(tr)))
        # estratificar exige ≥ 2 por categoría y al menos una imagen de val por categoría
        strat = tr["category"] if counts.min() >= 2 and n_val >= len(counts) else None
        _, val_idx = train_test_split(tr.index, test_size=val_size, random_state=seed, stratify=strat)
        df.loc[val_idx, "split"] = "val"
    return df


def _process_one(args: tuple[dict, str, int, bool]) -> dict:
    row, out_dir, size, clahe = args
    out = Path(out_dir)
    img = read_image_rgb(row["src_image"])
    vessel = cv2.imread(row["src_vessel"], cv2.IMREAD_UNCHANGED)
    if vessel is None:  # algunos TIF/GIF no los abre OpenCV
        from PIL import Image

        vessel = np.array(Image.open(row["src_vessel"]))
    fov = cv2.imread(row["src_fov"], cv2.IMREAD_UNCHANGED) if row["src_fov"] else None
    im, fv, vs, g = preprocess_fundus(img, size=size, fov=fov, vessels=vessel, clahe=clahe)
    paths = {k: out / k / f"{row['id']}.png" for k in ("images", "vessels", "fov")}
    cv2.imwrite(str(paths["images"]), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(paths["vessels"]), vs * 255)
    cv2.imwrite(str(paths["fov"]), fv * 255)
    return {
        "id": row["id"], "dataset": row["dataset"], "category": row["category"], "split": row["split"],
        "image": str(paths["images"].relative_to(out)), "vessel": str(paths["vessels"].relative_to(out)),
        "fov": str(paths["fov"].relative_to(out)), "orig_w": g.orig_w, "orig_h": g.orig_h,
        "vessel_frac": float(vs[fv > 0].mean()) if fv.any() else 0.0,
    }


def prepare(
    raw_dir: Path, out_dir: Path, size: int = 1024, clahe: bool = True, val_size: float = 0.15,
    seed: int = SEED, limit: int | None = None, workers: int = 4,
) -> pd.DataFrame:
    """Escanea, divide, preprocesa y escribe el manifiesto. Devuelve el manifiesto."""
    rows = scan_fives(raw_dir / "FIVES") + scan_hrf(raw_dir / "HRF")
    if not rows:
        raise FileNotFoundError(f"No se encontró FIVES ni HRF en {raw_dir}")
    df = assign_splits(pd.DataFrame(rows), val_size, seed)
    if limit:
        df = df.groupby("split", group_keys=False).head(limit)
    for k in ("images", "vessels", "fov"):
        (out_dir / k).mkdir(parents=True, exist_ok=True)
    jobs = [(r, str(out_dir), size, clahe) for r in df.to_dict("records")]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            out_rows = list(ex.map(_process_one, jobs, chunksize=4))
    else:
        out_rows = [_process_one(j) for j in jobs]
    manifest = pd.DataFrame(out_rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    (out_dir / "prepare_config.json").write_text(
        json.dumps({"size": size, "clahe": clahe, "val_size": val_size, "seed": seed}, indent=2)
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_DIR / "retina")
    ap.add_argument("--out", type=Path, default=PROCESSED_DIR / "retina")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=None, help="máx. imágenes por división (pruebas rápidas)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    m = prepare(a.raw, a.out, a.size, not a.no_clahe, a.val_size, limit=a.limit, workers=a.workers)
    print(m.groupby(["split", "dataset", "category"]).size().to_string())
    print(f"\nFracción media de vasos dentro del FOV: {m.vessel_frac.mean():.3f}")
    print(f"Manifiesto: {a.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
