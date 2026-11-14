"""Prepara el dataset del analizador de ojo externo (fotos del ojo con la cámara del teléfono).

Fuentes (HuggingFace, solo investigación; ver docs/PLAN.md):

* ``chaudharyusman/Pink-Eye-z``  → ``infected`` = enrojecimiento, ``normal`` = normal.
* ``mostafasmart/pterygium2class`` → pterigión; sus negativos traen la enfermedad en el nombre:
  ``cataract*`` = catarata, ``blepharitis*`` = enrojecimiento, ``normal*`` y sin nombre = normal.

Las fuentes traen copias aumentadas (Roboflow ``*.rf.<hash>``, ``-Copy``, ``pterygium_res``,
espejos). Se agrupan por nombre base **y** por huella visual (dHash, insensible a espejo) con
union-find, y la división train/val/test se hace por grupo para que ninguna copia de una foto de
prueba se vea al entrenar.

    python -m src.data.prepare_ojo_externo --raw ~/TemrariosSpace/datasets/OjoExterno
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import PROJECT_ROOT

CLASES = ("normal", "enrojecimiento", "pterigion", "catarata")
LADO = 288  # se guarda cuadrado (letterbox); el entrenamiento recorta 256
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def leer(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or min(img.shape[:2]) < 40:
        return None
    return img


def cuadrar(img: np.ndarray, lado: int = LADO) -> np.ndarray:
    """Escala sin deformar y rellena con el borde replicado hasta un cuadrado."""
    h, w = img.shape[:2]
    s = lado / max(h, w)
    img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    t, l = (lado - h) // 2, (lado - w) // 2
    return cv2.copyMakeBorder(img, t, lado - h - t, l, lado - w - l, cv2.BORDER_REPLICATE)


def dhash(img: np.ndarray) -> bytes:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    a = (g[:, 1:] > g[:, :-1]).tobytes()
    b = (g[:, ::-1][:, 1:] > g[:, ::-1][:, :-1]).tobytes()
    return min(a, b)


def base_nombre(name: str) -> str:
    n = name.lower()
    n = re.sub(r"\.rf\.[0-9a-f]+", "", n)
    n = re.sub(r"\.(jpg|jpeg|png|webp|bmp)$", "", n)
    n = re.sub(r"_(jpg|jpeg|png|webp)$", "", n)
    n = re.sub(r"[ _-]*\(?copy\)?", "", n)
    n = re.sub(r"^(train|test)_\d+_", "", n)  # índice que puse al extraer el parquet
    n = n.replace("pterygium_res", "pterygium")
    return n


def listar(raw: Path) -> list[tuple[Path, str, str]]:
    out = []
    for d, clase in (("infected", "enrojecimiento"), ("normal", "normal")):
        for p in sorted((raw / "pinkeye").glob(f"*/{d}/*")):
            if p.suffix.lower() in EXTS:
                out.append((p, clase, "pinkeye"))
    for p in sorted((raw / "pterigion/imgs/pterygium").iterdir()):
        if p.suffix.lower() in EXTS:
            out.append((p, "pterigion", "pterigion"))
    for p in sorted((raw / "pterigion/imgs/notPtery").iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        n = p.name.lower()
        clase = "catarata" if "cataract" in n else "enrojecimiento" if "blephar" in n else "normal"
        out.append((p, clase, "pterigion"))
    return out


class UF:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.p[self.find(a)] = self.find(b)


def prepare(raw: Path, out: Path, seed: int = 42) -> pd.DataFrame:
    (out / "images").mkdir(parents=True, exist_ok=True)
    filas = []
    uf = UF()
    for i, (p, clase, fuente) in enumerate(listar(raw)):
        img = leer(p)
        if img is None:
            continue
        rel = f"images/{i:05d}.jpg"
        cv2.imwrite(str(out / rel), cuadrar(img), [cv2.IMWRITE_JPEG_QUALITY, 93])
        k_nombre = f"n:{fuente}:{base_nombre(p.name)}"
        k_hash = f"h:{dhash(img).hex()}"
        uf.union(rel, k_nombre)
        uf.union(rel, k_hash)
        filas.append({"image": rel, "clase": clase, "fuente": fuente, "origen": p.name})
    df = pd.DataFrame(filas)
    raices = {r: i for i, r in enumerate(sorted({uf.find(x) for x in df.image}))}
    df["grupo"] = [raices[uf.find(x)] for x in df.image]
    # Un grupo con clases mezcladas (misma foto en dos carpetas) se descarta: etiqueta dudosa.
    mezcla = df.groupby("grupo").clase.nunique()
    df = df[~df.grupo.isin(mezcla[mezcla > 1].index)].reset_index(drop=True)

    # 7 pliegues estratificados por clase y agrupados: 1 test, 1 val, 5 train (~70/15/15)
    y = df.clase.map({c: i for i, c in enumerate(CLASES)}).to_numpy()
    folds = list(StratifiedGroupKFold(7, shuffle=True, random_state=seed).split(df, y, df.grupo))
    df["split"] = "train"
    df.loc[folds[0][1], "split"] = "test"
    df.loc[folds[1][1], "split"] = "val"
    df.to_csv(out / "manifest.csv", index=False)
    resumen = {
        "clases": list(CLASES),
        "imagenes": len(df),
        "grupos": int(df.grupo.nunique()),
        "descartadas_por_mezcla": int(mezcla[mezcla > 1].sum()),
        "por_split": {s: dict(Counter(df[df.split == s].clase)) for s in ("train", "val", "test")},
        "grupos_por_clase": {c: int(df[df.clase == c].grupo.nunique()) for c in CLASES},
    }
    (out / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path.home() / "TemrariosSpace/datasets/OjoExterno")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/processed/ojo_externo")
    a = ap.parse_args()
    prepare(a.raw.expanduser(), a.out)


if __name__ == "__main__":
    main()
