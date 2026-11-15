"""Corre en el VPS (docker python:3.12-slim + pyarrow pillow): reduce los datasets de encías
para traerlos por una conexión lenta.

- Dentalai (Frallex/Teeth_Segmentation_Intraoral, MIT): foto ≤512 px + máscara PNG de dientes
  (Tooth ∪ Caries ∪ Cavity ∪ Crack = todo lo que es diente).
- Gingivitis (ekacare/IntraOral_Gingivitis_Image_Captioning = Mendeley 3253gj88rr, CC BY 4.0):
  foto ≤640 px + etiquetas (cajas por diente con grado, maxilar/mandíbula) + texto del dentista 1.
"""
import io, json, sys, tarfile, csv
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

raiz = Path(sys.argv[1]); sal = Path(sys.argv[2])

def reducir(img, lado):
    img.draft("RGB", (lado, lado))  # JPEG: decodifica ya reducida (mucha menos memoria)
    img = img.convert("RGB"); s = lado / max(img.size)
    return img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS) if s < 1 else img

# ── Dientes ──
d = sal / "dientes"; n = 0
with tarfile.open(raiz / "dientes/teeth.tar") as t:
    miembros = {m.name: m for m in t.getmembers()}
    for nombre, m in miembros.items():
        if "/ann/" not in nombre: continue
        split = nombre.split("/")[0]; base = Path(nombre).name[:-5]
        im_m = miembros.get(f"{split}/img/{base}")
        if im_m is None: continue
        stem0 = Path(base).stem
        if (d / split / f"{stem0}.png").exists(): n += 1; continue
        ann = json.load(t.extractfile(m))
        img = Image.open(t.extractfile(im_m)); W, H = img.size
        mask = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(mask)
        for o in ann["objects"]:
            pts = o.get("points", {}).get("exterior", [])
            if len(pts) >= 3: dr.polygon([tuple(p) for p in pts], fill=255)
        img = reducir(img, 512); mask = mask.resize(img.size, Image.NEAREST)
        (d / split).mkdir(parents=True, exist_ok=True)
        stem = Path(base).stem
        if (d / split / f"{stem}.png").exists(): n += 1; continue
        img.save(d / split / f"{stem}.jpg", quality=88); mask.save(d / split / f"{stem}.png")
        n += 1
print("dientes", n)

# ── Gingivitis ──
g = sal / "gingivitis"; g.mkdir(parents=True, exist_ok=True); filas = []
for pf in sorted((raiz / "gingivitis").glob("*.parquet")):
    split = pf.name.split("-")[0]
    for r in (x for lote in pq.ParquetFile(pf).iter_batches(batch_size=2) for x in lote.to_pylist()):
        img = r["Image"]; img = Image.open(io.BytesIO(img["bytes"] if isinstance(img, dict) else img))
        W, H = img.size; img = reducir(img, 640)
        nombre = f"{split}_{r['md5_image'][:12]}.jpg"; img.save(g / nombre, quality=90)
        filas.append({"archivo": nombre, "split": split, "ancho_orig": W, "alto_orig": H,
                      "labels": r["Labels"], "caption": r["Dentist_1_caption"]})
with open(g / "etiquetas.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
print("gingivitis", len(filas))
