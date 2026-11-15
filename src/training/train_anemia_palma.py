"""Entrena el modelo de anemia por la palma de la mano (datos de Ghana, CC BY 4.0).

    python -m src.training.train_anemia_palma

Una imagen por niño (las 12 copias aumentadas son idénticas en color), medidas de color de
``experimento_anemia_ghana.medir`` y bosque aleatorio. La validación que se guarda es la honesta:
la AUC con el color "limpio" del efecto del recorte (tamaño, forma, nitidez), que es la cifra que
no depende de cómo se tomaron las fotos. La zona gris se elige como en el modelo del ojo.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import PROJECT_ROOT
from src.training.experimento_anemia_ghana import COLOR, MOLESTIA, residualizar, tabla

ZONA_GRIS = (0.35, 0.65)


def oof(modelo, X, y, rep=10):
    p, n, aucs = np.zeros(len(y)), np.zeros(len(y)), []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=rep, random_state=42).split(X, y):
        q = clone(modelo).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], q))
        p[te] += q
        n[te] += 1
    return p / n, float(np.mean(aucs)), float(np.std(aucs))


def main() -> None:
    d = tabla()
    d = d[d.conjunto == "palma"].reset_index(drop=True)
    y = d.anemia.to_numpy()
    X = d[COLOR].to_numpy()
    modelo = RandomForestClassifier(400, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1)
    p, auc_color, _ = oof(modelo, X, y)
    _, auc_limpia, sd_limpia = oof(modelo, residualizar(d, COLOR, MOLESTIA), y)
    bajo, alto = ZONA_GRIS
    pos, neg = p >= alto, p < bajo
    dado = pos | neg
    val = {
        "auc_cv": round(auc_limpia, 4), "auc_sd": round(sd_limpia, 4), "auc_color_crudo": round(auc_color, 4),
        "zona_gris": [bajo, alto], "cobertura": round(float(dado.mean()), 4),
        "sensibilidad": round(float(pos[(y == 1) & dado].mean()), 4),
        "especificidad": round(float(neg[(y == 0) & dado].mean()), 4),
        "n": int(len(y)), "n_anemia": int(y.sum()), "modelo": "bosque",
        "datos": "Ghana palm images (Mendeley ccr8cm22vz, CC BY 4.0), niños de 6–59 meses",
        "metodo": "una imagen por niño, validación cruzada 5×10; AUC con el efecto del recorte removido",
    }
    final = clone(modelo).fit(X, y)
    sal = PROJECT_ROOT / "models/palma"
    sal.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": final, "caracteristicas": COLOR, "zona_gris": list(ZONA_GRIS), "validacion": val},
                sal / "clasificador.joblib")
    (sal / "config.json").write_text(json.dumps(val, indent=2, ensure_ascii=False))
    print(json.dumps(val, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
