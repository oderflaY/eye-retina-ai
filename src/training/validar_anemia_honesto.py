"""Validación honesta del modelo de anemia (conjuntiva) sin la confusión por tono de piel / país.

    python -m src.training.validar_anemia_honesto

Eyes-defy-anemia mezcla pacientes de India (Chennai) y de Italia (Bari) y la copia que usamos no
dice de dónde es cada foto. En los datos, la piel más oscura tiene mucha más anemia (65 % vs 22 %
entre el cuartil más oscuro y el más claro), así que una AUC "global" premia al modelo por
adivinar el país/tono y no la palidez. Aquí se mide la AUC comparando SOLO personas del mismo tono
de piel (cuartiles de L* de la piel alrededor del ojo, de ``experimento_anemia_piel``), y la
sensibilidad/especificidad con la zona gris ponderando para que cada tono pese con prevalencia 50/50.

No cambia el modelo (reentrenarlo con pesos por tono no mejoró: 0.773 vs 0.775 dentro del tono);
cambia los números que la app muestra, para que no prometa más de lo que es.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import PROJECT_ROOT
from src.deployment.inference_anemia import CARACTERISTICAS

REP = 20


def main() -> None:
    piel = pd.read_csv(PROJECT_ROOT / "reports/anemia_piel/medidas.csv").dropna(subset=["piel_L"])
    d = piel.reset_index(drop=True)
    d["tono"] = pd.qcut(d.piel_L, 4, labels=False)
    y = d.anemia.to_numpy()
    X = d[list(CARACTERISTICAS)].to_numpy()
    ruta = PROJECT_ROOT / "models/anemia/clasificador.joblib"
    paq = joblib.load(ruta)
    bajo, alto = paq["zona_gris"]

    # Peso por (tono, clase): cada tono queda 50/50 → la prevalencia por tono ya no ayuda.
    w = np.ones(len(y))
    for t, g in d.groupby("tono"):
        for c in (0, 1):
            m = ((d.tono == t) & (d.anemia == c)).to_numpy()
            w[m] = 0.5 * len(g) / max(m.sum(), 1)

    dentro, oofs = [], []
    for r in range(REP):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, d.tono * 2 + y):
            p[te] = clone(paq["modelo"]).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oofs.append(p)
        a, n = zip(*[(roc_auc_score(g.anemia, p[g.index]), len(g)) for _, g in d.groupby("tono")])
        dentro.append(np.average(a, weights=n))
    p = np.mean(oofs, axis=0)
    pos, neg = p >= alto, p < bajo
    dado = pos | neg
    wd = lambda m: float(np.sum(w[m]))  # noqa: E731
    sens = wd(pos & (y == 1) & dado) / wd((y == 1) & dado)
    esp = wd(neg & (y == 0) & dado) / wd((y == 0) & dado)

    v = paq["validacion"]
    v.setdefault("auc_cv_global", v["auc_cv"])
    v.update({
        "auc_cv": round(float(np.mean(dentro)), 4), "auc_sd": round(float(np.std(dentro)), 4),
        "auc_mismo_tono": round(float(np.mean(dentro)), 4),
        "sensibilidad": round(sens, 4), "especificidad": round(esp, 4),
        "exactitud": round((sens + esp) / 2, 4), "cobertura": round(float(dado.mean()), 4),
        "metodo": "AUC comparando solo personas del mismo tono de piel (cuartiles), 5×20; "
                  "sens/esp ponderadas 50/50 por tono",
        "nota_confusion": "El dataset mezcla India e Italia; la AUC global (0.83) estaba inflada por el tono de piel.",
    })
    for k in ("vpp", "vpn"):
        v.pop(k, None)  # dependen de la prevalencia del dataset; no aplican a la población real
    paq["validacion"] = v
    joblib.dump(paq, ruta)
    (PROJECT_ROOT / "models/anemia/config.json").write_text(json.dumps(v, indent=2, ensure_ascii=False))
    print(json.dumps(v, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
