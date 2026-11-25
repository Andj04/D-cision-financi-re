"""Étape 3 : régression logistique (baseline) + EBM, même liste de variables, calibration, métriques.

Protocole : tout est ajusté sur train ; le réglage (calibration, seuil F1, choix du modèle)
se fait sur validation ; le test n'est lu qu'une fois, après le choix, pour rapporter les métriques finales.
Usage : python src/models.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import style
from features import FEATURES, load_split, logreg_preprocessor

style.apply()

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def metrics(y, p, thr) -> dict:
    return {"AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p),
            "F1": f1_score(y, p >= thr), "Brier": brier_score_loss(y, p),
            "taux_defaut_reel": float(np.mean(y)), "pd_moyenne": float(np.mean(p)), "seuil_F1": thr}


def crossfit_isotonic(p, y, n_splits=5):
    """Probabilités calibrées hors-échantillon sur val (pour mesurer honnêtement la calibration)."""
    out = np.zeros_like(p)
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(p, y):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[tr], y[tr])
        out[te] = iso.predict(p[te])
    return out


LABELS = {"logreg": "Régression logistique (calibrée)", "ebm": "EBM (calibré)"}


def plot_calibration(yva, cal_val: dict, yte, cal_test: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, title, y, P in [(axes[0], "Validation (2013-2014)", yva, cal_val), (axes[1], "Test (2015-2016)", yte, cal_test)]:
        ax.plot([0, 0.3], [0, 0.3], ls="--", lw=1, color="#A9ADB0", label="calibration parfaite")
        for n in ["ebm", "logreg"]:
            fr, mp = calibration_curve(y, P[n], n_bins=10, strategy="quantile")
            ax.plot(mp, fr, marker="o", ms=4, lw=1.8, label=LABELS[n],
                    color=style.GREEN if n == "ebm" else style.GREY)
        ax.set_xlabel("probabilité prédite"); ax.set_ylabel("fréquence observée"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT / "reports/calibration.png", dpi=130)


def figures_only():
    """Redessine la figure de calibration à partir du modèle sauvegardé (sans réentraîner) : mêmes données,
    même calibration croisée (même graine), donc mêmes courbes."""
    art = joblib.load(ROOT / "models/risk_models.joblib")
    Xva, yva, _ = load_split("val")
    Xte, yte, _ = load_split("test")
    yva, yte = yva.values, yte.values
    cal_val = {n: crossfit_isotonic(art[n].predict_proba(Xva)[:, 1], yva) for n in ("logreg", "ebm")}
    cal_test = {n: art["calibrators"][n].predict(art[n].predict_proba(Xte)[:, 1]) for n in ("logreg", "ebm")}
    plot_calibration(yva, cal_val, yte, cal_test)


def main():
    Xtr, ytr, _ = load_split("train")
    Xva, yva, _ = load_split("val")
    ytr, yva = ytr.values, yva.values
    print(f"train {Xtr.shape} défaut {ytr.mean():.4f} | val {Xva.shape} défaut {yva.mean():.4f}", flush=True)

    # --- petit réglage des hyperparamètres : chaque candidat est ajusté sur train et jugé sur l'AUROC de validation
    #     (le test n'intervient pas). Grille volontairement courte (coût de l'EBM sur 720 000 lignes).
    grid, cands = [], {"logreg": [], "ebm": []}
    for C in [0.01, 0.1, 1.0, 10.0]:
        t = time.time()
        m = Pipeline([("prep", logreg_preprocessor()),
                      ("clf", LogisticRegression(max_iter=2000, C=C, random_state=SEED))]).fit(Xtr, ytr)
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        grid.append({"modele": "logreg", "hyperparametres": f"C={C:g}", "AUROC_val_brut": auc, "duree_s": time.time() - t})
        cands["logreg"].append((auc, m))
        print(grid[-1], flush=True)
    for cfg in [dict(interactions=3, learning_rate=0.05), dict(interactions=10, learning_rate=0.05)]:
        t = time.time()
        m = ExplainableBoostingClassifier(max_rounds=1000, outer_bags=4, random_state=SEED, n_jobs=-1, **cfg).fit(Xtr, ytr)
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        grid.append({"modele": "ebm", "hyperparametres": ", ".join(f"{k}={v}" for k, v in cfg.items()),
                     "AUROC_val_brut": auc, "duree_s": time.time() - t})
        cands["ebm"].append((auc, m))
        print(grid[-1], flush=True)
    g = pd.DataFrame(grid)
    g["retenu"] = g.groupby("modele")["AUROC_val_brut"].transform("max") == g["AUROC_val_brut"]
    g.to_csv(ROOT / "reports/reglage_hyperparametres.csv", index=False)
    logreg = max(cands["logreg"], key=lambda x: x[0])[1]
    ebm = max(cands["ebm"], key=lambda x: x[0])[1]

    raw = {"logreg": logreg.predict_proba(Xva)[:, 1], "ebm": ebm.predict_proba(Xva)[:, 1]}
    rows, cal_val, calibrators = [], {}, {}
    for name, p in raw.items():
        cal_val[name] = crossfit_isotonic(p, yva)                 # évaluation honnête
        calibrators[name] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, yva)
        for tag, pp in [("brut", p), ("calibre", cal_val[name])]:
            thr = best_f1_threshold(yva, pp)
            rows.append({"modele": name, "version": tag, **metrics(yva, pp, thr)})
    val = pd.DataFrame(rows).set_index(["modele", "version"])
    val.to_csv(ROOT / "reports/metriques_validation.csv")
    print(val.round(4).to_string(), flush=True)

    # --- choix du modèle sur VALIDATION uniquement : plus faible Brier calibré (la proba sert au MCDA)
    cal_rows = val.xs("calibre", level="version")
    chosen = cal_rows["Brier"].idxmin()
    print("modèle retenu (Brier calibré le plus bas sur validation):", chosen, flush=True)

    # --- test : lu une seule fois, après le choix
    Xte, yte, _ = load_split("test")
    yte = yte.values
    rows = []
    for name, model in [("logreg", logreg), ("ebm", ebm)]:
        p_raw = model.predict_proba(Xte)[:, 1]
        p_cal = calibrators[name].predict(p_raw)
        thr = best_f1_threshold(yva, cal_val[name])               # seuil fixé sur validation
        for tag, pp, th in [("brut", p_raw, best_f1_threshold(yva, raw[name])), ("calibre", p_cal, thr)]:
            rows.append({"modele": name, "version": tag, **metrics(yte, pp, th)})
    test = pd.DataFrame(rows).set_index(["modele", "version"])
    test.to_csv(ROOT / "reports/metriques_test.csv")
    print(test.round(4).to_string(), flush=True)

    # --- courbes de calibration (validation puis test)
    plot_calibration(yva, cal_val, yte, {n: calibrators[n].predict(m.predict_proba(Xte)[:, 1])
                                         for n, m in [("logreg", logreg), ("ebm", ebm)]})

    joblib.dump({"logreg": logreg, "ebm": ebm, "calibrators": calibrators, "chosen": chosen,
                 "features": FEATURES, "seuil_F1": {n: best_f1_threshold(yva, cal_val[n]) for n in cal_val}},
                ROOT / "models/risk_models.joblib", compress=3)
    (ROOT / "reports/modele_retenu.json").write_text(json.dumps(
        {"modele_retenu": chosen, "critere": "Brier calibré le plus bas sur validation",
         "taux_defaut_train": float(ytr.mean()), "taux_defaut_validation": float(yva.mean()),
         "remarque": "pd_ml est calibrée sur la validation : le taux de référence cohérent pour le filtre est celui "
                     "de la validation (le taux d'entraînement inclut la crise de 2008)."}, indent=2, ensure_ascii=False),
        encoding="utf-8")


if __name__ == "__main__":
    import sys
    figures_only() if "--figures" in sys.argv else main()
