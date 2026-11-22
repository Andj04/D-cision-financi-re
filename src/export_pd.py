"""Étape 5 : exporte pd_ml (probabilité de défaut calibrée du modèle retenu) pour chaque prêt du jeu de test."""
from pathlib import Path

import joblib
import pandas as pd

from features import load_split

ROOT = Path(__file__).resolve().parents[1]


def main():
    art = joblib.load(ROOT / "models/risk_models.joblib")
    name = art["chosen"]
    X, y, ids = load_split("test")
    p = art["calibrators"][name].predict(art[name].predict_proba(X)[:, 1])
    out = pd.DataFrame({"loan_id": ids.values, "pd_ml": p, "vraie_issue_default_60m": y.values})
    out.to_csv(ROOT / "data/processed/test_predictions.csv", index=False)
    print(f"modèle {name} : {len(out)} prêts, pd_ml moyenne {out.pd_ml.mean():.4f}, taux réel {out.vraie_issue_default_60m.mean():.4f}")


if __name__ == "__main__":
    main()
