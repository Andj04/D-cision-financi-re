"""Étape 7 : assemble project_scores (40 lignes, 8 critères aux noms fixés par la décision 2)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["pd_ml", "van", "delai_recuperation", "dscr_min", "dette_fonds_propres",
           "couverture_garanties", "esg_score", "emplois_par_million"]


def main():
    sim = pd.read_csv(ROOT / "data/processed/simulation_detail.csv")
    scores = sim[["loan_id"] + COLUMNS]
    assert scores.shape == (40, 9) and not scores.isna().any().any(), "40 lignes x 8 critères, sans valeur manquante"
    assert scores["loan_id"].is_unique
    scores.to_csv(ROOT / "data/processed/project_scores.csv", index=False)
    print(scores.describe().T.round(3))


if __name__ == "__main__":
    main()
