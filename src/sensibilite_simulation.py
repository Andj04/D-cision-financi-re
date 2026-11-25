"""Sensibilité des décisions à la SIMULATION : mêmes 40 prêts, mêmes pd_ml, mêmes poids et paramètres MCDA,
mais les indicateurs financiers (VAN, DSCR, garanties, ESG...) sont retirés avec 20 graines différentes.

Question posée : les catégories viennent-elles des projets ou du hasard de la simulation ?
Usage : python src/sensibilite_simulation.py   (après simulate.py, models.py et run_mcda.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import mcda
from project_scores import COLUMNS
from simulate import SEED, simulate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed"
N_SEEDS = 20


def main():
    pred = pd.read_csv(OUT / "test_predictions.csv").set_index("loan_id")["pd_ml"]
    test = pd.read_parquet(OUT / "test.parquet")
    ref = json.loads((ROOT / "reports/modele_retenu.json").read_text(encoding="utf-8"))["taux_defaut_validation"]
    w = pd.read_csv(ROOT / "params/poids_ahp.csv").set_index("critere")["poids_ahp"]
    base = pd.read_csv(OUT / "mcda_resultats.csv").set_index("loan_id")["categorie_finale"]

    cats, summary = {}, []
    for s in range(1, N_SEEDS + 1):
        sim = simulate(test, pred, seed=SEED, indicator_seed=SEED + s)
        res = mcda.run_mcda(sim[["loan_id"] + COLUMNS], w, ref).set_index("loan_id")
        cats[s] = res["categorie_tri_pessimiste"]
        summary.append({"graine_indicateurs": SEED + s, "ecartes": int(res["ecarte"].sum()),
                        **res["categorie_tri_pessimiste"].value_counts().reindex(mcda.CATEGORIES, fill_value=0).to_dict()})
    cats = pd.DataFrame(cats).reindex(base.index)
    per_project = pd.DataFrame({
        "categorie_base": base,
        "part_graines_meme_categorie": cats.eq(base, axis=0).mean(axis=1),
        **{f"part_{c}": cats.eq(c).mean(axis=1) for c in mcda.CATEGORIES},
    }).reset_index()
    per_project.to_csv(OUT / "sensibilite_graine_projets.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "sensibilite_graine_resume.csv", index=False)
    print(pd.DataFrame(summary).describe().round(2).to_string())
    print("part moyenne de graines donnant la même catégorie :", round(per_project.part_graines_meme_categorie.mean(), 3))
    print(per_project.groupby("categorie_base").part_graines_meme_categorie.mean().round(3).to_string())


if __name__ == "__main__":
    main()
