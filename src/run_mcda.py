"""Exécute le MCDA sur project_scores et écrit résultats + sensibilité (consommés par l'agent et l'appli)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import mcda

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed"


def main():
    scores = pd.read_csv(OUT / "project_scores.csv")
    rate = json.loads((ROOT / "reports/modele_retenu.json").read_text(encoding="utf-8"))["taux_defaut_train"]
    m = mcda.ahp_matrix_from_targets(mcda.TARGET_WEIGHTS)
    w, lam, cr = mcda.ahp_weights(m)
    assert cr < 0.10, f"ratio de cohérence AHP {cr:.3f} >= 0,10"
    m.to_csv(ROOT / "params/ahp_matrice.csv")
    pd.DataFrame({"critere": w.index, "poids_ahp": w.values, "poids_cible": [mcda.TARGET_WEIGHTS[c] for c in w.index]}
                 ).to_csv(ROOT / "params/poids_ahp.csv", index=False)
    (ROOT / "params/ahp_coherence.json").write_text(json.dumps({"lambda_max": lam, "CR": cr}, indent=2))

    res = mcda.run_mcda(scores, w, rate)
    sens = mcda.sensitivity(scores, w, rate)
    frag = sens["fragilite_poids"]
    res = res.join(frag)
    res["categorie_finale"] = res["categorie_tri_pessimiste"]
    res["fragilite_texte"] = np.where(res["decision_fragile"], "fragile (catégorie différente dans au moins 25 % des scénarios de poids, seuils et coupure)",
                                      "stable (catégorie inchangée dans plus de 75 % des scénarios de poids, seuils et coupure)")
    res.to_csv(OUT / "mcda_resultats.csv", index=False)
    sens["seuils_et_coupure"].to_csv(OUT / "sensibilite_seuils_coupure.csv", index=False)
    sens["comparaison_methodes"].to_csv(OUT / "comparaison_methodes.csv", index=False)
    print(f"CR AHP = {cr:.4f}  | poids : {dict(w.round(3))}")
    print("écartés :", int(res.ecarte.sum()), "| catégories :", res.categorie_finale.value_counts().to_dict())
    print(sens["comparaison_methodes"].round(3).to_string()); print(sens["seuils_et_coupure"].round(3).to_string())
    print("décisions fragiles :", int(frag.decision_fragile.sum()))


if __name__ == "__main__":
    main()
