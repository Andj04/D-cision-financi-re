"""Exécute le MCDA sur project_scores et écrit résultats + sensibilité (consommés par l'agent et l'appli).

Entrées modifiables par l'équipe :
  * params/ahp_jugements.csv : les 28 comparaisons deux à deux (échelle de Saaty, « 3 » ou « 1/3 »).
    S'il est absent, il est initialisé une fois à partir des poids cibles de la décision 13 (proposition à
    valider : tant que l'équipe ne l'a pas relu, l'AHP ne fait que vérifier la cohérence de ces poids).
  * profils ELECTRE TRI : mcda.PROFILE_SPEC (repères absolus quand il en existe).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import mcda

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed"
PARAMS = ROOT / "params"
JUDGMENTS = PARAMS / "ahp_jugements.csv"


def load_judgments() -> pd.DataFrame:
    if not JUDGMENTS.exists():
        mcda.judgments_table(mcda.ahp_matrix_from_targets(mcda.TARGET_WEIGHTS)).to_csv(JUDGMENTS, index=False)
        print("params/ahp_jugements.csv initialisé à partir des poids cibles (à faire valider par l'équipe)")
    return pd.read_csv(JUDGMENTS, dtype=str)


def main():
    scores = pd.read_csv(OUT / "project_scores.csv")
    info = json.loads((ROOT / "reports/modele_retenu.json").read_text(encoding="utf-8"))
    ref = info["taux_defaut_validation"]       # régime sur lequel pd_ml est calibrée (voir mcda.eliminatory_filter)

    m = mcda.ahp_matrix_from_judgments(load_judgments(), mcda.CRITERIA)
    w, lam, cr = mcda.ahp_weights(m)
    assert cr < 0.10, f"ratio de cohérence AHP {cr:.3f} >= 0,10 : revoir params/ahp_jugements.csv"
    m.to_csv(PARAMS / "ahp_matrice.csv")
    pd.DataFrame({"critere": w.index, "poids_ahp": w.values, "poids_cible": [mcda.TARGET_WEIGHTS[c] for c in w.index]}
                 ).to_csv(PARAMS / "poids_ahp.csv", index=False)
    (PARAMS / "ahp_coherence.json").write_text(json.dumps({"lambda_max": lam, "CR": cr}, indent=2))

    res = mcda.run_mcda(scores, w, ref)
    alive = res[~res["ecarte"]]
    mcda.make_profiles(alive, ref)[1].to_csv(PARAMS / "profils_electre_tri.csv", index=False)
    (PARAMS / "filtre_eliminatoire.json").write_text(json.dumps(
        {"dscr_min_plancher": 1.0, "taux_defaut_reference": ref, "multiple": 2.0, "seuil_pd": 2 * ref,
         "source_taux": "validation 2013-2014 (régime de calibration de pd_ml)",
         "taux_entrainement_non_utilise": info["taux_defaut_train"]}, indent=2, ensure_ascii=False), encoding="utf-8")

    sens = mcda.sensitivity(scores, w, ref)
    frag = sens["fragilite_poids"]
    res = res.join(frag)
    res["categorie_finale"] = res["categorie_tri_pessimiste"]
    lo, hi = mcda.LAMBDA_RANGE
    res["fragilite_texte"] = np.where(
        res["decision_fragile"],
        f"fragile (catégorie différente dans au moins 25 % des scénarios de poids, seuils et coupure {lo:g}-{hi:g})",
        f"stable (catégorie inchangée dans plus de 75 % des scénarios de poids, seuils et coupure {lo:g}-{hi:g})")
    # Incomparabilité ELECTRE III : les deux distillations placent le projet très différemment
    ecart = (res["rang_distillation_descendante"] - res["rang_distillation_ascendante"]).abs()
    res["ecart_distillations"] = ecart
    res["position_ambigue"] = ecart >= 0.2 * len(alive)
    names = pd.read_csv(OUT / "simulation_detail.csv").set_index("loan_id")["nom_projet"]
    res.insert(1, "nom_projet", res["loan_id"].map(names))
    res.to_csv(OUT / "mcda_resultats.csv", index=False)
    sens["seuils_et_coupure"].to_csv(OUT / "sensibilite_seuils_coupure.csv", index=False)
    sens["comparaison_methodes"].to_csv(OUT / "comparaison_methodes.csv", index=False)

    # Contrôle a posteriori, À TITRE D'ILLUSTRATION : la vraie issue n'est lue qu'ici, APRÈS les décisions,
    # et n'intervient dans aucun calcul (40 projets, ~4 % de défaut : quelques cas, aucune valeur de preuve).
    truth = pd.read_csv(OUT / "test_predictions.csv")[["loan_id", "vraie_issue_default_60m"]]
    post = res[["loan_id", "categorie_finale"]].merge(truth, on="loan_id")
    (post.groupby("categorie_finale")["vraie_issue_default_60m"].agg(projets="size", defauts_reels="sum")
         .reindex(["Financer", "Conditionner", "Refuser"]).reset_index()
         .to_csv(OUT / "controle_a_posteriori.csv", index=False))

    print(f"CR AHP = {cr:.4f}  | poids : {dict(w.round(3))}")
    print(f"taux de référence {ref:.4f} -> seuil pd {2 * ref:.4f}")
    print("écartés :", int(res.ecarte.sum()), res.motif_ecarte[res.ecarte].value_counts().to_dict())
    print("catégories :", res.categorie_finale.value_counts().to_dict())
    print(sens["comparaison_methodes"].round(3).to_string()); print(sens["seuils_et_coupure"].round(3).to_string())
    print("décisions fragiles :", int(frag.decision_fragile.sum()), "| positions ambiguës :", int(res.position_ambigue.sum()))
    print(pd.read_csv(OUT / "controle_a_posteriori.csv").to_string(index=False))


if __name__ == "__main__":
    main()
