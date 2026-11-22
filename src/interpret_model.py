"""Étape 4 : interprétabilité du modèle retenu (EBM) : explications natives, permutation, erreurs, explications locales."""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import load_split

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
rng = np.random.default_rng(0)


def permutation_importance(predict, X, y, n_repeats=5, max_rows=30000):
    idx = rng.choice(len(X), size=min(max_rows, len(X)), replace=False)
    X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    base = roc_auc_score(y, predict(X))
    rows = []
    for c in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy(); Xp[c] = Xp[c].sample(frac=1, random_state=int(rng.integers(1e9))).values
            drops.append(base - roc_auc_score(y, predict(Xp)))
        rows.append({"variable": c, "baisse_AUROC_moyenne": np.mean(drops), "ecart_type": np.std(drops)})
    return pd.DataFrame(rows).sort_values("baisse_AUROC_moyenne", ascending=False), base


def main():
    art = joblib.load(ROOT / "models/risk_models.joblib")
    ebm, iso = art["ebm"], art["calibrators"]["ebm"]
    Xte, yte, ids = load_split("test")
    pred = lambda X: ebm.predict_proba(X)[:, 1]  # noqa: E731

    # 1. importances natives de l'EBM (contribution absolue moyenne par terme, échelle log-odds)
    glob = pd.DataFrame({"terme": ebm.term_names_, "importance_native": [float(np.mean(np.abs(s))) for s in ebm.term_scores_]}
                        ).sort_values("importance_native", ascending=False)
    glob.to_csv(REP / "ebm_importance_native.csv", index=False)

    # 2. permutation sur le jeu de test (modèle retenu)
    perm, base = permutation_importance(pred, Xte, yte.values)
    perm.to_csv(REP / "permutation_importance_test.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    g = glob.head(12)[::-1]; ax[0].barh(g.terme, g.importance_native); ax[0].set_title("EBM : importance native (|log-odds| moyen)")
    p = perm.head(12)[::-1]; ax[1].barh(p.variable, p.baisse_AUROC_moyenne, xerr=p.ecart_type)
    ax[1].set_title(f"Permutation sur le test (AUROC de base {base:.3f})"); fig.tight_layout()
    fig.savefig(REP / "importance_variables.png", dpi=130)

    # 3. courbes de forme de l'EBM pour les 4 variables les plus importantes (effet de chaque variable)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax_, term in zip(axes.ravel(), [t for t in glob.terme if " & " not in t][:4]):
        d = ebm.explain_global().data(ebm.term_names_.index(term))
        names, vals = list(d["names"]), np.asarray(d["scores"], float)
        if len(names) == len(vals) + 1:                               # variable continue : bornes de bins
            ax_.step([float(x) for x in names[:-1]], vals, where="post")
        else:                                                         # catégorielle : 10 plus fortes modalités
            top = np.argsort(-np.abs(vals))[:10]
            ax_.barh([str(names[j]) for j in top][::-1], vals[top][::-1])
        ax_.set_title(f"Effet : {term}"); ax_.set_ylabel("log-odds")
    fig.tight_layout(); fig.savefig(REP / "ebm_effets.png", dpi=130)

    # 4. analyse des erreurs sur le test (secteur, État, procédure) : défauts réels vs prédits, seuil F1 du modèle
    p_cal = iso.predict(pred(Xte)); thr = art["seuil_F1"]["ebm"]
    df = pd.DataFrame({"secteur": Xte.secteur_naics2.values, "etat": Xte.etat.values, "procedure": Xte.procedure.values,
                       "y": yte.values, "pd": p_cal, "pred": (p_cal >= thr).astype(int)})
    out = []
    for dim in ["secteur", "etat", "procedure"]:
        g = df.groupby(dim).agg(n=("y", "size"), defauts_reels=("y", "sum"), pd_moyenne=("pd", "mean"),
                                fn=("y", lambda s: int(((s == 1) & (df.loc[s.index, "pred"] == 0)).sum())),
                                fp=("y", lambda s: int(((s == 0) & (df.loc[s.index, "pred"] == 1)).sum()))).reset_index()
        g["taux_reel"] = g.defauts_reels / g.n
        g["ecart_calibration"] = g.pd_moyenne - g.taux_reel
        g["taux_faux_negatifs"] = g.fn / g.defauts_reels.replace(0, np.nan)
        g.insert(0, "dimension", dim); g = g.rename(columns={dim: "modalite"}); out.append(g[g.n >= 300])
    err = pd.concat(out).sort_values(["dimension", "n"], ascending=[True, False])
    err.to_csv(REP / "erreurs_par_segment_test.csv", index=False)

    # 5. explications locales pour chaque prêt du test pd_ml déjà exporté : 3 facteurs qui augmentent le plus le risque
    scores = pd.read_csv(ROOT / "data/processed/project_scores.csv")
    sel = Xte[ids.isin(scores.loan_id).values]; sel_ids = ids[ids.isin(scores.loan_id).values].values
    contrib = ebm.eval_terms(sel)
    names = np.array(ebm.term_names_); rows = []
    for k, lid in enumerate(sel_ids):
        order = np.argsort(-contrib[k])[:3]
        txt = "; ".join(" & ".join(f"{v} = {sel.iloc[k][v]}" for v in names[j].split(" & ")) for j in order if contrib[k][j] > 0)
        rows.append({"loan_id": lid, "facteurs_risque": txt or "aucun facteur augmentant nettement le risque",
                     **{f"contrib_{names[j]}": float(contrib[k][j]) for j in np.argsort(-np.abs(contrib[k]))[:5]}})
    pd.DataFrame(rows).to_csv(ROOT / "data/processed/explications_projets.csv", index=False)
    print(glob.head(8).round(3).to_string()); print(perm.head(8).round(4).to_string()); print(err.head(12).round(3).to_string())


if __name__ == "__main__":
    main()
