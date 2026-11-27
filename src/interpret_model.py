"""Étape 4 : interprétabilité du modèle retenu (EBM) : explications natives, permutation, erreurs, explications locales.

Règle « test lu une seule fois » : la permutation et l'analyse d'erreurs se font sur la VALIDATION (2013-2014).
Le test ne sert ici qu'aux explications locales des 40 projets, qui sont des prêts du test (aucune métrique).
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import style
from features import load_split

style.apply()

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
rng = np.random.default_rng(0)


VAR_FR = {"duree_classe": "durée du prêt", "etat": "État", "age_entreprise": "âge de l'entreprise",
          "taux_initial": "taux d'intérêt initial", "procedure": "procédure SBA", "type_entreprise": "type d'entreprise",
          "ligne_renouvelable": "ligne renouvelable", "secteur_naics2": "secteur NAICS", "franchise": "franchise",
          "taux_fixe_variable": "taux fixe ou variable", "part_garantie": "part garantie par la SBA",
          "taux_manquant": "taux non renseigné", "log_montant": "montant"}


def fr_term(term: str) -> str:
    """« log_montant & procedure » -> « montant × procédure SBA » (libellés des figures)."""
    return " × ".join(VAR_FR.get(t, t) for t in str(term).split(" & "))


def readable(var: str, x) -> str:
    """« log_montant = 11.73 » -> « montant ≈ 125 000 $ » ; noms de variables en français."""
    if var == "log_montant":
        return f"montant ≈ {np.expm1(float(x)):,.0f} $".replace(",", " ")
    if var == "part_garantie":
        return f"part garantie par la SBA = {100 * float(x):.0f} %"
    if var == "taux_initial":
        return f"taux d'intérêt initial = {float(x):.2f} %"
    return f"{VAR_FR.get(var, var)} = {x}"


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


def shap_importance(lgbm, Xva, n=5000):
    """Valeurs SHAP exactes pour les arbres (TreeExplainer) : |contribution| moyenne en log-odds par variable,
    comparée à l'importance native de l'EBM. Écrit reports/shap_lightgbm.png et shap_lightgbm.csv."""
    import shap
    sample = Xva.sample(min(n, len(Xva)), random_state=0)
    sv = shap.TreeExplainer(lgbm.model).shap_values(lgbm.frame(sample))
    sv = sv[1] if isinstance(sv, list) else sv
    imp = (pd.DataFrame({"variable": sample.columns, "shap_moyen_abs": np.abs(sv).mean(axis=0)})
           .sort_values("shap_moyen_abs", ascending=False))
    imp.to_csv(REP / "shap_lightgbm.csv", index=False)
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(top.variable.map(fr_term), top.shap_moyen_abs, color="#D39B2A")
    ax.set_xlabel("|valeur SHAP| moyenne (log-odds)")
    ax.set_title("LightGBM : contribution moyenne de chaque variable (SHAP, validation)")
    fig.tight_layout(); fig.savefig(REP / "shap_lightgbm.png", dpi=130); plt.close(fig)
    print(imp.round(4).to_string(index=False))


def main():
    art = joblib.load(ROOT / "models/risk_models.joblib")
    ebm, iso = art["ebm"], art["calibrators"]["ebm"]
    Xva, yva, _ = load_split("val")
    pred = lambda X: ebm.predict_proba(X)[:, 1]  # noqa: E731

    # 1. importances natives de l'EBM (contribution absolue moyenne par terme, échelle log-odds)
    glob = pd.DataFrame({"terme": ebm.term_names_, "importance_native": [float(np.mean(np.abs(s))) for s in ebm.term_scores_]}
                        ).sort_values("importance_native", ascending=False)
    glob.to_csv(REP / "ebm_importance_native.csv", index=False)

    # 2. permutation sur la validation (modèle retenu)
    perm, base = permutation_importance(pred, Xva, yva.values)
    perm.to_csv(REP / "permutation_importance_validation.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    g = glob.head(12)[::-1]; ax[0].barh(g.terme.map(fr_term), g.importance_native, color=style.GREY); ax[0].set_title("EBM : importance native (|log-odds| moyen)")
    p = perm.head(12)[::-1]; ax[1].barh(p.variable.map(fr_term), p.baisse_AUROC_moyenne, xerr=p.ecart_type, color=style.GREEN, ecolor=style.DARK)
    ax[1].set_title(f"Permutation sur la validation (AUROC de base {base:.3f})"); fig.tight_layout()
    fig.savefig(REP / "importance_variables.png", dpi=130)

    # 3. courbes de forme de l'EBM pour les 4 variables les plus importantes (effet de chaque variable)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax_, term in zip(axes.ravel(), [t for t in glob.terme if " & " not in t][:4]):
        d = ebm.explain_global().data(ebm.term_names_.index(term))
        names, vals = list(d["names"]), np.asarray(d["scores"], float)
        if len(names) == len(vals) + 1:                               # variable continue : bornes de bins
            ax_.step([float(x) for x in names[:-1]], vals, where="post", color=style.GREEN, lw=1.8)
        else:                                                         # catégorielle : 10 plus fortes modalités
            top = np.argsort(-np.abs(vals))[:10]
            v = vals[top][::-1]
            ax_.barh([str(names[j]) for j in top][::-1], v,
                     color=["#C0503F" if x > 0 else style.GREEN for x in v])   # rouge = augmente le risque
        ax_.set_title(f"Effet : {VAR_FR.get(term, term)}"); ax_.set_ylabel("log-odds")
    fig.tight_layout(); fig.savefig(REP / "ebm_effets.png", dpi=130)

    # 4. analyse des erreurs sur la validation (secteur, État, procédure) : défauts réels vs prédits, seuil F1
    p_cal = iso.predict(pred(Xva)); thr = art["seuil_F1"]["ebm"]
    df = pd.DataFrame({"secteur": Xva.secteur_naics2.values, "etat": Xva.etat.values, "procedure": Xva.procedure.values,
                       "y": yva.values, "pd": p_cal, "pred": (p_cal >= thr).astype(int)})
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
    err.to_csv(REP / "erreurs_par_segment_validation.csv", index=False)

    # 4 bis. SHAP pour LightGBM (référence « boîte noire ») : contribution moyenne de chaque variable, sur la validation
    if "lgbm" in art:
        shap_importance(art["lgbm"], Xva)

    # 5. explications locales des 40 projets (prêts du test) : 3 facteurs qui augmentent le plus le risque
    Xte, _, ids = load_split("test")
    scores = pd.read_csv(ROOT / "data/processed/project_scores.csv")
    sel = Xte[ids.isin(scores.loan_id).values]; sel_ids = ids[ids.isin(scores.loan_id).values].values
    contrib = ebm.eval_terms(sel)
    names = np.array(ebm.term_names_); rows = []
    for k, lid in enumerate(sel_ids):
        order = np.argsort(-contrib[k])[:3]
        txt = "; ".join(" & ".join(readable(v, sel.iloc[k][v]) for v in names[j].split(" & ")) for j in order if contrib[k][j] > 0)
        rows.append({"loan_id": lid, "facteurs_risque": txt or "aucun facteur augmentant nettement le risque",
                     **{f"contrib_{names[j]}": float(contrib[k][j]) for j in np.argsort(-np.abs(contrib[k]))[:5]}})
    pd.DataFrame(rows).to_csv(ROOT / "data/processed/explications_projets.csv", index=False)
    print(glob.head(8).round(3).to_string()); print(perm.head(8).round(4).to_string()); print(err.head(12).round(3).to_string())


if __name__ == "__main__":
    main()
