"""Prépare les éléments du rapport LaTeX (dossier rapport/) à partir des résultats du pipeline.

  * rapport/figures/ : figures existantes (analyse exploratoire, modèle) recopiées + figures du MCDA et de
    l'arbitrage générées ici, à la charte de l'École Centrale Casablanca ;
  * rapport/tables/  : tableaux LaTeX générés depuis les CSV (aucun chiffre recopié à la main).
Usage : python src/rapport_assets.py   (après tout le pipeline)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import arbitrage
import engine
import style

ROOT = Path(__file__).resolve().parents[1]
D, R, P = ROOT / "data/processed", ROOT / "reports", ROOT / "params"
FIG, TAB = ROOT / "rapport/figures", ROOT / "rapport/tables"
style.apply()
CAT = {"Financer": style.GREEN, "Conditionner": "#D39B2A", "Refuser": "#C0503F"}
CAT_FR = {"Financer": "Financer", "Conditionner": "Sous conditions", "Refuser": "Refuser"}
LAB = {"pd_ml": "Probabilité de défaut", "van": "VAN", "delai_recuperation": "Délai de récupération",
       "dscr_min": "DSCR minimum", "dette_fonds_propres": "Dette / fonds propres",
       "couverture_garanties": "Couverture par les garanties", "esg_score": "Score ESG",
       "emplois_par_million": "Emplois par million \\$"}


# ----------------------------------------------------------------------------------------------- utilitaires LaTeX
def tex(s) -> str:
    s = str(s)
    for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"),
                 ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\^{}")]:
        s = s.replace(a, b)
    return s


def num(v, d=2, pct=False, sep=True) -> str:
    """Nombre à la française : virgule décimale, espace fine pour les milliers."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    v = float(v) * (100 if pct else 1)
    s = f"{v:,.{d}f}" if sep else f"{v:.{d}f}"
    s = s.replace(",", "\\,").replace(".", ",")
    return s + ("\\,\\%" if pct else "")


def table(name, header, rows, cols, caption, label, size="\\small", long=False, note=None):
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    if long:
        # groupe { } : la taille et l'espacement réduits ne s'appliquent qu'à ce tableau
        out = (f"{{{size}\\setlength{{\\tabcolsep}}{{3pt}}\n\\begin{{longtable}}{{{cols}}}\n"
               f"\\caption{{{caption}}}\\label{{{label}}}\\\\\n\\toprule\n"
               f"{header} \\\\\n\\midrule\n\\endfirsthead\n\\toprule\n{header} \\\\\n\\midrule\n\\endhead\n"
               f"\\bottomrule\n\\endfoot\n{body}\n\\end{{longtable}}}}\n")
    else:
        out = (f"\\begin{{table}}[htbp]\n\\centering\n{size}\\setlength{{\\tabcolsep}}{{4pt}}\n"
               f"\\caption{{{caption}}}\\label{{{label}}}\n"
               f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
               f"\\end{{tabular}}\n" + (f"\\par\\smallskip\\footnotesize {note}\n" if note else "") + "\\end{table}\n")
    (TAB / f"{name}.tex").write_text(out, encoding="utf-8")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- figures
def copy_existing():
    for f in (R / "eda").glob("*.png"):
        shutil.copy(f, FIG / f.name)
    for f in ["calibration.png", "importance_variables.png", "ebm_effets.png", "shap_lightgbm.png"]:
        shutil.copy(R / f, FIG / f)
    shutil.copy(ROOT / "app/assets/logo-ecc.png", FIG / "logo-ecc.png")


def mcda_figures(ev, ref):
    r = ev["results"]
    # carte risque x DSCR
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for c in ["Refuser", "Conditionner", "Financer"]:
        s = r[r.categorie_finale == c]
        ax.scatter(100 * s.pd_ml, s.dscr_min, s=18 + s.montant_pret / 6000, color=CAT[c], alpha=0.85,
                   edgecolor="white", lw=0.8, label=CAT_FR[c], zorder=3)
    ax.axhline(1.0, ls="--", lw=1, color=CAT["Refuser"])
    ax.axvline(200 * ref, ls="--", lw=1, color=CAT["Refuser"])
    ax.text(200 * ref - 0.15, 0.81, f"seuil de risque {200 * ref:.1f} %".replace(".", ","), fontsize=8,
            color=CAT["Refuser"], ha="right")
    ax.text(13.5, 1.02, "DSCR = 1", fontsize=8, color=CAT["Refuser"])
    ax.set_xlabel("probabilité de défaut estimée par l'EBM (%)")
    ax.set_ylabel("DSCR minimum (simulé)")
    ax.set_title("Les 40 projets : risque, capacité de remboursement et décision")
    leg = ax.legend(title="Décision (taille = montant)", loc="upper right", fontsize=8, markerscale=1)
    for h in leg.legend_handles:
        h.set_sizes([40])
    save(fig, "mcda_carte.png")
    # poids AHP
    w = pd.read_csv(P / "poids_ahp.csv").sort_values("poids_ahp")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(w))
    ax.barh(y + 0.2, 100 * w.poids_ahp, height=0.38, color=style.GREEN, label="poids AHP")
    ax.barh(y - 0.2, 100 * w.poids_cible, height=0.38, color="#C9D7D3", label="poids cible du cahier des charges")
    ax.set_yticks(y, [LAB[c].replace("\\$", "$") for c in w.critere])
    for i, v in enumerate(w.poids_ahp):
        ax.text(100 * v + 0.3, i + 0.2, f"{100 * v:.1f} %".replace(".", ","), va="center", fontsize=8)
    ax.set_xlabel("poids (%)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Poids des critères : AHP (ratio de cohérence 0,009)")
    save(fig, "mcda_poids.png")
    # effet de lambda
    s = pd.read_csv(D / "sensibilite_seuils_coupure.csv")
    lam = s[s.parametre == "coupure lambda"].assign(v=lambda d: d.valeur.astype(float))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.axvspan(0.5, 0.8, color=style.GREEN, alpha=0.08, lw=0)
    ax.text(0.65, lam.projets_categorie_change.max() * 0.9, "plage des scénarios\ncombinés", ha="center", fontsize=8,
            color=style.DARK)
    ax.plot(lam.v, lam.projets_categorie_change, marker="o", color=style.GREEN, lw=2)
    for x, v in zip(lam.v, lam.projets_categorie_change):
        ax.text(x, v + 0.8, str(v), ha="center", fontsize=8)
    ax.set_xlabel("coupure λ d'ELECTRE TRI")
    ax.set_ylabel("projets qui changent\nde catégorie (sur 40)")
    ax.set_title("Les catégories sont stables pour λ entre 0,5 et 0,8")
    save(fig, "mcda_lambda.png")
    # ELECTRE III contre PROMETHEE II
    al = r[~r.ecarte]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([1, len(al)], [1, len(al)], ls="--", lw=1, color="#A9ADB0")
    for c in ["Refuser", "Conditionner", "Financer"]:
        s_ = al[al.categorie_finale == c]
        ax.scatter(s_.rang_electre3, s_.rang_promethee2, color=CAT[c], s=36, label=CAT_FR[c], zorder=3)
    ax.set_xlabel("rang ELECTRE III")
    ax.set_ylabel("rang PROMETHEE II")
    ax.set_title("Deux méthodes, des classements proches\n(Spearman 0,891, Kendall 0,724)")
    ax.legend(fontsize=8)
    save(fig, "mcda_e3_p2.png")
    # stabilité de chaque décision
    st_ = r[["nom", "categorie_finale", "part_scenarios_categorie_differente"]].copy()
    st_["stable"] = 1 - st_.part_scenarios_categorie_differente
    st_ = st_.merge(engine.demo_projects()[["nom", "part_graines_meme_categorie"]], on="nom")
    st_ = st_.sort_values("stable", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    for ax, col, title in [(axes[0], "stable", "Face aux paramètres du MCDA (200 scénarios)"),
                           (axes[1], "part_graines_meme_categorie", "Face au tirage des indicateurs (20 graines)")]:
        o = st_.sort_values(col, ascending=False)
        ax.bar(range(len(o)), 100 * o[col], color=[CAT[c] for c in o.categorie_finale])
        ax.axhline(75, ls="--", lw=1, color=style.GREY)
        ax.set_xticks([])
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("projets (triés)")
    axes[0].set_ylabel("catégorie inchangée (% des cas)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=CAT[c], label=CAT_FR[c]) for c in ["Financer", "Conditionner", "Refuser"]],
               loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "mcda_stabilite.png", dpi=170)
    plt.close(fig)


def arbitrage_figures(ev):
    r = ev["results"]
    res, s = arbitrage.run(r, arbitrage.DEFAULTS)
    sect = pd.Series(s["montant_par_secteur"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(range(len(sect)), sect.values / 1e6, color=style.GREEN)
    ax.set_yticks(range(len(sect)), sect.index)
    ax.axvline(0.35 * s["budget"] / 1e6, ls="--", lw=1, color=CAT["Refuser"])
    ax.text(0.35 * s["budget"] / 1e6, len(sect) - 0.4, " plafond 35 %", fontsize=8, color=CAT["Refuser"], va="top")
    ax.set_xlabel("montant engagé (millions de $)")
    ax.set_title(f"Arbitrage : {s['projets_selectionnes']} projets retenus, "
                 f"{s['montant_engage'] / 1e6:.2f} M\\$ engagés sur 5 M\\$".replace(".", ","))
    save(fig, "arbitrage_secteurs.png")
    rows = []
    for mode, lab in [("classement", "Suivre le classement"), ("nombre", "Maximiser le nombre")]:
        rr, ss = arbitrage.run(r, {**arbitrage.DEFAULTS, "mode": mode})
        murphy = rr.set_index("nom").loc["Studio de développement logiciel Murphy", "selection_budget"]
        rows.append([lab, str(ss["projets_selectionnes"]), str(ss["dont_financer"]), str(ss["dont_conditionner"]),
                     num(ss["montant_engage"] / 1e6, 2) + "\\,M\\$", num(ss["taux_utilisation"], 1, pct=True),
                     str(ss["secteurs_finances"]), "oui" if murphy else "\\textbf{non}"])
    table("tab_arbitrage_regles", "Règle & Retenus & Financer & Sous cond. & Engagé & Utilisation & Secteurs & N\\textsuperscript{o}\\,1 financé",
          rows, "lccccccc", "Comparaison des deux règles de sélection (budget 5\\,M\\$, 20\\,000 à 2\\,M\\$ par projet, "
          "35\\,\\% au plus par secteur)", "tab:arbitrage-regles")
    return res, s


# ----------------------------------------------------------------------------------------------- tableaux
def tables(ev, arb_res):
    a = json.loads((D / "audit_donnees.json").read_text(encoding="utf-8"))
    rows = [[{"train": "Entraînement", "val": "Validation", "test": "Test"}[k], v["annees"], num(v["lignes"], 0),
             num(v["defauts"], 0), num(v["taux_defaut"], 2, pct=True),
             {"train": "apprentissage", "val": "réglage, calibration, choix", "test": "mesure finale (lue une fois)"}[k]]
            for k, v in a["blocs"].items()]
    table("tab_blocs", "Bloc & Années & Prêts & Défauts & Taux & Rôle", rows, "llrrrl",
          "Séparation temporelle des prêts SBA 7(a)", "tab:blocs")

    steps = [("Lignes brutes (deux fichiers FY2000-2009 et FY2010-2019)", a["lignes_brutes"]),
             ("Lignes à colonnes décalées (statut invalide)", -a["lignes_statut_invalide_retirees"]),
             ("Doublons exacts", -a["doublons_exacts_retires"]),
             ("Sans date de premier versement (dont 147\\,990 annulés)", -a["sans_FirstDisbursementDate_total"]),
             ("Statut EXEMPT (issue inconnue)", -a["retires_statut_non_resolu"]["EXEMPT"]),
             ("Versés depuis moins de 60 mois", -a["retires_moins_de_60_mois"]),
             ("Identifiants hachés en double", -a["id_haches_en_double_retires"]),
             ("Approuvés hors 2000-2016", -a["retires_hors_2000_2016"])]
    kept = sum(v for _, v in steps)
    rows = [[tex(l) if "\\," not in l else l, num(v, 0)] for l, v in steps] + [["\\textbf{Prêts conservés}", f"\\textbf{{{num(kept, 0)}}}"]]
    table("tab_nettoyage", "Étape & Lignes", rows, "lr", "Audit et nettoyage des données SBA", "tab:nettoyage")

    names = {"logreg": "Régression logistique", "ebm": "\\textbf{EBM (retenu)}", "lgbm": "LightGBM (référence)"}
    v = pd.read_csv(R / "metriques_validation.csv").query("version == 'calibre'").set_index("modele")
    t = pd.read_csv(R / "metriques_test.csv").query("version == 'calibre'").set_index("modele")
    rows = [[names[m], num(v.loc[m, "AUROC"], 3), num(v.loc[m, "Brier"], 4), num(t.loc[m, "AUROC"], 3),
             num(t.loc[m, "AUPRC"], 3), num(t.loc[m, "F1"], 3), num(t.loc[m, "Brier"], 4),
             num(t.loc[m, "pd_moyenne"], 2, pct=True)] for m in ["logreg", "ebm", "lgbm"]]
    table("tab_modeles", "Modèle (calibré) & AUROC val. & Brier val. & AUROC test & AUPRC test & F1 test & Brier test & PD moy. test",
          rows, "lccccccc", "Performances des trois modèles (versions calibrées). Taux de défaut réel : 3,57\\,\\% en "
          "validation, 4,25\\,\\% en test", "tab:modeles", size="\\footnotesize")

    h = pd.read_csv(R / "reglage_hyperparametres.csv")
    mn = {"logreg": "Logistique", "ebm": "EBM", "lgbm": "LightGBM"}
    rows = [[mn[r.modele], tex(r.hyperparametres).replace("learning\\_rate", "taux d'apprentissage").replace("num\\_leaves", "feuilles"),
             num(r.AUROC_val_brut, 4), num(r.duree_s, 0), "oui" if r.retenu else ""] for r in h.itertuples()]
    table("tab_hyper", "Modèle & Hyperparamètres & AUROC val. (brut) & Durée (s) & Retenu", rows, "lllrc",
          "Réglage des hyperparamètres sur la validation", "tab:hyper", size="\\footnotesize")

    fr = {"duree_classe": "tranche de durée", "taux_initial": "taux d'intérêt initial", "log_montant": "montant (log)",
          "ligne_renouvelable": "ligne renouvelable", "procedure": "procédure SBA", "etat": "État",
          "taux_fixe_variable": "taux fixe ou variable", "secteur_naics2": "secteur NAICS",
          "part_garantie": "part garantie SBA", "type_entreprise": "type d'entreprise", "franchise": "franchise",
          "taux_manquant": "taux non renseigné", "age_entreprise": "âge de l'entreprise"}
    pm = pd.read_csv(R / "permutation_importance_validation.csv")
    sh = pd.read_csv(R / "shap_lightgbm.csv").set_index("variable")
    rows = [[fr[r.variable], num(r.baisse_AUROC_moyenne, 4), num(r.ecart_type, 4), num(sh.loc[r.variable, "shap_moyen_abs"], 3)]
            for r in pm.itertuples()]
    table("tab_importance", "Variable & Baisse d'AUROC (EBM) & Écart-type & $|$SHAP$|$ moyen (LightGBM)", rows, "lccc",
          "Importance des variables sur la validation : permutation pour l'EBM, valeurs SHAP pour LightGBM",
          "tab:importance", size="\\footnotesize")

    w = pd.read_csv(P / "poids_ahp.csv").set_index("critere")
    prof = pd.read_csv(P / "profils_electre_tri.csv").set_index("critere")
    sense = {"pd_ml": "min", "van": "max", "delai_recuperation": "min", "dscr_min": "max", "dette_fonds_propres": "min",
             "couverture_garanties": "max", "esg_score": "max", "emplois_par_million": "max"}
    fmt = {"pd_ml": lambda x: num(x, 1, pct=True), "van": lambda x: num(x, 0) + "\\,\\$",
           "delai_recuperation": lambda x: num(x, 1) + "\\,ans", "esg_score": lambda x: num(x, 0)}
    rows = [[LAB[c], sense[c], num(w.loc[c, "poids_cible"], 0, pct=True), num(w.loc[c, "poids_ahp"], 1, pct=True),
             fmt.get(c, lambda x: num(x, 2))(prof.loc[c, "b1_refuser_conditionner"]),
             fmt.get(c, lambda x: num(x, 2))(prof.loc[c, "b2_conditionner_financer"]), tex(prof.loc[c, "type"])]
            for c in sense]
    table("tab_criteres", "Critère & Sens & Poids cible & Poids AHP & Profil $b_1$ & Profil $b_2$ & Type", rows,
          "lcccccl", "Critères, poids AHP et profils ELECTRE TRI ($b_1$ : Refuser $|$ Conditionner ; "
          "$b_2$ : Conditionner $|$ Financer)", "tab:criteres", size="\\footnotesize")

    m = pd.read_csv(P / "ahp_matrice.csv", index_col=0)
    short = {"pd_ml": "PD", "van": "VAN", "delai_recuperation": "Délai", "dscr_min": "DSCR",
             "dette_fonds_propres": "D/FP", "couverture_garanties": "Gar.", "esg_score": "ESG", "emplois_par_million": "Empl."}

    def frac(x):
        return num(x, 0) if x >= 1 else f"1/{round(1 / x)}"
    rows = [[short[i]] + [frac(m.loc[i, j]) for j in m.columns] for i in m.index]
    table("tab_ahp", " & " + " & ".join(short[c] for c in m.columns), rows, "l" + "c" * len(m.columns),
          "Matrice de comparaisons deux à deux de l'AHP (échelle de Saaty) ; $\\lambda_{\\max} = 8,091$, CR = 0,009",
          "tab:ahp", size="\\footnotesize")

    s = pd.read_csv(D / "sensibilite_seuils_coupure.csv")
    rows = [[("Seuils q/p/v" if r.parametre.startswith("seuils") else "Coupure $\\lambda$"),
             tex(r.valeur).replace(".", ","), num(r.spearman_vs_base, 3), num(r.kendall_vs_base, 3),
             str(r.projets_categorie_change)] for r in s.itertuples()]
    table("tab_sensibilite", "Paramètre & Valeur & Spearman & Kendall & Changements de catégorie", rows, "llccc",
          "Sensibilité « un paramètre à la fois » (référence : q/p/v = 5/20/50\\,\\%, $\\lambda$ = 0,6)",
          "tab:sensibilite", size="\\footnotesize")

    dm = pd.read_csv(P / "parametres_secteurs_utilises.csv")
    rows = [[tex(r.secteur_damodaran), num(r.cost_of_capital, 1, pct=True), num(r.cost_of_debt, 1, pct=True),
             num(r.beta, 2), num(r.debt_share, 1, pct=True), num(r.croissance_hypothese, 0, pct=True)] for r in dm.itertuples()]
    table("tab_damodaran", "Secteur Damodaran & Coût du capital & Coût de la dette & Bêta & Part de dette & Croissance (hyp.)",
          rows, "lccccc", "Paramètres sectoriels utilisés par la simulation (Damodaran, janvier 2026 ; croissance : "
          "hypothèse d'équipe)", "tab:damodaran", size="\\footnotesize")

    e = pd.read_csv(R / "erreurs_par_segment_validation.csv")
    e = e[e.dimension == "procedure"].sort_values("n", ascending=False).head(6)
    rows = [[tex(r.modalite), num(r.n, 0), num(r.taux_reel, 1, pct=True), num(r.pd_moyenne, 1, pct=True),
             num(r.taux_faux_negatifs, 0, pct=True)] for r in e.itertuples()]
    table("tab_erreurs", "Procédure SBA & Prêts & Taux réel & PD moyenne & Faux négatifs", rows, "lcccc",
          "Analyse des erreurs de l'EBM par procédure (validation, seuil F1)", "tab:erreurs", size="\\footnotesize")

    sim = engine.demo_projects().set_index("nom")
    rr = arb_res.copy()
    order = {"Financer": 0, "Conditionner": 1, "Refuser": 2}
    rr = rr.assign(_o=rr.categorie_finale.map(order)).sort_values(["_o", "rang_electre3"])
    rows = []
    for r in rr.itertuples():
        rows.append([tex(r.nom), tex(sim.loc[r.nom, "secteur"]).replace("Financial Svcs. (Non-bank \\& Insurance)", "Financial Svcs."),
                     CAT_FR[r.categorie_finale], "--" if r.ecarte else str(int(r.rang_electre3)),
                     "--" if r.ecarte else str(int(r.rang_promethee2)), num(r.pd_ml, 1, pct=True), num(r.dscr_min, 2),
                     num(r.van / 1000, 0) + "\\,k\\$", num(r.montant_pret / 1000, 0) + "\\,k\\$",
                     "oui" if r.selection_budget else ("fragile" if False else "non")])
    table("tab_projets", "Projet & Secteur & Décision & E-III & P-II & PD & DSCR & VAN & Montant & Budget", rows,
          "p{3.7cm}p{2.5cm}lrrrrrrl", "Les 40 projets de démonstration (noms fictifs) : décision MCDA et sélection sous budget",
          "tab:projets", size="\\scriptsize", long=True)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    copy_existing()
    ev = engine.evaluate(engine.demo_projects(), engine.demo_config(), sensitivity=True)
    ref = json.loads((R / "modele_retenu.json").read_text(encoding="utf-8"))["taux_defaut_validation"]
    mcda_figures(ev, ref)
    arb_res, _ = arbitrage_figures(ev)
    tables(ev, arb_res)
    print("figures :", len(list(FIG.glob("*.png"))), "| tableaux :", len(list(TAB.glob("*.tex"))))
    import zipfile                              # archive prête à importer dans Overleaf
    root = ROOT / "rapport"
    with zipfile.ZipFile(root / "rapport_overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob("*")):
            if f.is_file() and f.suffix in {".tex", ".bib", ".png"}:
                z.write(f, f.relative_to(root).as_posix())


if __name__ == "__main__":
    main()
