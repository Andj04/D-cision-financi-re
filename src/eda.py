"""Analyse exploratoire des données SBA 7(a) (obligatoire d'après les consignes) : figures et statistiques.

Lit les trois blocs produits par data_prep.py et écrit dans reports/eda/ :
  * fig_defaut_par_annee.png      taux de défaut et volume par année d'approbation (changement de régime)
  * fig_desequilibre.png          déséquilibre des classes par bloc
  * fig_defaut_par_secteur.png    taux de défaut par secteur (entraînement)
  * fig_duree.png                 défaut par tranche de durée + fuite de la durée exacte (durées rondes ou non)
  * fig_montant.png               distribution des montants et défaut par décile de montant
  * fig_variables_categorielles.png  défaut selon l'âge de l'entreprise, la procédure, la ligne renouvelable, le taux
  * fig_valeurs_manquantes.png    part de valeurs manquantes par variable et par bloc
  * statistiques_descriptives.csv, resume_blocs.csv
Toutes les statistiques sont descriptives : aucune n'est utilisée pour ajuster le modèle.
Usage : python src/eda.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

ROOT = Path(__file__).resolve().parents[1]
D, OUT = ROOT / "data/processed", ROOT / "reports/eda"
style.apply()
RED = "#C0503F"
NAICS_FR = {"11": "Agriculture", "21": "Mines", "22": "Services publics", "23": "Construction", "31": "Industrie",
            "32": "Industrie", "33": "Industrie", "42": "Commerce de gros", "44": "Commerce de détail",
            "45": "Commerce de détail", "48": "Transport", "49": "Transport", "51": "Information", "52": "Finance",
            "53": "Immobilier", "54": "Services professionnels", "55": "Gestion d'entreprises",
            "56": "Services administratifs", "61": "Enseignement", "62": "Santé", "71": "Arts et loisirs",
            "72": "Hébergement, restauration", "81": "Autres services", "92": "Administration"}
BLOCS = {"train": "Entraînement 2000-2012", "val": "Validation 2013-2014", "test": "Test 2015-2016"}
ROUND_TERMS = {12, 24, 36, 48, 60, 72, 84, 96, 120, 180, 240, 300}
COLS = ["ApprovalFY", "default_60m", "GrossApproval", "SBAGuaranteedApproval", "TermInMonths", "InitialInterestRate",
        "FixedorVariableInterestInd", "RevolverStatus", "NaicsCode", "BorrState", "BusinessType", "BusinessAge",
        "FranchiseCode", "ProcessingMethod", "CollateralInd", "JobsSupported"]


def load() -> pd.DataFrame:
    parts = []
    for b in BLOCS:
        d = pd.read_parquet(D / f"{b}.parquet", columns=COLS)
        d["bloc"] = b
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["secteur"] = df["NaicsCode"].astype("string").str[:2].map(NAICS_FR).fillna("Non renseigné")
    df["duree_classe"] = pd.cut(df["TermInMonths"].astype(float), [-1, 60, 120, 10_000],
                                labels=["≤ 60 mois", "61-120 mois", "> 120 mois"])
    return df


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def rate_bar(ax, s: pd.Series, title: str, ref: float | None = None, color=style.GREEN):
    s = s.sort_values()
    ax.barh(s.index.astype(str), 100 * s.values, color=color)
    for y, v in enumerate(s.values):
        ax.text(100 * v + 0.3, y, f"{100 * v:.1f} %".replace(".", ","), va="center", fontsize=8, color=style.DARK)
    if ref is not None:
        ax.axvline(100 * ref, ls="--", lw=1, color=style.GREY)
    ax.set_xlabel("taux de défaut à 60 mois (%)")
    ax.set_title(title)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    tr = df[df.bloc == "train"]
    base = tr.default_60m.mean()

    # 1. par année : volume et taux de défaut, blocs en fond
    g = df.groupby("ApprovalFY").agg(prets=("default_60m", "size"), taux=("default_60m", "mean"))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax2 = ax.twinx()
    ax2.bar(g.index, g.prets / 1000, color="#D9DEE0", width=0.7, zorder=0)
    ax2.set_ylabel("prêts approuvés (milliers)", color=style.GREY)
    ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.plot(g.index, 100 * g.taux, marker="o", color=style.GREEN, lw=2)
    top_ = 100 * g.taux.max() * 1.12
    ax.set_ylim(0, top_)
    for (lo, hi), lab, a in zip([(1999.5, 2012.5), (2012.5, 2014.5), (2014.5, 2016.5)],
                                ["Entraînement", "Valid.", "Test"], [0.03, 0.10, 0.18]):
        ax.axvspan(lo, hi, color=style.GREEN, alpha=a, lw=0)
        ax.text((lo + hi) / 2, top_ * 0.95, lab, ha="center", fontsize=8.5, color=style.DARK)
    ax.set_ylabel("taux de défaut à 60 mois (%)")
    ax.set_xticks(range(2000, 2017, 2))
    ax.set_title("Le taux de défaut culmine avec la crise de 2008 puis retombe sous 5 %")
    save(fig, "fig_defaut_par_annee.png")

    # 2. déséquilibre des classes par bloc
    b = df.groupby("bloc").default_60m.agg(["size", "sum", "mean"]).reindex(list(BLOCS))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(len(b))
    ax.barh(y, b["size"] - b["sum"], color="#C9D7D3", label="remboursé ou en cours (0)")
    ax.barh(y, b["sum"], left=b["size"] - b["sum"], color=RED, label="défaut à 60 mois (1)")
    for i, (n, d, m) in enumerate(zip(b["size"], b["sum"], b["mean"])):
        ax.text(n + b["size"].max() * 0.015, i,
                f"{n:,.0f} prêts · {100 * m:.2f} % de défauts".replace(",", " ").replace(".", ","),
                va="center", fontsize=8.5, color=style.DARK)
    ax.set_yticks(y, list(BLOCS.values()))
    ax.set_xlim(0, b["size"].max() * 1.6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f} k"))
    ax.set_xlabel("nombre de prêts")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax.set_title("Des classes très déséquilibrées : 3,6 à 13,5 % de défauts selon le bloc")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_desequilibre.png")

    # 3. par secteur (entraînement), secteurs d'au moins 2 000 prêts
    s = tr.groupby("secteur").default_60m.agg(["size", "mean"])
    s = s[s["size"] >= 2000]["mean"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    rate_bar(ax, s, "Taux de défaut par secteur (entraînement 2000-2012)", ref=base)
    ax.text(100 * base + 0.3, -0.9, f"moyenne {100 * base:.1f} %", fontsize=8, color=style.GREY)
    save(fig, "fig_defaut_par_secteur.png")

    # 4. durée : tranches retenues + fuite de la durée exacte
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    rate_bar(axes[0], tr.groupby("duree_classe", observed=True).default_60m.mean(),
             "Par tranche de durée (variable retenue)")
    ronde = tr["TermInMonths"].isin(ROUND_TERMS).map({True: "durée ronde (12, 24, 60, 120 mois…)",
                                                     False: "durée non ronde (87, 129 mois…)"})
    rate_bar(axes[1], tr.groupby(ronde).default_60m.mean(), "Durée exacte : la fuite détectée", color=RED)
    save(fig, "fig_duree.png")

    # 5. montants
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    bins = np.logspace(np.log10(5e3), np.log10(5e6), 40)
    for lab, c, sub in [("pas de défaut", style.GREY, tr[tr.default_60m == 0]), ("défaut", RED, tr[tr.default_60m == 1])]:
        axes[0].hist(sub["GrossApproval"], bins=bins, density=True, alpha=0.55, color=c, label=lab)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("montant approuvé ($, échelle log)")
    axes[0].set_ylabel("densité")
    axes[0].legend()
    axes[0].set_title("Distribution des montants (entraînement)")
    dec = pd.qcut(tr["GrossApproval"], 10, duplicates="drop")
    r = tr.groupby(dec, observed=True).default_60m.mean()
    axes[1].bar(range(len(r)), 100 * r.values, color=style.GREEN)
    axes[1].set_xticks(range(len(r)), [f"D{i + 1}" for i in range(len(r))])
    axes[1].set_xlabel("décile de montant (D1 = plus petits prêts)")
    axes[1].set_ylabel("taux de défaut (%)")
    axes[1].set_title("Les petits prêts font plus souvent défaut")
    save(fig, "fig_montant.png")

    # 6. variables catégorielles
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    age_fr = {"Existing, 5 or more years": "5 ans ou plus", "New, Less than 1 Year old": "moins d'un an",
              "Startup, Loan Funds will Open Business": "création (startup)", "Unanswered": "non renseigné",
              "Less than 3 years old but at least 2": "2 à 3 ans", "Less than 4 years old but at least 3": "3 à 4 ans",
              "Less than 5 years old but at least 4": "4 à 5 ans", "Change of Ownership": "changement de propriétaire"}
    age = tr.groupby(tr["BusinessAge"].map(lambda v: age_fr.get(v, v))).default_60m.agg(["size", "mean"])
    rate_bar(axes[0, 0], age[age["size"] >= 1000]["mean"], "Âge de l'entreprise", ref=base)
    proc = tr.groupby("ProcessingMethod").default_60m.agg(["size", "mean"]).sort_values("size", ascending=False).head(8)
    rate_bar(axes[0, 1], proc["mean"], "Procédure SBA (8 plus fréquentes)", ref=base)
    rate_bar(axes[1, 0], tr.groupby("RevolverStatus").default_60m.mean().rename({"Y": "oui", "N": "non"}),
             "Ligne de crédit renouvelable", ref=base)
    rate_bar(axes[1, 1], tr.groupby(tr["InitialInterestRate"].isna().map({True: "non renseigné", False: "renseigné"}))
             .default_60m.mean(), "Taux d'intérêt initial", ref=base)
    save(fig, "fig_variables_categorielles.png")

    # 7. valeurs manquantes
    miss_cols = ["InitialInterestRate", "FranchiseCode", "BusinessAge", "NaicsCode", "BusinessType",
                 "FixedorVariableInterestInd", "CollateralInd", "TermInMonths", "GrossApproval"]
    m = df.groupby("bloc")[miss_cols].apply(lambda x: x.isna().mean()).T[list(BLOCS)]
    fig, ax = plt.subplots(figsize=(9, 4))
    yy = np.arange(len(m))
    for k, (bl, col) in enumerate(zip(BLOCS, [style.GREEN, style.GREY, "#D39B2A"])):
        ax.barh(yy + (k - 1) * 0.27, 100 * m[bl], height=0.27, color=col, label=BLOCS[bl])
    ax.set_yticks(yy, m.index)
    ax.set_xlabel("valeurs manquantes (%)")
    ax.legend()
    ax.set_title("Valeurs manquantes : le taux d'intérêt manque surtout avant 2013")
    save(fig, "fig_valeurs_manquantes.png")

    # statistiques
    num = df[["GrossApproval", "SBAGuaranteedApproval", "TermInMonths", "InitialInterestRate", "JobsSupported"]]
    num.describe(percentiles=[.25, .5, .75]).T.round(2).to_csv(OUT / "statistiques_descriptives.csv")
    b.rename(columns={"size": "prets", "sum": "defauts", "mean": "taux_defaut"}).round(4).to_csv(OUT / "resume_blocs.csv")
    print("figures écrites dans", OUT)
    print(b.round(4).to_string())
    print(s.sort_values().round(3).to_string())


if __name__ == "__main__":
    main()
