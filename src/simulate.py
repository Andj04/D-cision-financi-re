"""Étape 6 : simulation des indicateurs financiers pour 40 prêts réels du jeu de test (décisions 6 à 12).

Principes : graine fixe ; simulation INDÉPENDANTE de l'issue réelle (default_60m n'est jamais lue ici) ;
aucune variable simulée ne sert à entraîner ou évaluer le modèle de risque.
Ordre : 6 tirage -> 8 NAICS/Damodaran -> 7 paramètres -> 11 fonds propres/dette -> 9 DSCR, flux, VAN, délai
        -> 10 garanties -> 12 ESG.
Usage : python src/simulate.py  (écrit params/*.csv et data/processed/simulation_detail.csv)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from damodaran import industry_table
from features import duree_classe

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026                  # graine du tirage des 40 prêts ET (par défaut) des indicateurs simulés
# La durée exacte (TermInMonths) est altérée après l'octroi pour les prêts en défaut (durée non ronde : ~19 % de
# défaut sur le test contre ~0,4 % pour une durée ronde). La simulation n'utilise donc que la TRANCHE de durée,
# convertie en une durée standard (hypothèse d'équipe), pour rester indépendante de l'issue réelle.
DUREE_STANDARD_ANS = {"<=60 mois": 5, "61-120 mois": 10, ">120 mois": 25}
N_PROJECTS = 40
DISPERSION = 0.20            # décision 7 : +/- 20 % autour de la valeur sectorielle (choix d'équipe, à tester)
DEBT_SHARE = (0.50, 0.80)    # décision 11 : part financée par la dette
COLLATERAL_RATIO = (0.5, 1.5)  # décision 10
DSCR_BANDS = [("faible", 0.20, (0.8, 1.0)), ("moyen", 0.60, (1.0, 1.5)), ("solide", 0.20, (1.5, 2.0))]  # décision 9
ESG_CENTER = {"A": 30, "B": 60, "C": 85}   # décision 12
ESG_NOISE = 5.0
TERMINAL_GROWTH = 0.0        # hypothèse : valeur résiduelle = perpétuité sans croissance, décotée
TERMINAL_HAIRCUT = 0.5       # hypothèse prudente : on ne retient que 50 % de la valeur résiduelle
MAX_HORIZON = 10             # années de flux explicites
SHOCK_SD = 0.05              # bruit annuel des flux (pour que « le pire moment » ait un sens)

# Décision 8 : tableau fait à la main NAICS (2 chiffres) -> secteur Damodaran (feuille Industry Averages)
NAICS_TO_DAMODARAN = {
    "11": "Farming/Agriculture", "21": "Metals & Mining", "22": "Utility (General)",
    "23": "Engineering/Construction", "31": "Food Processing", "32": "Chemical (Diversified)",
    "33": "Machinery", "42": "Retail (Distributors)", "44": "Retail (General)", "45": "Retail (General)",
    "48": "Transportation", "49": "Transportation", "51": "Information Services",
    "52": "Financial Svcs. (Non-bank & Insurance)", "53": "Real Estate (Operations & Services)",
    "54": "Business & Consumer Services", "55": "Diversified", "56": "Business & Consumer Services",
    "61": "Education", "62": "Healthcare Support Services", "71": "Recreation", "72": "Restaurant/Dining",
    "81": "Business & Consumer Services", "92": "Total Market (without financials)",
}
FALLBACK_SECTOR = "Total Market (without financials)"   # « moyenne du marché » pour les cas flous
# Hypothèses d'équipe (NON sourcées) : croissance annuelle des flux par secteur, à tester en sensibilité
GROWTH_ASSUMPTION = {"Farming/Agriculture": 0.02, "Metals & Mining": 0.03, "Utility (General)": 0.02,
                     "Engineering/Construction": 0.03, "Food Processing": 0.03, "Chemical (Diversified)": 0.03,
                     "Machinery": 0.03, "Retail (Distributors)": 0.03, "Retail (General)": 0.03,
                     "Transportation": 0.03, "Information Services": 0.05,
                     "Financial Svcs. (Non-bank & Insurance)": 0.04,
                     "Real Estate (Operations & Services)": 0.03, "Business & Consumer Services": 0.04,
                     "Diversified": 0.03, "Education": 0.03, "Healthcare Support Services": 0.04, "Recreation": 0.03,
                     "Restaurant/Dining": 0.03, "Total Market (without financials)": 0.03}
ESG_CLASS_PROBS = {  # décision 12 : probabilités de catégorie A/B/C selon le groupe de secteurs
    "industriel": {"A": 0.40, "B": 0.50, "C": 0.10},   # mines, énergie, industrie, construction, agriculture
    "services": {"A": 0.05, "B": 0.25, "C": 0.70},     # commerce et services
}
INDUSTRIAL_NAICS = {"11", "21", "22", "23", "31", "32", "33", "48", "49"}


def annuity(rate: float, years: int) -> float:
    return 1 / years if rate == 0 else rate / (1 - (1 + rate) ** -years)


def damodaran_params() -> pd.DataFrame:
    t = industry_table().set_index("Industry Name")
    keep = {"cost_of_capital": "Cost of Capital", "cost_of_debt": "Cost of Debt", "tax_rate": "Tax Rate",
            "beta": "Beta", "debt_share": "D/(D+E)"}
    return t[list(keep.values())].rename(columns={v: k for k, v in keep.items()})


def simulate(test: pd.DataFrame, pd_ml: pd.Series, seed: int = SEED, n: int = N_PROJECTS,
             indicator_seed: int | None = None) -> pd.DataFrame:
    """seed : tirage des prêts ; indicator_seed : tirage des indicateurs (par défaut = seed). Faire varier
    indicator_seed seul garde les mêmes 40 prêts et mesure la sensibilité des résultats à la simulation."""
    rng = np.random.default_rng(seed if indicator_seed is None else indicator_seed)
    dam = damodaran_params()
    test = test.drop(columns=["default_60m"], errors="ignore")   # la cible n'est même pas accessible ici
    sample = test.sample(n=n, random_state=seed)          # décision 6
    # décision 9 : répartition exacte 20 % faibles / 60 % moyens / 20 % solides, ordre mélangé (graine fixe)
    bands = np.repeat(np.arange(len(DSCR_BANDS)), [int(round(b[1] * n)) for b in DSCR_BANDS])
    bands = rng.permutation(bands)
    rows = []
    for i, (_, r) in enumerate(sample.iterrows()):
        naics2 = str(r["NaicsCode"])[:2] if pd.notna(r["NaicsCode"]) else ""
        sector = NAICS_TO_DAMODARAN.get(naics2, FALLBACK_SECTOR)                 # décision 8
        p = dam.loc[sector]

        def jit(v):                                                              # décision 7
            return float(v) * rng.uniform(1 - DISPERSION, 1 + DISPERSION)

        disc = jit(p["cost_of_capital"])
        growth = jit(GROWTH_ASSUMPTION[sector])
        loan = float(r["GrossApproval"])
        n_years = DUREE_STANDARD_ANS[duree_classe(pd.Series([r["TermInMonths"]])).iloc[0]]
        rate = float(r["InitialInterestRate"]) / 100 if pd.notna(r["InitialInterestRate"]) else jit(p["cost_of_debt"])

        d = rng.uniform(*DEBT_SHARE)                                  # décision 11
        invest = loan / d
        equity = invest - loan
        de_ratio = loan / equity

        ds = loan * annuity(rate, n_years)                            # service annuel de la dette
        band = int(bands[i])                                          # décision 9 : DSCR piloté
        target = rng.uniform(*DSCR_BANDS[band][2])
        horizon = min(n_years, MAX_HORIZON)
        t = np.arange(1, horizon + 1)
        cf = (1 + growth) ** (t - 1) * (1 + rng.normal(0, SHOCK_SD, horizon))
        cf = cf * target / (cf / ds).min()                            # le DSCR minimum vaut exactement la cible
        dscr_min = float((cf / ds).min())
        terminal = TERMINAL_HAIRCUT * cf[-1] * (1 + TERMINAL_GROWTH) / (disc - TERMINAL_GROWTH)
        van = float(-invest + (cf / (1 + disc) ** t).sum() + terminal / (1 + disc) ** horizon)
        cum = np.cumsum(cf)                                           # délai de récupération (non actualisé)
        if cum[-1] >= invest:
            k = int(np.argmax(cum >= invest))
            prev = cum[k - 1] if k else 0.0
            payback = k + (invest - prev) / cf[k]
        else:
            payback = horizon + (invest - cum[-1]) / cf[-1]
        payback = float(min(payback, 30.0))

        coverage = 0.0 if r["CollateralInd"] == "N" else float(rng.uniform(*COLLATERAL_RATIO))   # décision 10

        group = "industriel" if naics2 in INDUSTRIAL_NAICS else "services"     # décision 12
        cls = rng.choice(list(ESG_CLASS_PROBS[group]), p=list(ESG_CLASS_PROBS[group].values()))
        esg = float(np.clip(ESG_CENTER[cls] + rng.normal(0, ESG_NOISE), 0, 100))

        jobs = float(r["JobsSupported"]) / (loan / 1e6)               # critère SBA réel
        rows.append({"loan_id": r["loan_id"], "secteur_naics2": naics2 or "inconnu", "secteur_damodaran": sector,
                     "etat": r["BorrState"], "montant_pret": loan, "duree_standard_annees": n_years,
                     "investissement": invest, "fonds_propres": equity, "taux_dette": rate,
                     "taux_actualisation": disc, "croissance_flux": growth, "categorie_dscr": DSCR_BANDS[band][0],
                     "categorie_esg": cls, "van": van, "delai_recuperation": payback, "dscr_min": dscr_min,
                     "dette_fonds_propres": de_ratio, "couverture_garanties": coverage, "esg_score": esg,
                     "emplois_par_million": jobs})
    return pd.DataFrame(rows).merge(pd_ml.rename("pd_ml"), left_on="loan_id", right_index=True, how="left")


if __name__ == "__main__":
    pred = pd.read_csv(ROOT / "data/processed/test_predictions.csv").set_index("loan_id")["pd_ml"]
    test = pd.read_parquet(ROOT / "data/processed/test.parquet")
    sim = simulate(test, pred)
    sim.to_csv(ROOT / "data/processed/simulation_detail.csv", index=False)
    dam = damodaran_params()
    used = sorted(set(sim["secteur_damodaran"]))
    (ROOT / "params").mkdir(exist_ok=True)
    pd.DataFrame([{"secteur_damodaran": s, **dam.loc[s].to_dict(), "croissance_hypothese": GROWTH_ASSUMPTION[s]}
                  for s in used]).to_csv(ROOT / "params/parametres_secteurs_utilises.csv", index=False)
    print(sim.describe().T.round(3))
