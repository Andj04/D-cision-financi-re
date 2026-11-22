"""Étape 8 : MCDA déterministe (aucun agent, aucun LLM) : AHP, filtre éliminatoire, ELECTRE III,
ELECTRE TRI (comparaison) et PROMETHEE II (comparaison), plus l'analyse de sensibilité.

Implémenté avec numpy pour que chaque étape soit lisible et testable à la main (voir tests/test_mcda.py).
Convention interne : tous les critères sont ramenés au sens « plus grand = meilleur »
(les critères à minimiser sont multipliés par -1) ; les seuils q, p, v sont exprimés en unités du critère.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

CRITERIA = ["pd_ml", "van", "delai_recuperation", "dscr_min", "dette_fonds_propres",
            "couverture_garanties", "esg_score", "emplois_par_million"]
SENSE = {"pd_ml": -1, "van": 1, "delai_recuperation": -1, "dscr_min": 1, "dette_fonds_propres": -1,
         "couverture_garanties": 1, "esg_score": 1, "emplois_par_million": 1}
# Poids cibles de la décision 13 (ordre de CRITERIA)
TARGET_WEIGHTS = {"pd_ml": 0.25, "dscr_min": 0.20, "van": 0.15, "couverture_garanties": 0.10,
                  "dette_fonds_propres": 0.10, "delai_recuperation": 0.08, "esg_score": 0.07,
                  "emplois_par_million": 0.05}
CATEGORIES = ["Refuser", "Conditionner", "Financer"]
SAATY = np.array([1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4, 5, 6, 7, 8, 9])
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


# ----------------------------------------------------------------------------- AHP (décision 13)
def ahp_matrix_from_targets(targets: dict[str, float]) -> pd.DataFrame:
    """Matrice de comparaisons deux à deux (28 jugements pour 8 critères) : chaque rapport
    de poids cible est arrondi à l'échelle de Saaty (1/9 ... 9). Les jugements sont donc des entrées explicites."""
    names = list(targets)
    m = np.ones((len(names), len(names)))
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            r = targets[a] / targets[names[j]]
            v = SAATY[np.argmin(np.abs(np.log(SAATY) - np.log(r)))]
            m[i, j], m[j, i] = v, 1 / v
    return pd.DataFrame(m, index=names, columns=names)


def ahp_weights(matrix: pd.DataFrame) -> tuple[pd.Series, float, float]:
    """Poids = vecteur propre principal ; renvoie (poids, lambda_max, ratio de cohérence CR)."""
    a = matrix.values
    vals, vecs = np.linalg.eig(a)
    k = np.argmax(vals.real)
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    n = len(a)
    ci = (vals[k].real - n) / (n - 1)
    cr = ci / RI[n] if RI[n] else 0.0
    return pd.Series(w, index=matrix.index), float(vals[k].real), float(cr)


# ----------------------------------------------------------------------------- filtre (décision 14)
def eliminatory_filter(scores: pd.DataFrame, train_default_rate: float, dscr_floor: float = 1.0,
                       pd_multiple: float = 2.0) -> pd.DataFrame:
    """Écarte un projet si DSCR min < 1 ou pd_ml > 2 x taux de défaut moyen de l'entraînement."""
    out = scores.copy()
    out["motif_ecarte"] = ""
    out.loc[out["dscr_min"] < dscr_floor, "motif_ecarte"] = f"DSCR minimum < {dscr_floor:g}"
    high = out["pd_ml"] > pd_multiple * train_default_rate
    out.loc[high, "motif_ecarte"] = (out.loc[high, "motif_ecarte"].where(out.loc[high, "motif_ecarte"] == "", out.loc[high, "motif_ecarte"] + " ; ")
                                     + f"pd_ml > {pd_multiple:g} x taux de défaut moyen ({train_default_rate:.3f})")
    out["ecarte"] = out["motif_ecarte"] != ""
    return out


# ----------------------------------------------------------------------------- ELECTRE III (décision 15)
@dataclass
class Thresholds:
    q: np.ndarray
    p: np.ndarray
    v: np.ndarray


def make_thresholds(perf: np.ndarray, q_pct=0.05, p_pct=0.20, v_pct=0.50) -> Thresholds:
    """Seuils en pourcentage de l'étendue observée de chaque critère (décision 15)."""
    rng = perf.max(axis=0) - perf.min(axis=0)
    return Thresholds(q_pct * rng, p_pct * rng, v_pct * rng)


def oriented(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([SENSE[c] * df[c].to_numpy(float) for c in CRITERIA])


def credibility(perf: np.ndarray, other: np.ndarray, w: np.ndarray, th: Thresholds) -> np.ndarray:
    """Matrice sigma[i, k] = crédibilité de « perf[i] surclasse other[k] » (concordance atténuée par le discordance)."""
    diff = other[None, :, :] - perf[:, None, :]            # g_j(b) - g_j(a) : > 0 si b meilleur
    q, p, v = th.q, th.p, th.v
    with np.errstate(divide="ignore", invalid="ignore"):
        cj = np.where(diff <= q, 1.0, np.where(diff >= p, 0.0, (p - diff) / (p - q)))
        dj = np.where(diff <= p, 0.0, np.where(diff >= v, 1.0, (diff - p) / (v - p)))
    cj = np.nan_to_num(cj, nan=1.0)
    dj = np.nan_to_num(dj, nan=0.0)
    c = (cj * w).sum(axis=2) / w.sum()
    sigma = c.copy()
    for j in range(perf.shape[1]):
        d = dj[:, :, j]
        mask = d > c
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = np.where(mask, sigma * (1 - d) / (1 - c), sigma)
    return sigma


def _distil(sigma: np.ndarray, idx: list[int], descending: bool) -> list[list[int]]:
    """Distillation (Roy 1991) : renvoie une liste de classes ordonnées (meilleure d'abord si descendante)."""
    ranking, remaining = [], list(idx)
    while remaining:
        cls = _distil_class(sigma, remaining, descending)
        ranking.append(cls)
        remaining = [i for i in remaining if i not in cls]
    return ranking


def _distil_class(sigma: np.ndarray, cand: list[int], descending: bool) -> list[int]:
    sub = sigma[np.ix_(cand, cand)]
    if len(cand) == 1:
        return list(cand)
    s_ = lambda x: 0.30 - 0.15 * x                                    # noqa: E731
    off = sub[~np.eye(len(cand), dtype=bool)]
    lam_max = off.max()
    below = off[off < lam_max - s_(lam_max)]
    lam = below.max() if below.size else 0.0
    out = (sub > lam) & (sub > sub.T + s_(sub))
    np.fill_diagonal(out, False)
    qual = out.sum(axis=1) - out.sum(axis=0)
    best = qual.max() if descending else qual.min()
    keep = [cand[i] for i in range(len(cand)) if qual[i] == best]
    if len(keep) > 1 and len(keep) < len(cand):
        return _distil_class(sigma, keep, descending)
    return keep


def electre_iii(perf: np.ndarray, w: np.ndarray, th: Thresholds) -> dict:
    sigma = credibility(perf, perf, w, th)
    np.fill_diagonal(sigma, 0.0)
    n = len(perf)
    desc = _distil(sigma, list(range(n)), True)
    asc = _distil(sigma, list(range(n)), False)[::-1]
    rank_d, rank_a = np.zeros(n), np.zeros(n)
    for r, cls in enumerate(desc):
        rank_d[cls] = r + 1
    for r, cls in enumerate(asc):
        rank_a[cls] = r + 1
    mean_rank = (rank_d + rank_a) / 2
    net = sigma.sum(axis=1) - sigma.sum(axis=0)           # départage les ex aequo du rang moyen
    order = np.lexsort((-net, mean_rank))
    final = np.empty(n, dtype=int)
    final[order] = np.arange(1, n + 1)
    return {"sigma": sigma, "rank_desc": rank_d, "rank_asc": rank_a, "rank": final, "net_credibility": net}


# ----------------------------------------------------------------------------- ELECTRE TRI (décision 16)
def make_profiles(df: pd.DataFrame, dscr_profiles=(1.0, 1.25)) -> np.ndarray:
    """Profils limites (refuser|conditionner, conditionner|financer) : DSCR 1,0 et 1,25 ; sinon tertiles 33e/66e centile."""
    rows = []
    for k, qtile in enumerate((1 / 3, 2 / 3)):
        row = []
        for c in CRITERIA:
            if c == "dscr_min":
                row.append(SENSE[c] * dscr_profiles[k])
            else:
                row.append(SENSE[c] * np.quantile(df[c].to_numpy(float), qtile if SENSE[c] > 0 else 1 - qtile))
        rows.append(row)
    return np.array(rows)


def electre_tri(perf: np.ndarray, profiles: np.ndarray, w: np.ndarray, th: Thresholds, cut: float = 0.6) -> dict:
    """Affectation pessimiste (principale) et optimiste ; 0 = Refuser, 1 = Conditionner, 2 = Financer."""
    s_ab = credibility(perf, profiles, w, th) >= cut      # a surclasse b_h
    s_ba = credibility(profiles, perf, w, th).T >= cut    # b_h surclasse a
    n = len(perf)
    pess = np.zeros(n, dtype=int)
    for i in range(n):
        for h in range(len(profiles) - 1, -1, -1):        # du profil le plus haut au plus bas
            if s_ab[i, h]:
                pess[i] = h + 1
                break
    opt = np.full(n, len(profiles), dtype=int)
    for i in range(n):
        for h in range(len(profiles)):
            if s_ba[i, h] and not s_ab[i, h]:             # b_h domine strictement a
                opt[i] = h
                break
    return {"pessimiste": pess, "optimiste": opt}


# ----------------------------------------------------------------------------- PROMETHEE II
def promethee_ii(perf: np.ndarray, w: np.ndarray, th: Thresholds) -> dict:
    """Préférence linéaire avec indifférence (type V) : P=0 sous q, 1 au-delà de p ; flux net."""
    diff = perf[:, None, :] - perf[None, :, :]            # g_j(a) - g_j(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        pref = np.where(diff <= th.q, 0.0, np.where(diff >= th.p, 1.0, (diff - th.q) / (th.p - th.q)))
    pref = np.nan_to_num(pref)
    pi = (pref * (w / w.sum())).sum(axis=2)
    n = len(perf)
    phi = (pi.sum(axis=1) - pi.sum(axis=0)) / (n - 1)
    rank = (-phi).argsort().argsort() + 1
    return {"phi": phi, "rank": rank}


# ----------------------------------------------------------------------------- pipeline complet
@dataclass
class McdaConfig:
    q_pct: float = 0.05
    p_pct: float = 0.20
    v_pct: float = 0.50
    cut: float = 0.6


def run_mcda(scores: pd.DataFrame, weights: pd.Series, train_default_rate: float,
             cfg: McdaConfig = McdaConfig(), filter_first: bool = True) -> pd.DataFrame:
    """Applique filtre -> ELECTRE III + ELECTRE TRI + PROMETHEE II sur les survivants."""
    df = eliminatory_filter(scores, train_default_rate) if filter_first else scores.assign(ecarte=False, motif_ecarte="")
    alive = df[~df["ecarte"]]
    w = weights.reindex(CRITERIA).to_numpy(float)
    perf = oriented(alive)
    th = make_thresholds(perf, cfg.q_pct, cfg.p_pct, cfg.v_pct)
    e3 = electre_iii(perf, w, th)
    tri = electre_tri(perf, make_profiles(alive), w, th, cfg.cut)
    pm = promethee_ii(perf, w, th)
    out = df.copy()
    for col in ["rang_electre3", "rang_promethee2", "flux_net_promethee", "credibilite_nette"]:
        out[col] = np.nan
    out["categorie_tri_pessimiste"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    out["categorie_tri_optimiste"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    out.loc[alive.index, "rang_electre3"] = e3["rank"]
    out.loc[alive.index, "credibilite_nette"] = e3["net_credibility"]
    out.loc[alive.index, "categorie_tri_pessimiste"] = [CATEGORIES[i] for i in tri["pessimiste"]]
    out.loc[alive.index, "categorie_tri_optimiste"] = [CATEGORIES[i] for i in tri["optimiste"]]
    out.loc[alive.index, "rang_promethee2"] = pm["rank"]
    out.loc[alive.index, "flux_net_promethee"] = pm["phi"]
    out.loc[out["ecarte"], "categorie_tri_pessimiste"] = "Refuser"
    out.loc[out["ecarte"], "categorie_tri_optimiste"] = "Refuser"
    return out


# ----------------------------------------------------------------------------- sensibilité
def sensitivity(scores: pd.DataFrame, weights: pd.Series, train_default_rate: float, seed: int = 7,
                n_weight_draws: int = 200) -> dict[str, pd.DataFrame]:
    base = run_mcda(scores, weights, train_default_rate)
    alive_idx = base.index[~base["ecarte"]]
    base_rank = base.loc[alive_idx, "rang_electre3"]
    base_cat = base["categorie_tri_pessimiste"]

    def compare(res: pd.DataFrame) -> dict:
        r = res.loc[alive_idx, "rang_electre3"]
        return {"spearman_vs_base": spearmanr(base_rank, r)[0], "kendall_vs_base": kendalltau(base_rank, r)[0],
                "projets_categorie_change": int((res["categorie_tri_pessimiste"] != base_cat).sum())}

    # (a) seuils q/p/v
    rows = []
    for q, p, v in [(0.03, 0.15, 0.40), (0.05, 0.20, 0.50), (0.10, 0.30, 0.60), (0.05, 0.20, 0.80)]:
        rows.append({"parametre": "seuils q/p/v", "valeur": f"{q:.0%}/{p:.0%}/{v:.0%}",
                     **compare(run_mcda(scores, weights, train_default_rate, McdaConfig(q, p, v, 0.6)))})
    # (b) niveau de coupure lambda
    for cut in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        rows.append({"parametre": "coupure lambda", "valeur": f"{cut:g}",
                     **compare(run_mcda(scores, weights, train_default_rate, McdaConfig(cut=cut)))})
    table = pd.DataFrame(rows)

    # (c) scénarios combinés : poids +/-30 %, seuils q/p/v et coupure lambda tirés dans les fourchettes testées
    rng = np.random.default_rng(seed)
    cats, ranks = [], []
    for _ in range(n_weight_draws):
        w = weights * rng.uniform(0.7, 1.3, len(weights))
        cfg = McdaConfig(q_pct=rng.uniform(0.03, 0.10), p_pct=rng.uniform(0.15, 0.30),
                         v_pct=rng.uniform(0.40, 0.80), cut=rng.uniform(0.5, 1.0))
        res = run_mcda(scores, w / w.sum(), train_default_rate, cfg)
        cats.append(res["categorie_tri_pessimiste"].to_numpy())
        ranks.append(res["rang_electre3"].to_numpy())
    cats, ranks = np.array(cats), np.array(ranks)
    frag = pd.DataFrame({
        "part_scenarios_categorie_differente": (cats != base_cat.to_numpy()).mean(axis=0),
        "rang_moyen": np.nanmean(ranks, axis=0), "rang_ecart_type": np.nanstd(ranks, axis=0),
    }, index=base.index)
    frag["decision_fragile"] = frag["part_scenarios_categorie_differente"] >= 0.25

    # comparaison des trois méthodes
    mask = ~base["ecarte"]
    meth = pd.DataFrame({
        "paire": ["ELECTRE III vs PROMETHEE II"],
        "spearman": [spearmanr(base.loc[mask, "rang_electre3"], base.loc[mask, "rang_promethee2"])[0]],
        "kendall": [kendalltau(base.loc[mask, "rang_electre3"], base.loc[mask, "rang_promethee2"])[0]],
    })
    pess_opt = int((base["categorie_tri_pessimiste"] != base["categorie_tri_optimiste"]).sum())
    meth["projets_ou_pessimiste_differe_de_optimiste"] = pess_opt
    return {"seuils_et_coupure": table, "fragilite_poids": frag, "comparaison_methodes": meth}
