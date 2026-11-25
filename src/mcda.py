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
    """Matrice PROPOSÉE (point de départ seulement) : chaque rapport de poids cible est arrondi à l'échelle de
    Saaty. Utilisée pour initialiser params/ahp_jugements.csv ; la matrice de travail est lue dans ce fichier."""
    names = list(targets)
    m = np.ones((len(names), len(names)))
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            r = targets[a] / targets[names[j]]
            v = SAATY[np.argmin(np.abs(np.log(SAATY) - np.log(r)))]
            m[i, j], m[j, i] = v, 1 / v
    return pd.DataFrame(m, index=names, columns=names)


def judgments_table(matrix: pd.DataFrame) -> pd.DataFrame:
    """Matrice -> 28 jugements lisibles (critère A, critère B, importance de A par rapport à B, échelle 1/9..9)."""
    names, rows = list(matrix.index), []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = float(matrix.loc[a, b])
            rows.append({"critere_a": a, "critere_b": b,
                         "jugement": f"{v:g}" if v >= 1 else f"1/{round(1 / v):d}"})
    return pd.DataFrame(rows)


def ahp_matrix_from_judgments(table: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    """Reconstruit la matrice réciproque à partir des jugements (« 3 » ou « 1/3 »), contrôle l'échelle de Saaty."""
    names = names or list(dict.fromkeys(list(table["critere_a"]) + list(table["critere_b"])))
    m = pd.DataFrame(np.ones((len(names), len(names))), index=names, columns=names)
    for _, r in table.iterrows():
        s = str(r["jugement"]).strip()
        v = 1 / float(s.split("/")[1]) if s.startswith("1/") else float(s)
        assert np.isclose(SAATY, v).any(), f"jugement hors échelle de Saaty : {s}"
        m.loc[r["critere_a"], r["critere_b"]], m.loc[r["critere_b"], r["critere_a"]] = v, 1 / v
    n_expected = len(names) * (len(names) - 1) // 2
    assert len(table) == n_expected, f"{len(table)} jugements au lieu de {n_expected}"
    return m


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
def eliminatory_filter(scores: pd.DataFrame, reference_rate: float, dscr_floor: float = 1.0,
                       pd_multiple: float = 2.0) -> pd.DataFrame:
    """Écarte un projet si DSCR min < 1 ou pd_ml > 2 x taux de défaut de référence.

    Le taux de référence doit être celui du régime sur lequel pd_ml est calibrée (validation 2013-2014) :
    comparer une probabilité calibrée sur 2013-2014 au taux 2000-2012 (crise de 2008, 13,5 %) rendrait le
    filtre inopérant (seuil de 27 % jamais atteint)."""
    out = scores.copy()
    out["motif_ecarte"] = ""
    out.loc[out["dscr_min"] < dscr_floor, "motif_ecarte"] = f"DSCR minimum < {dscr_floor:g}"
    high = out["pd_ml"] > pd_multiple * reference_rate
    out.loc[high, "motif_ecarte"] = (out.loc[high, "motif_ecarte"].where(out.loc[high, "motif_ecarte"] == "", out.loc[high, "motif_ecarte"] + " ; ")
                                     + f"probabilité de défaut > {pd_multiple:g} x taux de référence "
                                       f"({100 * pd_multiple * reference_rate:.1f} %)")
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
    # Le rang final est un ordre TOTAL de lecture (moyenne des deux distillations, ex aequo départagés par la
    # crédibilité nette). Les deux préordres sont aussi renvoyés : un écart important entre eux signale un projet
    # INCOMPARABLE à une partie des autres (information propre à ELECTRE III, à ne pas masquer).
    return {"sigma": sigma, "rank_desc": rank_d, "rank_asc": rank_a, "rank": final, "net_credibility": net}


# ----------------------------------------------------------------------------- ELECTRE TRI (décision 16)
# Profils limites b1 (Refuser | Conditionner) et b2 (Conditionner | Financer).
# Repères ABSOLUS quand il en existe un (la catégorie ne dépend alors pas du lot de projets) ; sinon tertiles
# du lot (relatif, signalé). « ref » = taux de défaut de référence (régime de calibration de pd_ml).
PROFILE_SPEC = {
    "pd_ml": ("absolu", lambda ref: (1.5 * ref, 1.0 * ref),
              "1,5 x puis 1 x le taux de défaut de référence (validation 2013-2014)"),
    "van": ("mixte", None, "b1 = 0 $ (le projet crée de la valeur) ; b2 = 66e centile du lot"),
    "delai_recuperation": ("mixte", None, "b1 = 10 ans (horizon explicite de la simulation) ; b2 = 33e centile du lot"),
    "dscr_min": ("absolu", lambda ref: (1.0, 1.25), "1,0 (flux = service de la dette) et 1,25 (repère du guide)"),
    "dette_fonds_propres": ("absolu", lambda ref: (3.0, 1.5),
                            "3,0 (75 % de dette) et 1,5 (60 % de dette) : hypothèse d'équipe"),
    "couverture_garanties": ("absolu", lambda ref: (0.5, 1.0),
                             "les garanties couvrent la moitié (0,5) puis la totalité (1,0) du prêt"),
    "esg_score": ("absolu", lambda ref: (50.0, 70.0), "50 et 70 : entre les centres des classes A/B/C (30/60/85)"),
    "emplois_par_million": ("relatif", None, "33e et 66e centiles du lot (aucun repère externe)"),
}


def make_profiles(df: pd.DataFrame, reference_rate: float) -> tuple[np.ndarray, pd.DataFrame]:
    """Renvoie (profils orientés 2 x 8, table lisible des profils en unités d'origine)."""
    def q(c, qt):
        return float(np.quantile(df[c].to_numpy(float), qt))

    raw = {}
    for c in CRITERIA:
        kind, fn, _ = PROFILE_SPEC[c]
        if kind == "absolu":
            raw[c] = fn(reference_rate)
        elif c == "van":
            raw[c] = (0.0, max(0.0, q(c, 2 / 3)))
        elif c == "delai_recuperation":
            raw[c] = (10.0, min(10.0, q(c, 1 / 3)))
        else:
            raw[c] = (q(c, 1 / 3), q(c, 2 / 3))
    prof = np.array([[SENSE[c] * raw[c][k] for c in CRITERIA] for k in (0, 1)])
    prof[1] = np.maximum(prof[1], prof[0])            # b2 doit dominer b1 (sens « plus grand = meilleur »)
    table = pd.DataFrame({"critere": CRITERIA, "type": [PROFILE_SPEC[c][0] for c in CRITERIA],
                          "b1_refuser_conditionner": [SENSE[c] * prof[0, j] for j, c in enumerate(CRITERIA)],
                          "b2_conditionner_financer": [SENSE[c] * prof[1, j] for j, c in enumerate(CRITERIA)],
                          "justification": [PROFILE_SPEC[c][2] for c in CRITERIA]})
    return prof, table


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


def run_mcda(scores: pd.DataFrame, weights: pd.Series, reference_rate: float,
             cfg: McdaConfig = McdaConfig(), filter_first: bool = True) -> pd.DataFrame:
    """Applique filtre -> ELECTRE III + ELECTRE TRI + PROMETHEE II sur les survivants."""
    df = eliminatory_filter(scores, reference_rate) if filter_first else scores.assign(ecarte=False, motif_ecarte="")
    alive = df[~df["ecarte"]]
    w = weights.reindex(CRITERIA).to_numpy(float)
    perf = oriented(alive)
    th = make_thresholds(perf, cfg.q_pct, cfg.p_pct, cfg.v_pct)
    e3 = electre_iii(perf, w, th)
    tri = electre_tri(perf, make_profiles(alive, reference_rate)[0], w, th, cfg.cut)
    pm = promethee_ii(perf, w, th)
    out = df.copy()
    for col in ["rang_electre3", "rang_distillation_descendante", "rang_distillation_ascendante",
                "rang_promethee2", "flux_net_promethee", "credibilite_nette"]:
        out[col] = np.nan
    out["categorie_tri_pessimiste"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    out["categorie_tri_optimiste"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    out.loc[alive.index, "rang_electre3"] = e3["rank"]
    out.loc[alive.index, "rang_distillation_descendante"] = e3["rank_desc"]
    out.loc[alive.index, "rang_distillation_ascendante"] = e3["rank_asc"]
    out.loc[alive.index, "credibilite_nette"] = e3["net_credibility"]
    out.loc[alive.index, "categorie_tri_pessimiste"] = [CATEGORIES[i] for i in tri["pessimiste"]]
    out.loc[alive.index, "categorie_tri_optimiste"] = [CATEGORIES[i] for i in tri["optimiste"]]
    out.loc[alive.index, "rang_promethee2"] = pm["rank"]
    out.loc[alive.index, "flux_net_promethee"] = pm["phi"]
    out.loc[out["ecarte"], "categorie_tri_pessimiste"] = "Refuser"
    out.loc[out["ecarte"], "categorie_tri_optimiste"] = "Refuser"
    return out


# ----------------------------------------------------------------------------- sensibilité
LAMBDA_RANGE = (0.5, 0.8)   # coupure plausible pour les scénarios combinés ; 0,9-1 (quasi-unanimité) n'est
                            # montré que dans le tableau « un paramètre à la fois », comme cas extrême


def sensitivity(scores: pd.DataFrame, weights: pd.Series, reference_rate: float, seed: int = 7,
                n_weight_draws: int = 200) -> dict[str, pd.DataFrame]:
    base = run_mcda(scores, weights, reference_rate)
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
                     **compare(run_mcda(scores, weights, reference_rate, McdaConfig(q, p, v, 0.6)))})
    # (b) niveau de coupure lambda
    for cut in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        rows.append({"parametre": "coupure lambda", "valeur": f"{cut:g}",
                     **compare(run_mcda(scores, weights, reference_rate, McdaConfig(cut=cut)))})
    table = pd.DataFrame(rows)

    # (c) scénarios combinés : poids +/-30 %, seuils q/p/v et coupure lambda dans [0,5 ; 0,8]
    rng = np.random.default_rng(seed)
    cats, ranks = [], []
    for _ in range(n_weight_draws):
        w = weights * rng.uniform(0.7, 1.3, len(weights))
        cfg = McdaConfig(q_pct=rng.uniform(0.03, 0.10), p_pct=rng.uniform(0.15, 0.30),
                         v_pct=rng.uniform(0.40, 0.80), cut=rng.uniform(*LAMBDA_RANGE))
        res = run_mcda(scores, w / w.sum(), reference_rate, cfg)
        cats.append(res["categorie_tri_pessimiste"].to_numpy())
        ranks.append(res["rang_electre3"].to_numpy())
    cats, ranks = np.array(cats), np.array(ranks, dtype=float)
    ranked = ~np.isnan(ranks).all(axis=0)              # projets écartés : jamais classés -> rang NaN
    rmean, rstd = np.full(ranks.shape[1], np.nan), np.full(ranks.shape[1], np.nan)
    rmean[ranked], rstd[ranked] = np.nanmean(ranks[:, ranked], axis=0), np.nanstd(ranks[:, ranked], axis=0)
    frag = pd.DataFrame({
        "part_scenarios_categorie_differente": (cats != base_cat.to_numpy()).mean(axis=0),
        "rang_moyen": rmean, "rang_ecart_type": rstd,
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
