"""Agent d'arbitrage (guide, composante 6) : sélectionner les projets à financer sous contrainte de budget.

Le MCDA donne une catégorie à chaque projet ; il ne tient pas compte de l'argent disponible. L'arbitrage choisit,
parmi les projets « Financer » (puis, si on l'autorise, « Conditionner »), ceux qui entrent dans le budget :

  * budget global alloué sur la période ;
  * montant minimum et maximum par projet (hors bornes = non éligible) ;
  * diversification : la part d'un secteur dans le budget ne dépasse pas un plafond.

Deux règles de sélection (paramètre « mode ») :

  * « classement » (par défaut) : on suit l'ordre de priorité du comité — les « Financer » par rang ELECTRE III,
    puis les « Conditionner » — et on retient chaque projet s'il respecte encore le budget, ses bornes et le
    plafond de son secteur. C'est l'optimum EXACT quand tout projet mieux classé doit passer avant tous les
    projets moins bien classés (poids lexicographiques) ; chaque refus a un motif précis.
  * « nombre » : programmation linéaire en nombres entiers (sac à dos multi-contraintes, scipy.optimize.milp /
    HiGHS) qui maximise le nombre de « Financer », puis de « Conditionner », puis la qualité (flux PROMETHEE II).
    Utile pour comparer : elle peut écarter un projet très bien classé mais coûteux au profit de plusieurs petits.

Déterministe, aucun LLM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

DEFAULTS = {"budget": 5_000_000.0, "montant_min": 20_000.0, "montant_max": 2_000_000.0,
            "part_max_secteur": 0.35, "inclure_conditionner": True, "mode": "classement",
            "colonne_montant": "montant_pret", "colonne_secteur": "secteur"}


def params_with_defaults(p: dict | None) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (p or {}).items() if v is not None})
    return out


def run(results: pd.DataFrame, params: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Renvoie (résultats + colonnes selection_budget / motif_arbitrage, résumé)."""
    p = params_with_defaults(params)
    col_m, col_s = p["colonne_montant"], p["colonne_secteur"]
    res = results.copy()
    if col_m not in res:
        res["selection_budget"] = False
        res["motif_arbitrage"] = "pas de colonne de montant : arbitrage impossible"
        return res, {"actif": False, "raison": f"colonne « {col_m} » absente"}
    amount = pd.to_numeric(res[col_m], errors="coerce").fillna(np.inf).to_numpy(float)
    cat = res["categorie_finale"].to_numpy()
    sector = res[col_s].astype(str).to_numpy() if col_s in res else np.array(["(tous)"] * len(res))
    budget = float(p["budget"])

    in_bounds = (amount >= float(p["montant_min"])) & (amount <= float(p["montant_max"]))
    eligible_cat = (cat == "Financer") | ((cat == "Conditionner") & bool(p["inclure_conditionner"]))
    eligible = in_bounds & eligible_cat & np.isfinite(amount)

    # qualité dans [0, 1] : flux net PROMETHEE II (sinon rang ELECTRE III), 0 pour les écartés
    if "flux_net_promethee" in res and res["flux_net_promethee"].notna().any():
        q = res["flux_net_promethee"].astype(float)
    else:
        q = -res["rang_electre3"].astype(float)
    q = ((q - q.min()) / ((q.max() - q.min()) or 1)).fillna(0).to_numpy(float)
    n = len(res)
    n_cond = int(((cat == "Conditionner") & eligible).sum())
    w_c = n + 1                                   # un « Conditionner » vaut plus que toute différence de qualité
    w_f = w_c * (n_cond + 1) + n + 1              # un « Financer » vaut plus que tous les « Conditionner »
    value = np.where(cat == "Financer", w_f, np.where(cat == "Conditionner", w_c, 0)) + q

    sel = np.zeros(n, dtype=bool)
    sectors = sorted(set(sector))
    cap = float(p["part_max_secteur"]) * budget
    use_cap = col_s in res and float(p["part_max_secteur"]) < 1
    motifs = [""] * n
    for i in range(n):                                    # motifs qui ne dépendent pas de la sélection
        if cat[i] == "Refuser":
            motifs[i] = "refusé par le MCDA"
        elif cat[i] == "Conditionner" and not p["inclure_conditionner"]:
            motifs[i] = "sous conditions : non retenu (option désactivée)"
        elif not in_bounds[i]:
            motifs[i] = f"montant hors bornes ({p['montant_min']:,.0f} à {p['montant_max']:,.0f})".replace(",", " ")

    if p["mode"] == "nombre" and eligible.any():
        rows, ub = [np.where(eligible, amount, 0.0)], [budget]           # budget global
        if use_cap:
            for s_ in sectors:                                             # plafond par secteur
                rows.append(np.where(eligible & (sector == s_), amount, 0.0))
                ub.append(cap)
        sol = milp(c=-np.where(eligible, value, 0.0), integrality=np.ones(n),
                   bounds=Bounds(0, eligible.astype(float)),
                   constraints=LinearConstraint(np.vstack(rows), -np.inf, np.array(ub)))
        if sol.status == 0 and sol.x is not None:
            sel = sol.x > 0.5
        used_s = {s_: float(amount[sel & (sector == s_)].sum()) for s_ in sectors}
        used = float(amount[sel].sum())
        for i in np.where(eligible & ~sel)[0]:
            if use_cap and used_s[sector[i]] + amount[i] > cap + 1e-6:
                motifs[i] = "plafond du secteur atteint"
            elif used + amount[i] > budget + 1e-6:
                motifs[i] = "budget insuffisant"
            else:
                motifs[i] = "non retenu par l'optimisation"
    else:                                                 # « classement » : ordre de priorité du comité
        rank = res["rang_electre3"].astype(float).fillna(np.inf).to_numpy()
        prio = np.where(cat == "Financer", 0, 1)
        used, used_s = 0.0, {s_: 0.0 for s_ in sectors}
        for i in sorted(np.where(eligible)[0], key=lambda k: (prio[k], rank[k], -q[k])):
            if used + amount[i] > budget + 1e-6:
                motifs[i] = "budget insuffisant"
            elif use_cap and used_s[sector[i]] + amount[i] > cap + 1e-6:
                motifs[i] = "plafond du secteur atteint"
            else:
                sel[i] = True
                used += amount[i]
                used_s[sector[i]] += amount[i]
    for i in np.where(sel)[0]:
        motifs[i] = "sélectionné" + (" (sous conditions)" if cat[i] == "Conditionner" else "")

    used = float(amount[sel].sum()) if sel.any() else 0.0
    by_sector = {s_: float(amount[sel & (sector == s_)].sum()) for s_ in sectors}
    by_sector = {k: v for k, v in by_sector.items() if v > 0}
    res["selection_budget"] = sel
    res["motif_arbitrage"] = motifs
    summary = {
        "actif": True, "parametres": {k: p[k] for k in ("budget", "montant_min", "montant_max",
                                                       "part_max_secteur", "inclure_conditionner", "mode")},
        "budget": budget, "montant_engage": used, "budget_restant": budget - used,
        "taux_utilisation": used / budget if budget else 0.0,
        "projets_selectionnes": int(sel.sum()),
        "dont_financer": int((sel & (cat == "Financer")).sum()),
        "dont_conditionner": int((sel & (cat == "Conditionner")).sum()),
        "financer_non_retenus": int(((cat == "Financer") & ~sel).sum()),
        "montant_par_secteur": by_sector, "secteurs_finances": len(by_sector),
    }
    return res, summary
