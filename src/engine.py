"""Moteur réutilisable : évaluer N'IMPORTE QUEL jeu de projets avec une configuration explicite.

Une configuration (dict sérialisable en JSON) décrit tout ce qui est réglable :
  * les critères : colonne, libellé, sens (max / min), poids, format d'affichage, profils ELECTRE TRI b1 / b2 ;
  * le mode de pondération : « manuel » (poids donnés) ou « ahp » (jugements deux à deux, ratio de cohérence) ;
  * le filtre éliminatoire : liste de règles « critère opérateur seuil » ;
  * les seuils ELECTRE (q, p, v en % de l'étendue) et la coupure λ.

Utilisé par l'application (onglet « Configurer ») et en ligne de commande (src/evaluer.py).
Le jeu de démonstration (40 prêts SBA) et sa configuration sont rechargés depuis les fichiers du pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import arbitrage
import mcda

ROOT = Path(__file__).resolve().parents[1]
D, P = ROOT / "data/processed", ROOT / "params"

FORMATS = {"pct": "pourcentage (0,05 -> 5,0 %)", "money": "montant ($)", "dec2": "2 décimales",
           "dec1": "1 décimale", "int": "entier"}
OPERATEURS = list(mcda.OPS)
# Champs bruts d'un prêt SBA nécessaires pour calculer pd_ml avec le modèle entraîné (features.make_features)
MODEL_COLUMNS = ["GrossApproval", "SBAGuaranteedApproval", "TermInMonths", "InitialInterestRate",
                 "FixedorVariableInterestInd", "RevolverStatus", "NaicsCode", "BorrState", "BusinessType",
                 "BusinessAge", "FranchiseCode", "ProcessingMethod"]
DEMO_LABELS = {"pd_ml": ("Probabilité de défaut", "pct"), "van": ("VAN", "money"),
               "delai_recuperation": ("Délai de récupération (ans)", "dec1"), "dscr_min": ("DSCR minimum", "dec2"),
               "dette_fonds_propres": ("Dette / fonds propres", "dec2"),
               "couverture_garanties": ("Couverture par les garanties", "dec2"), "esg_score": ("Score ESG", "int"),
               "emplois_par_million": ("Emplois par million $", "dec1")}


# ----------------------------------------------------------------------------------------------- formats
def fmt(v, kind: str = "dec2") -> str:
    """Affichage d'une valeur (les mêmes chaînes servent de « faits » à l'agent et au conseiller)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n.d."
    v = float(v)
    if kind == "pct":
        return f"{100 * v:.1f} %"
    if kind == "money":
        return f"{v:,.0f} $".replace(",", " ")
    if kind == "int":
        return f"{v:.0f}"
    if kind == "dec1":
        return f"{v:.1f}"
    return f"{v:.2f}"


# ----------------------------------------------------------------------------------------------- démonstration
def demo_projects() -> pd.DataFrame:
    """Les 40 projets de démonstration : identifiant, nom lisible, 8 critères, informations descriptives."""
    sim = pd.read_csv(D / "simulation_detail.csv")
    expl = pd.read_csv(D / "explications_projets.csv")[["loan_id", "facteurs_risque"]]
    df = sim.merge(expl, on="loan_id", how="left").rename(columns={"loan_id": "id", "nom_projet": "nom",
                                                                   "secteur_damodaran": "secteur"})
    seed = D / "sensibilite_graine_projets.csv"
    if seed.exists():
        df = df.merge(pd.read_csv(seed)[["loan_id", "part_graines_meme_categorie"]].rename(columns={"loan_id": "id"}),
                      on="id", how="left")
    return df


def demo_config() -> dict:
    """Configuration qui reproduit exactement les résultats du pipeline (src/run_mcda.py)."""
    w = pd.read_csv(P / "poids_ahp.csv").set_index("critere")["poids_ahp"]
    prof = pd.read_csv(P / "profils_electre_tri.csv").set_index("critere")
    filt = json.loads((P / "filtre_eliminatoire.json").read_text(encoding="utf-8"))
    crit = [{"colonne": c, "libelle": DEMO_LABELS[c][0], "sens": "max" if mcda.SENSE[c] > 0 else "min",
             "poids": round(float(w[c]), 4), "format": DEMO_LABELS[c][1],
             "b1": float(prof.loc[c, "b1_refuser_conditionner"]), "b2": float(prof.loc[c, "b2_conditionner_financer"])}
            for c in mcda.CRITERIA]
    return {"nom": "Démonstration : 40 prêts SBA 7(a), indicateurs simulés",
            "criteres": crit, "mode_poids": "ahp",
            "ahp_jugements": pd.read_csv(P / "ahp_jugements.csv", dtype=str).to_dict("records"),
            "filtre": mcda.default_rules(filt["taux_defaut_reference"], filt["dscr_min_plancher"], filt["multiple"]),
            "seuils": {"q": 0.05, "p": 0.20, "v": 0.50}, "coupure": 0.6,
            "colonnes_info": ["secteur", "etat", "montant_pret", "investissement", "duree_standard_annees"],
            "arbitrage": dict(arbitrage.DEFAULTS), "devise": "$"}


# ----------------------------------------------------------------------------------------------- nouveau jeu
INFO_PREFIXES = ("montant", "investissement", "duree_standard", "annee", "naics")   # descriptives, pas des critères
MIN_HINTS = ("pd", "risque", "defaut", "défaut", "delai", "délai", "cout", "coût", "dette", "emission", "émission",
             "perte", "duree", "durée", "volatil")


def guess_config(df: pd.DataFrame, name: str = "Mon analyse") -> dict:
    """Configuration de départ pour un jeu importé : toutes les colonnes numériques deviennent des critères
    (sens deviné d'après le nom, poids égaux, profils = tertiles), aucun filtre. Tout est modifiable ensuite."""
    skip = {"id", "nom", "part_graines_meme_categorie"}
    num = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in num if c not in MODEL_COLUMNS and not c.startswith(INFO_PREFIXES)]
    sense = {c: -1 if any(h in c.lower() for h in MIN_HINTS) else 1 for c in num}
    prof = mcda.tertile_profiles(df, sense) if len(df) else {}
    crit = [{"colonne": c, "libelle": DEMO_LABELS.get(c, (c.replace("_", " "), "dec2"))[0],
             "sens": "max" if sense[c] > 0 else "min", "poids": round(1 / max(len(num), 1), 4),
             "format": DEMO_LABELS.get(c, ("", "dec2"))[1], "b1": prof[c][0], "b2": prof[c][1]} for c in num]
    info = [c for c in df.columns if c not in num and c not in {"id", "nom"} and c not in MODEL_COLUMNS][:4]
    cfg = {"nom": name, "criteres": crit, "mode_poids": "manuel", "ahp_jugements": [], "filtre": [],
           "seuils": {"q": 0.05, "p": 0.20, "v": 0.50}, "coupure": 0.6, "colonnes_info": info}
    money = next((c for c in df.columns if str(c).startswith("montant") and pd.api.types.is_numeric_dtype(df[c])), None)
    if money:                      # budget proposé : la moitié des montants demandés, à ajuster
        tot = float(pd.to_numeric(df[money], errors="coerce").sum())
        cfg["arbitrage"] = {**arbitrage.DEFAULTS, "colonne_montant": money, "budget": round(tot / 2, -3) or tot,
                            "montant_min": 0.0, "montant_max": float(pd.to_numeric(df[money], errors="coerce").max()),
                            "part_max_secteur": 0.4 if "secteur" in df.columns else 1.0}
    return cfg


def prepare_projects(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise un fichier importé : colonnes id / nom créées si absentes, pd_ml calculée si possible."""
    from naming import human_names
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    # Nombres écrits à la française (« 1 234,5 », « 12,3 % ») : convertis si la colonne est très majoritairement
    # numérique ; les quelques valeurs illisibles (« n.c. ») deviennent manquantes et sont signalées par validate().
    for c in out.columns:
        if c in ("id", "nom") or c in MODEL_COLUMNS or pd.api.types.is_numeric_dtype(out[c]):
            continue
        txt = out[c].astype("string").str.strip()
        conv = pd.to_numeric(txt.str.replace(r"[\s  %]", "", regex=True).str.replace(",", ".", regex=False),
                             errors="coerce")
        filled = txt.notna() & (txt != "")
        if filled.sum() and conv[filled].notna().mean() >= 0.75:
            out[c] = conv.astype(float)
    if "id" not in out.columns:
        out.insert(0, "id", [f"P{i + 1:03d}" for i in range(len(out))])
    out["id"] = out["id"].astype(str)
    if "nom" not in out.columns:
        out.insert(1, "nom", human_names(out["id"], out["NaicsCode"] if "NaicsCode" in out else None).values)
    out["nom"] = out["nom"].astype(str)
    if "pd_ml" not in out.columns and all(c in out.columns for c in MODEL_COLUMNS):
        out["pd_ml"] = score_pd(out)
    return out


def score_pd(df: pd.DataFrame) -> np.ndarray:
    """pd_ml calibrée du modèle retenu pour des prêts décrits par les champs SBA bruts (MODEL_COLUMNS)."""
    import joblib
    from features import make_features
    art = joblib.load(ROOT / "models/risk_models.joblib")
    name = art["chosen"]
    X = make_features(df.assign(NaicsCode=df["NaicsCode"].astype(str), BorrState=df["BorrState"].astype(str)))
    return art["calibrators"][name].predict(art[name].predict_proba(X)[:, 1])


def template_projects() -> pd.DataFrame:
    """Modèle de fichier à remplir (3 projets fictifs). Les colonnes de critères sont libres."""
    return pd.DataFrame({
        "id": ["P001", "P002", "P003"], "nom": ["Boulangerie Martin", "Garage Lefèvre", "Studio Web Diallo"],
        "secteur": ["Commerce", "Services", "Numérique"],
        "pd_ml": [0.025, 0.061, 0.034], "van": [180000, 45000, 320000], "delai_recuperation": [4.5, 7.2, 3.8],
        "dscr_min": [1.45, 1.08, 1.62], "dette_fonds_propres": [1.8, 3.2, 1.2],
        "couverture_garanties": [0.9, 0.6, 0.4], "esg_score": [62, 48, 75], "emplois_par_million": [22.0, 14.5, 30.0]})


# ----------------------------------------------------------------------------------------------- validation
def validate(df: pd.DataFrame, config: dict) -> list[str]:
    """Liste des problèmes bloquants (vide = on peut évaluer)."""
    errs = []
    crit = config.get("criteres", [])
    if len(crit) < 2:
        errs.append("Il faut au moins 2 critères.")
    if len(df) < 2:
        errs.append("Il faut au moins 2 projets.")
    cols = [c["colonne"] for c in crit]
    if len(set(cols)) != len(cols):
        errs.append("Un même critère est déclaré deux fois.")
    for c in crit:
        col = c["colonne"]
        if col not in df.columns:
            errs.append(f"Colonne absente du fichier : « {col} ».")
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        if v.isna().any():
            errs.append(f"« {col} » contient {int(v.isna().sum())} valeur(s) manquante(s) ou non numérique(s).")
        if c.get("sens") not in ("max", "min"):
            errs.append(f"Sens invalide pour « {col} » (max ou min).")
    for r in config.get("filtre", []):
        if r.get("critere") not in df.columns:
            errs.append(f"Règle de filtre sur une colonne absente : « {r.get('critere')} ».")
        if r.get("operateur") not in mcda.OPS:
            errs.append(f"Opérateur de filtre invalide : « {r.get('operateur')} ».")
    if config.get("mode_poids") == "manuel" and sum(float(c.get("poids", 0)) for c in crit) <= 0:
        errs.append("La somme des poids doit être positive.")
    if config.get("mode_poids") == "ahp":
        n = len(cols)
        pairs = {frozenset((j["critere_a"], j["critere_b"])) for j in config.get("ahp_jugements", [])
                 if j["critere_a"] in cols and j["critere_b"] in cols}
        if len(pairs) != n * (n - 1) // 2:
            errs.append(f"AHP : {len(pairs)} jugement(s) sur {n * (n - 1) // 2} nécessaires pour {n} critères.")
    arb = config.get("arbitrage")
    if arb:
        if arb.get("colonne_montant", "montant_pret") not in df.columns:
            errs.append(f"Arbitrage : colonne de montant « {arb.get('colonne_montant', 'montant_pret')} » absente.")
        if float(arb.get("budget", 0)) <= 0:
            errs.append("Arbitrage : le budget doit être positif.")
        if float(arb.get("montant_min", 0)) > float(arb.get("montant_max", float("inf"))):
            errs.append("Arbitrage : le montant minimum dépasse le maximum.")
        if not 0 < float(arb.get("part_max_secteur", 1)) <= 1:
            errs.append("Arbitrage : la part maximale par secteur doit être entre 0 et 100 %.")
    s = config.get("seuils", {})
    if not (0 <= s.get("q", 0) <= s.get("p", 0) <= s.get("v", 0)):
        errs.append("Seuils ELECTRE : il faut q ≤ p ≤ v.")
    if not 0.5 <= float(config.get("coupure", 0.6)) <= 1:
        errs.append("La coupure λ doit être entre 0,5 et 1.")
    return errs


# ----------------------------------------------------------------------------------------------- évaluation
def weights_from_config(config: dict) -> tuple[pd.Series, float | None]:
    cols = [c["colonne"] for c in config["criteres"]]
    if config.get("mode_poids") == "ahp":
        table = pd.DataFrame([j for j in config["ahp_jugements"] if j["critere_a"] in cols and j["critere_b"] in cols])
        w, _, cr = mcda.ahp_weights(mcda.ahp_matrix_from_judgments(table, cols))
        return w, cr
    w = pd.Series({c["colonne"]: float(c["poids"]) for c in config["criteres"]})
    return w / w.sum(), None


def evaluate(df: pd.DataFrame, config: dict, sensitivity: bool = True, n_scenarios: int = 200) -> dict:
    """Évalue un jeu de projets. Renvoie résultats, poids, ratio de cohérence, profils et sensibilités."""
    errs = validate(df, config)
    if errs:
        raise ValueError(" ; ".join(errs))
    crit = config["criteres"]
    sense = {c["colonne"]: 1 if c["sens"] == "max" else -1 for c in crit}
    profiles = {c["colonne"]: (float(c["b1"]), float(c["b2"])) for c in crit}
    s = config["seuils"]
    cfg = mcda.McdaConfig(s["q"], s["p"], s["v"], float(config["coupure"]))
    w, cr = weights_from_config(config)
    scores = df.copy()
    for c in sense:
        scores[c] = pd.to_numeric(scores[c])
    spec = {"sense": sense, "profiles": profiles, "rules": config.get("filtre", [])}
    res = mcda.run_mcda(scores, w, None, cfg, **spec)
    out = {"weights": w, "cr": cr, "config": config, "sensibilite": None}
    if sensitivity and (~res["ecarte"]).sum() >= 3:
        sens = mcda.sensitivity(scores, w, None, n_weight_draws=n_scenarios, cfg=cfg, **spec)
        res = res.join(sens["fragilite_poids"])
        out["sensibilite"] = sens
    res["categorie_finale"] = res["categorie_tri_pessimiste"]
    alive = int((~res["ecarte"]).sum())
    ecart = (res["rang_distillation_descendante"] - res["rang_distillation_ascendante"]).abs()
    res["ecart_distillations"] = ecart
    res["position_ambigue"] = ecart >= 0.2 * max(alive, 1)
    if "decision_fragile" in res:
        res["fragilite_texte"] = np.where(res["decision_fragile"],
                                          "fragile (catégorie différente dans au moins 25 % des scénarios)",
                                          "stable (catégorie inchangée dans plus de 75 % des scénarios)")
    else:
        res["fragilite_texte"] = "non évaluée"
    out["arbitrage"] = None
    if config.get("arbitrage"):                     # agent d'arbitrage : sélection sous budget
        res, out["arbitrage"] = arbitrage.run(res, config["arbitrage"])
    out["results"] = res
    return out


def config_to_json(config: dict) -> str:
    return json.dumps(config, indent=2, ensure_ascii=False, default=float)


def config_from_json(text: str) -> dict:
    cfg = json.loads(text)
    for k in ("criteres", "seuils", "coupure"):
        if k not in cfg:
            raise ValueError(f"configuration incomplète : clé « {k} » absente")
    cfg.setdefault("filtre", []); cfg.setdefault("mode_poids", "manuel"); cfg.setdefault("ahp_jugements", [])
    cfg.setdefault("colonnes_info", []); cfg.setdefault("nom", "Analyse importée"); cfg.setdefault("devise", "")
    return cfg
