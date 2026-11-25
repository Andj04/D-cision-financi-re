"""Conseiller financier conversationnel : répond aux questions sur le jeu de projets, les résultats et la méthode.

Principe (le LLM ne calcule jamais) : le LLM ne voit pas les données brutes. Il appelle des OUTILS Python
déterministes (fiche d'un projet, liste triée, statistiques, comparaison, scénario de poids, paramètres de la
méthode, données historiques SBA) et reformule leurs sorties. Un vérificateur signale tout nombre de la réponse
qui n'apparaît dans aucune sortie d'outil.

Fournisseurs gratuits : Gemini (appel de fonctions natif) ou Groq (format OpenAI). Sans clé : les mêmes outils
restent accessibles par des « questions rapides » dans l'application (sans LLM).
"""
from __future__ import annotations

import difflib
import json
import re
import urllib.error

import numpy as np
import pandas as pd

import agent
import engine
import mcda

SYSTEM = (
    "Tu es le conseiller financier virtuel d'un comité d'investissement bancaire qui évalue des demandes de crédit "
    "de PME. Tu réponds en français, de façon claire, structurée et pédagogique (listes courtes, phrases simples).\n"
    "RÈGLES STRICTES :\n"
    "1. Tout chiffre que tu cites doit venir d'un outil appelé dans cette conversation. Tu n'effectues AUCUN calcul "
    "toi-même (pas de moyenne, somme, différence, pourcentage, arrondi) : si un calcul est nécessaire, appelle l'outil "
    "« statistiques », « simuler_scenario » ou « comparer_projets ».\n"
    "2. Avant de répondre sur un projet, appelle « fiche_projet » ; sur l'ensemble des résultats, « resume_resultats ».\n"
    "3. Désigne les projets par leur nom. Si une information n'est disponible dans aucun outil, dis-le simplement.\n"
    "4. Signale la fragilité d'une décision quand elle est pertinente. Dans la démonstration, seuls les indicateurs "
    "financiers (VAN, délai, DSCR, dette, garanties, ESG) sont simulés ; la probabilité de défaut vient d'un modèle "
    "entraîné sur des prêts réels, et les données historiques SBA sont réelles.\n"
    "7. Pas de formule LaTeX ni de symbole « $ » autour des mots ; écris les montants en dollars en toutes lettres "
    "ou avec « $ » après le nombre.\n"
    "5. Tu conseilles, tu ne décides pas : la décision finale revient au comité. Aucune promesse de rendement.\n"
    "6. Pour expliquer la méthode (AHP, ELECTRE, PROMETHEE, filtre, modèle de risque), appelle « parametres_methode ».\n"
    "8. Les SEULES méthodes utilisées sont : AHP (pondération), ELECTRE III (classement), ELECTRE TRI avec la règle "
    "d'affectation pessimiste (catégories) et PROMETHEE II (comparaison) ; modèle de risque : régression logistique et "
    "EBM (Explainable Boosting Machine). N'emploie aucun autre nom de méthode (pas d'ELECTRE Tri-C, IV, IS, TOPSIS...).\n"
    "9. Si l'on te demande ce que tu sais faire ou ce que tu ne peux pas faire, appelle « capacites » et décris "
    "exactement ce qu'il renvoie, limites comprises, sans rien ajouter."
)

ALLOWED_METHODS = ["AHP", "ELECTRE III", "ELECTRE TRI", "PROMETHEE II", "EBM", "régression logistique"]
# noms de méthodes que le conseiller ne doit pas attribuer au projet (signalés s'ils apparaissent)
_FOREIGN = re.compile(                                  # sensible à la casse : « ELECTRE TRI c'est » n'est pas « Tri-C »
    r"\bELECTRE[\s-]*(?:(?:TRI|Tri)[\s-]?n?C\b|(?:TRI|Tri)[\s-]?B\b|IV\b|IS\b|II\b|I\b)"
    r"|\bPROMETHEE[\s-]*(?:I|III|IV|V|VI)\b|\b(?:TOPSIS|VIKOR|MACBETH|SMARTER|BWM|WSM|WPM)\b|\b[Bb]est[- ][Ww]orst\b")

CAPABILITIES = {
    "ce_que_je_fais": [
        "Vue d'ensemble de l'analyse : nombre de projets, catégories, exclusions et motifs, poids, meilleurs projets, fragilité.",
        "Liste de projets filtrée (catégorie, texte) et triée par critère ou par rang ELECTRE III.",
        "Fiche d'un projet : critères, catégorie, rangs, position par rapport aux seuils b1/b2, fragilité, facteurs de risque.",
        "Comparaison de 2 à 6 projets critère par critère.",
        "Statistiques d'un critère (moyenne, médiane, min, max), globales ou par catégorie, secteur ou État si ces colonnes existent.",
        "Scénarios « et si » : changer un ou plusieurs poids à la fois, la coupure λ, ou désactiver le filtre, puis voir "
        "quels projets changent de catégorie (simulation temporaire, l'analyse affichée ne change pas).",
        "Explication de la méthode avec les paramètres réels : AHP, filtre, profils, ELECTRE III, ELECTRE TRI (règle "
        "pessimiste), PROMETHEE II, sensibilité.",
        "Démonstration uniquement : données historiques SBA (taux de défaut par période, audit, métriques et variables "
        "du modèle de risque, erreurs par État, secteur ou procédure)."],
    "ce_que_je_ne_fais_pas": [
        "Je ne calcule rien moi-même : chaque chiffre vient d'un outil.",
        "Je ne modifie pas l'analyse affichée (pour cela : page « Configurer l'analyse »).",
        "Je n'ajoute pas de projet et je ne consulte pas un prêt SBA individuel.",
        "Je n'ai pas accès à Internet ni à des informations sur les entreprises (les noms de la démonstration sont fictifs).",
        "Je ne décide pas : la décision revient au comité ; je ne promets aucun rendement."],
}

# ----------------------------------------------------------------------------------------------- espace de travail
class Workspace:
    """Ce que le conseiller peut consulter : résultats de l'évaluation active et sa configuration."""

    def __init__(self, evaluation: dict, demo: bool):
        self.res: pd.DataFrame = evaluation["results"].reset_index(drop=True)
        self.cfg: dict = evaluation["config"]
        self.w: pd.Series = evaluation["weights"]
        self.cr = evaluation.get("cr")
        self.sens = evaluation.get("sensibilite")
        self.demo = demo
        self.crit = self.cfg["criteres"]
        self.meta = {c["colonne"]: c for c in self.crit}

    # ---- utilitaires
    def f(self, col: str, v) -> str:
        return engine.fmt(v, self.meta[col].get("format", "dec2")) if col in self.meta else str(v)

    def find(self, name: str) -> pd.Series | None:
        names = self.res["nom"].astype(str)
        low = {n.lower(): i for i, n in enumerate(names)}
        q = str(name).strip().lower()
        if q in low:
            return self.res.iloc[low[q]]
        hit = [i for n, i in low.items() if q and (q in n or n in q)]
        if not hit:
            ids = self.res["id"].astype(str).str.lower()
            hit = list(np.where(ids == q)[0])
        if not hit:
            close = difflib.get_close_matches(q, list(low), n=1, cutoff=0.5)
            hit = [low[close[0]]] if close else []
        return self.res.iloc[hit[0]] if hit else None

    def short(self, r: pd.Series, cols: list[str] | None = None) -> dict:
        d = {"nom": r["nom"], "categorie": r["categorie_finale"],
             "rang_electre3": None if r["ecarte"] else int(r["rang_electre3"])}
        if r["ecarte"]:
            d["motif_exclusion"] = r["motif_ecarte"]
        for c in cols or [c["colonne"] for c in self.crit[:4]]:
            d[self.meta[c]["libelle"]] = self.f(c, r[c])
        return d

    # ---- outils
    def resume_resultats(self) -> dict:
        r = self.res
        out = {"jeu": self.cfg.get("nom"), "nombre_projets": len(r),
               "categories": r["categorie_finale"].value_counts().reindex(mcda.CATEGORIES[::-1], fill_value=0).to_dict(),
               "ecartes_par_le_filtre": int(r["ecarte"].sum()),
               "motifs_exclusion": r.loc[r["ecarte"], "motif_ecarte"].value_counts().to_dict(),
               "poids": {self.meta[c]["libelle"]: f"{100 * v:.1f} %" for c, v in self.w.items()},
               "top_5_electre3": r[~r["ecarte"]].sort_values("rang_electre3")["nom"].head(5).tolist()}
        if "decision_fragile" in r:
            out["decisions_fragiles"] = int(r["decision_fragile"].sum())
            out["fragiles_par_categorie"] = r[r["decision_fragile"]]["categorie_finale"].value_counts().to_dict()
        out["positions_ambigues_electre3"] = int(r["position_ambigue"].sum())
        if self.sens is not None:
            m = self.sens["comparaison_methodes"].iloc[0]
            out["correlation_electre3_promethee2"] = {"spearman": f"{m.spearman:.3f}", "kendall": f"{m.kendall:.3f}"}
        if self.demo and "part_graines_meme_categorie" in r:
            out["stabilite_moyenne_face_a_la_simulation"] = f"{100 * r['part_graines_meme_categorie'].mean():.0f} %"
            try:
                post = pd.read_csv(engine.D / "controle_a_posteriori.csv")
                out["controle_a_posteriori_defauts_reels"] = dict(zip(post.categorie_finale, post.defauts_reels.astype(int)))
            except FileNotFoundError:
                pass
        return out

    def lister_projets(self, categorie: str | None = None, tri: str | None = None, ordre: str = "decroissant",
                       limite: int = 10, inclure_ecartes: bool = True, texte: str | None = None) -> dict:
        r = self.res
        if categorie in mcda.CATEGORIES:
            r = r[r["categorie_finale"] == categorie]
        if not inclure_ecartes:
            r = r[~r["ecarte"]]
        if texte:
            mask = r["nom"].str.contains(texte, case=False, na=False)
            for col in ("secteur", "etat"):
                if col in r:
                    mask |= r[col].astype(str).str.contains(texte, case=False, na=False)
            r = r[mask]
        col = tri if tri in self.meta else None
        if col:
            r = r.sort_values(col, ascending=(ordre == "croissant"))
        else:
            r = r.sort_values(["ecarte", "rang_electre3"])
        cols = [col] + [c["colonne"] for c in self.crit[:3] if c["colonne"] != col] if col else None
        return {"nombre_total": len(r), "projets": [self.short(x, cols) for _, x in r.head(int(limite or 10)).iterrows()]}

    def fiche_projet(self, nom: str) -> dict:
        r = self.find(nom)
        if r is None:
            return {"erreur": f"aucun projet ne correspond à « {nom} »", "noms_disponibles": self.res["nom"].tolist()}
        facts = agent.build_facts(r.to_dict(), self.crit)
        pos = {}
        for c in self.crit:
            col, s = c["colonne"], (1 if c["sens"] == "max" else -1)
            v, b1, b2 = s * float(r[col]), s * float(c["b1"]), s * float(c["b2"])
            zone = "zone Financer" if v >= b2 else ("zone Conditionner" if v >= b1 else "zone Refuser")
            pos[c["libelle"]] = f"{self.f(col, r[col])} ({zone} ; seuils {self.f(col, c['b1'])} puis {self.f(col, c['b2'])})"
        out = {"faits": facts, "position_par_critere": pos,
               "categorie_optimiste_electre_tri": r.get("categorie_tri_optimiste")}
        if "part_scenarios_categorie_differente" in r and not pd.isna(r["part_scenarios_categorie_differente"]):
            out["part_scenarios_ou_la_categorie_change"] = f"{100 * r['part_scenarios_categorie_differente']:.0f} %"
        if r["position_ambigue"]:
            out["remarque_electre3"] = "position ambiguë : les deux distillations ELECTRE III classent ce projet très différemment"
        return out

    def comparer_projets(self, noms: list[str]) -> dict:
        rows, missing = [], []
        for n in noms[:6]:
            r = self.find(n)
            (missing.append(n) if r is None else rows.append(r))
        table = {}
        for c in self.crit:
            vals = {r["nom"]: self.f(c["colonne"], r[c["colonne"]]) for r in rows}
            if rows:
                s = 1 if c["sens"] == "max" else -1
                best = max(rows, key=lambda r: s * float(r[c["colonne"]]))
                vals["meilleur"] = best["nom"]
            table[c["libelle"]] = vals
        return {"comparaison": table, "categories": {r["nom"]: r["categorie_finale"] for r in rows},
                "rangs_electre3": {r["nom"]: (None if r["ecarte"] else int(r["rang_electre3"])) for r in rows},
                "non_trouves": missing}

    def statistiques(self, critere: str, groupe: str | None = None) -> dict:
        col = critere if critere in self.meta else next(
            (c["colonne"] for c in self.crit if c["libelle"].lower() == str(critere).lower()), None)
        if col is None:
            return {"erreur": f"critère inconnu « {critere} »", "criteres": {c["colonne"]: c["libelle"] for c in self.crit}}
        g = {"categorie": "categorie_finale", "secteur": "secteur", "etat": "etat"}.get(groupe or "", None)
        frames = self.res.groupby(g) if g and g in self.res else [("tous les projets", self.res)]
        out = {}
        for key, part in frames:
            v = part[col].astype(float)
            out[str(key)] = {"projets": len(v), "moyenne": self.f(col, v.mean()), "mediane": self.f(col, v.median()),
                             "minimum": self.f(col, v.min()), "maximum": self.f(col, v.max())}
        return {"critere": self.meta[col]["libelle"], "par": groupe or "aucun", "statistiques": out}

    def capacites(self) -> dict:
        return {**CAPABILITIES, "methodes_utilisees": ALLOWED_METHODS, "jeu_actif": self.cfg.get("nom")}

    def _col(self, critere: str) -> str | None:
        return critere if critere in self.meta else next(
            (c["colonne"] for c in self.crit if c["libelle"].lower() == str(critere).lower()), None)

    def simuler_scenario(self, critere: str | None = None, nouveau_poids: float | None = None,
                         poids: list[dict] | None = None, coupure: float | None = None,
                         sans_filtre: bool = False) -> dict:
        """poids : [{"critere": ..., "poids": 0.3}, ...] (un ou plusieurs) ; critere / nouveau_poids : forme courte."""
        cfg = json.loads(engine.config_to_json(self.cfg))
        w = self.w.copy()
        changes_req = list(poids or [])
        if critere:
            changes_req.append({"critere": critere, "poids": nouveau_poids})
        if changes_req:
            fixed = {}
            for ch in changes_req:
                col = self._col(ch.get("critere", ""))
                if col is None or ch.get("poids") is None:
                    return {"erreur": f"critère inconnu ou poids manquant : {ch}",
                            "criteres": {c["colonne"]: c["libelle"] for c in self.crit}}
                p = float(ch["poids"])
                fixed[col] = p / 100 if p > 1 else p                     # accepte 30 ou 0,30
            total = sum(fixed.values())
            others = [c for c in w.index if c not in fixed]
            if total > 1 + 1e-9:
                return {"erreur": f"la somme des poids imposés ({100 * total:.0f} %) dépasse 100 %"}
            for col, p in fixed.items():
                w[col] = p
            if others:
                rest = w[others].sum()
                w[others] = (w[others] / rest * (1 - total)) if rest > 0 else (1 - total) / len(others)
            w = w / w.sum()                                             # les autres gardent leurs proportions
        cfg["mode_poids"] = "manuel"
        for c in cfg["criteres"]:
            c["poids"] = float(w[c["colonne"]])
        if coupure is not None:
            cfg["coupure"] = float(coupure)
        if sans_filtre:
            cfg["filtre"] = []
        cols = list(dict.fromkeys(["id", "nom"] + [c["colonne"] for c in self.crit]
                                  + [r["critere"] for r in cfg.get("filtre", [])]))
        new = engine.evaluate(self.res[cols], cfg, sensitivity=False)["results"]
        before = self.res.set_index("id")["categorie_finale"]
        after = new.set_index("id")["categorie_finale"]
        names = self.res.set_index("id")["nom"]
        changes = [f"{names[i]} : {before[i]} -> {after[i]}" for i in before.index if before[i] != after[i]]
        cb = before.value_counts().reindex(mcda.CATEGORIES[::-1], fill_value=0)
        ca = after.value_counts().reindex(mcda.CATEGORIES[::-1], fill_value=0)
        return {"poids_utilises": {self.meta[c]["libelle"]: f"{100 * v:.1f} %" for c, v in w.items()},
                "coupure": cfg["coupure"], "filtre_actif": not sans_filtre,
                "categories_avant": cb.to_dict(), "categories_apres": ca.to_dict(),
                "evolution": {k: f"{int(ca[k] - cb[k]):+d}" for k in cb.index},
                "nombre_de_projets_qui_changent": len(changes), "projets_qui_changent": changes or ["aucun"]}

    def parametres_methode(self, sujet: str = "tout") -> dict:
        c = self.cfg
        txt = {
            "criteres": {x["libelle"]: {"colonne": x["colonne"], "sens": "à maximiser" if x["sens"] == "max" else "à minimiser",
                                        "poids": f"{100 * float(self.w[x['colonne']]):.1f} %"} for x in self.crit},
            "poids": {"mode": "AHP (comparaisons deux à deux, échelle de Saaty)" if c.get("mode_poids") == "ahp" else "poids saisis",
                      "ratio_de_coherence": None if self.cr is None else f"{self.cr:.4f}",
                      "explication": "AHP : on compare les critères deux à deux ; les poids sont le vecteur propre principal ; "
                                     "un ratio de cohérence inférieur à 0,10 indique des jugements cohérents."},
            "filtre": {"regles": [r.get("libelle") or f"{r['critere']} {r['operateur']} {r['seuil']}" for r in c.get("filtre", [])]
                       or ["aucune"], "explication": "Un projet qui déclenche une règle est écarté avant le classement (catégorie Refuser)."},
            "profils_electre_tri": {x["libelle"]: f"b1 (Refuser|Conditionner) = {self.f(x['colonne'], x['b1'])} ; "
                                                  f"b2 (Conditionner|Financer) = {self.f(x['colonne'], x['b2'])}" for x in self.crit},
            "seuils_et_coupure": {"q (indifférence)": f"{100 * c['seuils']['q']:.0f} % de l'étendue",
                                  "p (préférence)": f"{100 * c['seuils']['p']:.0f} % de l'étendue",
                                  "v (veto)": f"{100 * c['seuils']['v']:.0f} % de l'étendue", "coupure_lambda": c["coupure"]},
            "methodes": {"ELECTRE III": "méthode de surclassement : pour chaque paire, la crédibilité de « A est au moins aussi bon "
                                        "que B » combine concordance (poids des critères favorables) et veto (écart rédhibitoire) ; "
                                        "deux distillations donnent le classement.",
                         "ELECTRE TRI": "affecte chaque projet à Refuser / Conditionner / Financer en le comparant aux profils "
                                        "b1 et b2 avec la coupure λ (règle pessimiste retenue).",
                         "PROMETHEE II": "classement de comparaison par flux net de préférence ; sert à vérifier la robustesse.",
                         "sensibilite": "200 scénarios : poids ± 30 %, seuils q/p/v et λ entre 0,5 et 0,8 ; une décision est "
                                        "« fragile » si la catégorie change dans au moins 25 % des scénarios."},
        }
        if self.demo:
            txt["donnees_et_simulation"] = (
                "Démonstration : 40 prêts réels SBA 7(a) du jeu de test (2015-2016). La probabilité de défaut vient d'un modèle "
                "EBM entraîné sur 2000-2012 et calibré sur 2013-2014 ; VAN, délai, DSCR, dette/fonds propres, garanties et ESG "
                "sont SIMULÉS (paramètres Damodaran et hypothèses d'équipe). Les noms des projets sont fictifs.")
            txt["limites"] = ["indicateurs financiers simulés : une catégorie n'est conservée que dans environ la moitié des "
                              "tirages de simulation", "durée exacte des prêts altérée après l'octroi : remplacée par 3 tranches",
                              "le modèle sous-estime légèrement le risque en 2015-2016", "décision finale humaine"]
        return txt if sujet in (None, "", "tout") else {sujet: txt.get(sujet, f"sujet inconnu ; choix : {list(txt)}")}

    def donnees_historiques(self, sujet: str = "blocs", dimension: str | None = None, modalite: str | None = None) -> dict:
        if not self.demo:
            return {"erreur": "les données historiques SBA ne concernent que le jeu de démonstration"}
        D, R = engine.D, engine.ROOT / "reports"
        if sujet == "blocs":
            a = json.loads((D / "audit_donnees.json").read_text(encoding="utf-8"))
            return {"source": "prêts SBA 7(a) FOIA", "blocs": {k: {"annees": v["annees"], "prets": v["lignes"],
                                                                   "taux_de_defaut": f"{100 * v['taux_defaut']:.2f} %"}
                                                               for k, v in a["blocs"].items()},
                    "definition_defaut": "perte (CHGOFF) au plus 60 mois après le premier versement"}
        if sujet == "audit":
            a = json.loads((D / "audit_donnees.json").read_text(encoding="utf-8"))
            return {k: a[k] for k in ("lignes_brutes", "lignes_statut_invalide_retirees", "doublons_exacts_retires",
                                      "sans_FirstDisbursementDate_total", "date_du_fichier") if k in a}
        if sujet == "metriques_modele":
            t = pd.read_csv(R / "metriques_test.csv")
            t = t[t.version == "calibre"]
            return {"test_2015_2016": {m: {"AUROC": f"{x.AUROC:.3f}", "AUPRC": f"{x.AUPRC:.3f}", "Brier": f"{x.Brier:.4f}",
                                            "pd_moyenne": f"{100 * x.pd_moyenne:.2f} %", "taux_reel": f"{100 * x.taux_defaut_reel:.2f} %"}
                                        for m, x in zip(t.modele, t.itertuples())}, "modele_retenu": "ebm"}
        if sujet == "importance_variables":
            p = pd.read_csv(R / "permutation_importance_validation.csv").head(8)
            return {"permutation_validation_baisse_AUROC": {v: f"{b:.3f}" for v, b in zip(p.variable, p.baisse_AUROC_moyenne)}}
        if sujet == "erreurs_segment":
            e = pd.read_csv(R / "erreurs_par_segment_validation.csv")
            dim = {"secteur": "secteur", "etat": "etat", "procedure": "procedure"}.get(dimension or "etat", "etat")
            e = e[e.dimension == dim]
            if modalite:
                e = e[e.modalite.astype(str).str.upper() == str(modalite).upper()]
            e = e.sort_values("n", ascending=False).head(10)
            return {"dimension": dim, "segments": {str(m): {"prets": int(n), "taux_reel": f"{100 * t:.1f} %",
                                                            "pd_moyenne": f"{100 * p:.1f} %"}
                                                   for m, n, t, p in zip(e.modalite, e.n, e.taux_reel, e.pd_moyenne)}}
        return {"erreur": "sujet inconnu ; choix : blocs, audit, metriques_modele, importance_variables, erreurs_segment"}


# ----------------------------------------------------------------------------------------------- déclarations
def tool_specs(ws: Workspace) -> list[dict]:
    crit = [c["colonne"] for c in ws.crit]
    s = lambda d, **k: {"type": "string", "description": d, **k}  # noqa: E731
    specs = [
        ("capacites", "Ce que le conseiller sait faire et ne sait pas faire, et les méthodes réellement utilisées.", {}, []),
        ("resume_resultats", "Vue d'ensemble : nombre de projets, catégories, exclusions, poids, meilleurs projets, fragilité.", {}, []),
        ("lister_projets", "Liste de projets filtrée et triée (ex. les 5 meilleurs Financer, les plus risqués).",
         {"categorie": s("Financer, Conditionner ou Refuser", enum=mcda.CATEGORIES),
          "tri": s("critère de tri (colonne) ; absent = rang ELECTRE III", enum=crit),
          "ordre": s("croissant ou decroissant", enum=["croissant", "decroissant"]),
          "limite": {"type": "integer", "description": "nombre maximum de projets"},
          "inclure_ecartes": {"type": "boolean", "description": "inclure les projets écartés par le filtre"},
          "texte": s("filtre texte sur le nom, le secteur ou l'État")}, []),
        ("fiche_projet", "Tous les faits sur un projet : critères, catégorie, rangs, position par rapport aux seuils, fragilité.",
         {"nom": s("nom (ou partie du nom) du projet")}, ["nom"]),
        ("comparer_projets", "Compare 2 à 6 projets critère par critère.",
         {"noms": {"type": "array", "items": {"type": "string"}, "description": "noms des projets"}}, ["noms"]),
        ("statistiques", "Moyenne, médiane, minimum, maximum d'un critère, éventuellement par catégorie, secteur ou État.",
         {"critere": s("colonne du critère", enum=crit), "groupe": s("regroupement", enum=["categorie", "secteur", "etat"])},
         ["critere"]),
        ("simuler_scenario", "Et si ? Relance le MCDA avec un ou plusieurs nouveaux poids, une autre coupure λ ou sans "
                             "filtre, et liste les projets qui changent de catégorie. Les poids non cités gardent leurs "
                             "proportions et se partagent le reste.",
         {"poids": {"type": "array", "description": "nouveaux poids imposés (un ou plusieurs critères)",
                    "items": {"type": "object", "properties": {
                        "critere": s("colonne du critère", enum=crit),
                        "poids": {"type": "number", "description": "poids entre 0 et 1 (ex. 0.3 pour 30 %)"}},
                        "required": ["critere", "poids"]}},
          "coupure": {"type": "number", "description": "coupure λ entre 0,5 et 1"},
          "sans_filtre": {"type": "boolean", "description": "désactiver le filtre éliminatoire"}}, []),
        ("parametres_methode", "Paramètres et explication de la méthode (critères, poids/AHP, filtre, profils, seuils, méthodes, données, limites).",
         {"sujet": s("sujet", enum=["tout", "criteres", "poids", "filtre", "profils_electre_tri", "seuils_et_coupure",
                                     "methodes", "donnees_et_simulation", "limites"])}, []),
    ]
    if ws.demo:
        specs.append(("donnees_historiques", "Données historiques SBA 7(a) et modèle de risque : blocs et taux de défaut, audit, "
                                             "métriques, importance des variables, erreurs par État/secteur/procédure.",
                      {"sujet": s("sujet", enum=["blocs", "audit", "metriques_modele", "importance_variables", "erreurs_segment"]),
                       "dimension": s("pour erreurs_segment", enum=["etat", "secteur", "procedure"]),
                       "modalite": s("code d'État (ex. CA) ou de secteur NAICS à 2 chiffres")}, ["sujet"]))
    return [{"name": n, "description": d, "parameters": {"type": "object", "properties": p, "required": r}}
            for n, d, p, r in specs]


def run_tool(ws: Workspace, name: str, args: dict) -> dict:
    fn = getattr(ws, name, None)
    if fn is None or name.startswith("_") or name in ("find", "short", "f"):
        return {"erreur": f"outil inconnu : {name}"}
    try:
        return json.loads(json.dumps(fn(**(args or {})), ensure_ascii=False, default=str))
    except TypeError as e:
        return {"erreur": f"arguments invalides : {e}"}


# ----------------------------------------------------------------------------------------------- boucle LLM
def _gemini_schema(x):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "type":
                out[k] = str(v).upper()
            elif k == "required" and not v:
                continue
            else:
                out[k] = _gemini_schema(v)
        return out
    if isinstance(x, list):
        return [_gemini_schema(v) for v in x]
    return x


def _verify(answer: str, sources: list[str]) -> list[str]:
    allowed = set()
    for s in sources:
        allowed |= agent._numbers(s) | agent._numbers(s.replace(" ", "").replace(" ", ""))
    nums = agent._numbers(answer)
    return sorted(n for n in nums if n not in allowed and not (n.isdigit() and int(n) <= 10))


def ask(question: str, history: list[dict], ws: Workspace, max_steps: int = 6) -> dict:
    """Une question -> {answer, source, tools: [{name, args, result}], unverified: [...]}. history : [{role, content}]."""
    provider = (agent._secret("LLM_PROVIDER") or "gemini").lower()
    model = agent._secret("LLM_MODEL") or agent.DEFAULT_MODELS.get(provider, "")
    key = agent._secret("GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY")
    if not key:
        return {"answer": None, "source": f"pas de clé {provider.upper()} dans les secrets", "tools": [], "unverified": []}
    specs, calls = tool_specs(ws), []
    try:
        if provider == "gemini":
            contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                        for m in history[-10:]] + [{"role": "user", "parts": [{"text": question}]}]
            body = {"system_instruction": {"parts": [{"text": SYSTEM}]},
                    "tools": [{"functionDeclarations": [
                        {k: v for k, v in _gemini_schema(sp).items() if not (k == "parameters" and not sp["parameters"]["properties"])}
                        for sp in specs]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096}}
            def run(m):                                   # conversation complète avec un modèle donné
                calls.clear()
                conv = list(contents)
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
                for _ in range(max_steps):
                    out = agent._post(url, {**body, "contents": conv},
                                      {"Content-Type": "application/json", "x-goog-api-key": key})
                    content = out["candidates"][0].get("content", {"parts": []})
                    parts = content.get("parts", [])
                    fcalls = [p["functionCall"] for p in parts if "functionCall" in p]
                    if not fcalls:
                        return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
                    conv.append(content)
                    responses = []
                    for fc in fcalls:
                        res = run_tool(ws, fc["name"], fc.get("args", {}))
                        calls.append({"name": fc["name"], "args": fc.get("args", {}), "result": res})
                        responses.append({"functionResponse": {"name": fc["name"], "response": {"result": res}}})
                    conv.append({"role": "user", "parts": responses})
                return "Je n'ai pas pu conclure en un nombre raisonnable d'étapes ; reformulez la question."
            answer, model = agent.with_fallback(run, agent.gemini_models())
        elif provider == "groq":
            msgs = [{"role": "system", "content": SYSTEM}] + [
                {"role": m["role"], "content": m["content"]} for m in history[-10:]] + [{"role": "user", "content": question}]
            tools = [{"type": "function", "function": sp} for sp in specs]
            for _ in range(max_steps):
                out = agent._post("https://api.groq.com/openai/v1/chat/completions",
                                  {"model": model, "temperature": 0.2, "max_tokens": 1500, "messages": msgs, "tools": tools},
                                  {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
                msg = out["choices"][0]["message"]
                if not msg.get("tool_calls"):
                    answer = (msg.get("content") or "").strip()
                    break
                msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": msg["tool_calls"]})
                for tc in msg["tool_calls"]:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                    res = run_tool(ws, tc["function"]["name"], args)
                    calls.append({"name": tc["function"]["name"], "args": args, "result": res})
                    msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(res, ensure_ascii=False)})
            else:
                answer = "Je n'ai pas pu conclure en un nombre raisonnable d'étapes ; reformulez la question."
        else:
            return {"answer": None, "source": f"fournisseur inconnu : {provider}", "tools": [], "unverified": []}
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")[:200]
        except Exception:
            msg = ""
        return {"answer": None, "source": f"erreur HTTP {e.code} ({model}) {msg}".strip(), "tools": calls, "unverified": []}
    except (urllib.error.URLError, TimeoutError) as e:
        return {"answer": None, "source": f"réseau indisponible ({type(e).__name__})", "tools": calls, "unverified": []}
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {"answer": None, "source": f"réponse inattendue ({type(e).__name__})", "tools": calls, "unverified": []}
    sources = [json.dumps(c["result"], ensure_ascii=False) for c in calls] + [question] + [m["content"] for m in history]
    return {"answer": answer or "(réponse vide)", "source": f"{provider} {model}", "tools": calls,
            "unverified": _verify(answer or "", sources), "foreign_methods": foreign_methods(answer or "")}


def foreign_methods(answer: str) -> list[str]:
    """Noms de méthodes cités dans la réponse mais non utilisés par le projet (ex. « ELECTRE Tri-C »)."""
    return sorted({m.group(0).strip() for m in _FOREIGN.finditer(answer)})


# ----------------------------------------------------------------------------------------------- questions rapides
QUICK = {
    "Ce que je sais faire": ("capacites", {}),
    "Vue d'ensemble des résultats": ("resume_resultats", {}),
    "Les 5 meilleurs projets": ("lister_projets", {"limite": 5, "inclure_ecartes": False}),
    "Projets refusés et motifs": ("lister_projets", {"categorie": "Refuser", "limite": 20}),
    "Explication de la méthode": ("parametres_methode", {"sujet": "methodes"}),
    "Poids des critères": ("parametres_methode", {"sujet": "poids"}),
}
