"""Méthode : chaîne de décision, paramètres de l'analyse active, limites."""
import json

import pandas as pd
import streamlit as st

import common
import engine
from common import D, P


def render():
    act = common.active()
    cfg = act["config"]
    common.page_header("Référence", "Méthode", "Comment la recommandation est construite, avec quels paramètres, "
                       "et quelles sont ses limites.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        common.section("Chaîne de décision")
        for t, d in [("1. Données", "Démonstration : prêts SBA 7(a) FOIA ; cible `default_60m` (perte ≤ 60 mois après le "
                      "premier versement) ; séparation temporelle 2000-2012 / 2013-2014 / 2015-2016. Ou votre propre jeu."),
                     ("2. Modèle de risque", "Régression logistique et EBM, calibration isotonique ; EBM retenu. Sorties : "
                      "`pd_ml` et facteurs de risque. Peut noter un nouveau jeu de prêts au format SBA."),
                     ("3. Indicateurs", "Démonstration : VAN, délai, DSCR, dette, garanties, ESG simulés pour 40 prêts "
                      "(Damodaran + hypothèses d'équipe), indépendamment de l'issue réelle."),
                     ("4. MCDA", "Poids (saisis ou AHP), filtre éliminatoire, ELECTRE III (classement), ELECTRE TRI "
                      "(catégorie), PROMETHEE II (comparaison), sensibilité."),
                     ("5. IA générative", "L'agent rédige une note par projet ; le conseiller répond aux questions en "
                      "interrogeant des outils de calcul. Aucun des deux ne calcule ; leurs chiffres sont vérifiés."),
                     ("6. Comité", "Décision finale humaine.")]:
            st.markdown(f"<div class='step'><b>{t}</b><br><span class='muted'>{d}</span></div>", unsafe_allow_html=True)
    with c2:
        common.section("Paramètres de l'analyse active")
        st.markdown(f"**{cfg.get('nom')}**  \nPondération : "
                    f"{'AHP' if cfg.get('mode_poids') == 'ahp' else 'poids saisis'} · seuils q/p/v = "
                    f"{cfg['seuils']['q']:.0%} / {cfg['seuils']['p']:.0%} / {cfg['seuils']['v']:.0%} · λ = {cfg['coupure']:g}")
        st.markdown("**Filtre éliminatoire**  \n" + ("\n".join(f"- {r.get('libelle') or r['critere']}"
                                                              for r in cfg.get("filtre", [])) or "- aucune règle"))
        if act["demo"]:
            filt = json.loads((P / "filtre_eliminatoire.json").read_text(encoding="utf-8"))
            st.caption(f"Seuil de risque = 2 × {100 * filt['taux_defaut_reference']:.2f} % (taux de la validation, régime "
                       f"de calibration de `pd_ml`) ; le taux 2000-2012 ({100 * filt['taux_entrainement_non_utilise']:.1f} %, "
                       "crise de 2008) rendrait ce filtre inopérant.")
            post = D / "controle_a_posteriori.csv"
            if post.exists():
                common.section("Contrôle a posteriori (illustration)")
                st.dataframe(pd.read_csv(post), hide_index=True, width="stretch")
                st.caption("Vrais défauts parmi les 40 prêts, lus APRÈS les décisions et jamais utilisés dans un calcul. "
                           "Quelques cas : aucune valeur de preuve.")
    common.section("Critères et profils ELECTRE TRI")
    st.dataframe(pd.DataFrame([{"Critère": c["libelle"], "Colonne": c["colonne"],
                                "Sens": "à maximiser" if c["sens"] == "max" else "à minimiser",
                                "Poids": f"{100 * float(act['ev']['weights'][c['colonne']]):.1f} %",
                                "b1 Refuser | Conditionner": engine.fmt(c["b1"], c.get("format", "dec2")),
                                "b2 Conditionner | Financer": engine.fmt(c["b2"], c.get("format", "dec2"))}
                               for c in cfg["criteres"]]), hide_index=True, width="stretch")
    if act["demo"]:
        prof = pd.read_csv(P / "profils_electre_tri.csv")[["critere", "type", "justification"]]
        with st.expander("Justification des profils de la démonstration"):
            st.dataframe(prof, hide_index=True, width="stretch")
    common.section("Limites connues")
    st.markdown(
        "- Démonstration : indicateurs financiers **simulés** ; une catégorie n'est conservée que dans environ la moitié "
        "des tirages de simulation. Les noms des projets sont **fictifs** (activité réelle, nom de famille inventé).\n"
        "- Le filtre DSCR découle de la simulation : la tranche « faible » du guide (0,8-1,0) est toujours sous 1.\n"
        "- La durée exacte des prêts (`TermInMonths`) est altérée après l'octroi pour les prêts en défaut : seules "
        "3 tranches sont utilisées, dans le modèle comme dans la simulation.\n"
        "- Seuils q/p/v relatifs à l'étendue du lot : ajouter un projet peut les déplacer.\n"
        "- Les vérificateurs de l'IA contrôlent les nombres, pas le sens des phrases : relecture humaine nécessaire.")
