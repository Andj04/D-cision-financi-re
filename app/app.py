"""Étape 10 : application Streamlit : classement MCDA, critères, explications ML, note de l'agent.

Lecture seule des fichiers produits par le pipeline (data/processed, reports). Aucun calcul de décision ici :
le MCDA est déterministe (src/mcda.py) et l'agent LLM ne fait que rédiger (src/agent.py).
Lancer : streamlit run app/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import agent  # noqa: E402

D, R, P = ROOT / "data/processed", ROOT / "reports", ROOT / "params"
st.set_page_config(page_title="Décision d'investissement multicritère", page_icon="📊", layout="wide")


@st.cache_data
def load():
    res = pd.read_csv(D / "mcda_resultats.csv")
    sim = pd.read_csv(D / "simulation_detail.csv")[["loan_id", "secteur_damodaran", "etat", "montant_pret"]]
    expl = pd.read_csv(D / "explications_projets.csv")[["loan_id", "facteurs_risque"]]
    return res.merge(sim, on="loan_id").merge(expl, on="loan_id")


df = load()
CAT_ICON = {"Financer": "🟢 Financer", "Conditionner": "🟠 Financer si…", "Refuser": "🔴 Refuser"}
df["decision"] = df["categorie_finale"].map(CAT_ICON)

st.title("IA agentique pour la décision d'investissement multicritère")
st.caption("Le machine learning ne calcule que le risque, le MCDA décide, le LLM rédige. "
           "La décision finale revient au comité d'investissement.")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Classement", "Fiche projet", "Modèle de risque", "Sensibilité", "Méthode"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projets candidats", len(df))
    c2.metric("🟢 Financer", int((df.categorie_finale == "Financer").sum()))
    c3.metric("🟠 Conditionner", int((df.categorie_finale == "Conditionner").sum()))
    c4.metric("🔴 Refuser (dont écartés)", int((df.categorie_finale == "Refuser").sum()))
    view = df.sort_values(["ecarte", "rang_electre3"]).copy()
    view["pd_ml"] = (100 * view.pd_ml).round(1)
    cols = ["loan_id", "decision", "rang_electre3", "rang_promethee2", "ecarte", "motif_ecarte", "pd_ml", "van",
            "delai_recuperation", "dscr_min", "dette_fonds_propres", "couverture_garanties", "esg_score",
            "emplois_par_million", "categorie_tri_optimiste", "decision_fragile"]
    st.dataframe(view[cols].rename(columns={"pd_ml": "pd_ml (%)"}), use_container_width=True, hide_index=True)
    st.caption("Rang ELECTRE III (principal) et PROMETHEE II (comparaison) ; décision = ELECTRE TRI pessimiste. "
               "« ecarte » = filtre éliminatoire (DSCR < 1 ou probabilité de défaut > 2 × taux moyen d'entraînement).")

with tab2:
    pid = st.selectbox("Projet", df.sort_values(["ecarte", "rang_electre3"]).loan_id)
    row = df[df.loan_id == pid].iloc[0]
    a, b = st.columns(2)
    with a:
        st.subheader(row["decision"])
        st.write(f"**Secteur** : {row.secteur_damodaran} · **État** : {row.etat} · **Montant** : {row.montant_pret:,.0f} $")
        crit = pd.DataFrame({
            "Critère": ["Probabilité de défaut (ML)", "VAN ($)", "Délai de récupération (ans)", "DSCR minimum",
                        "Dette / fonds propres", "Couverture par les garanties", "Score ESG", "Emplois par million $"],
            "Valeur": [f"{100 * row.pd_ml:.1f} %", f"{row.van:,.0f}", f"{row.delai_recuperation:.1f}", f"{row.dscr_min:.2f}",
                       f"{row.dette_fonds_propres:.2f}", f"{row.couverture_garanties:.2f}", f"{row.esg_score:.0f}",
                       f"{row.emplois_par_million:.1f}"]})
        st.table(crit.set_index("Critère"))
        st.write(f"**Facteurs de risque (EBM)** : {row.facteurs_risque}")
        st.write(f"**Fragilité** : {row.fragilite_texte} (part de scénarios à catégorie différente : "
                 f"{row.part_scenarios_categorie_differente:.0%})")
    with b:
        st.subheader("Note de l'agent")
        if st.button("Rédiger / actualiser la note"):
            with st.spinner("Rédaction…"):
                st.session_state[f"note_{pid}"] = agent.write_note(row.to_dict())
        res = st.session_state.get(f"note_{pid}")
        if res is None:
            facts = agent.build_facts(row.to_dict())
            res = {"note": agent.template_note(facts), "source": "gabarit",
                   "verification": "note par défaut (sans appel LLM)"}
        st.info(res["note"])
        st.caption(f"Source : {res['source']} · Contrôle : {res['verification']}")
        st.warning("Note d'aide à la décision : l'agent recopie les résultats du MCDA, il ne calcule rien. "
                   "La décision finale revient au comité.")

with tab3:
    st.subheader("Validation (2013-2014)")
    st.dataframe(pd.read_csv(R / "metriques_validation.csv").round(4), hide_index=True, use_container_width=True)
    st.subheader("Test (2015-2016), lu une seule fois après le choix du modèle")
    st.dataframe(pd.read_csv(R / "metriques_test.csv").round(4), hide_index=True, use_container_width=True)
    st.image(str(R / "calibration.png"), caption="Calibration : logistique vs EBM")
    st.image(str(R / "importance_variables.png"), caption="Importance native EBM et par permutation")
    st.image(str(R / "ebm_effets.png"), caption="Effet des variables principales (EBM)")
    st.dataframe(pd.read_csv(R / "erreurs_par_segment_test.csv").round(3), hide_index=True, use_container_width=True)

with tab4:
    st.dataframe(pd.read_csv(D / "sensibilite_seuils_coupure.csv").round(3), hide_index=True, use_container_width=True)
    st.dataframe(pd.read_csv(D / "comparaison_methodes.csv").round(3), hide_index=True, use_container_width=True)
    st.dataframe(pd.read_csv(P / "poids_ahp.csv").round(3), hide_index=True, use_container_width=True)

with tab5:
    st.markdown(
        "**Données** : prêts réels SBA 7(a) FOIA (2000-2019) ; cible `default_60m` ; séparation temporelle "
        "2000-2012 / 2013-2014 / 2015-2016.\n\n"
        "**Simulé** : VAN, délai de récupération, DSCR, dette sur fonds propres, garanties, ESG (paramètres sectoriels "
        "Damodaran, hypothèses d'équipe documentées dans `params/`). Les 40 projets sont des prêts réels du jeu de test "
        "dont les indicateurs financiers sont **simulés**.\n\n"
        "**Limite connue** : la durée exacte du prêt (`TermInMonths`) est altérée après l'octroi pour les prêts en défaut ; "
        "elle est remplacée par 3 tranches.")
