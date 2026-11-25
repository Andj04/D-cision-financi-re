"""Application Streamlit : outil réutilisable d'aide à la décision d'investissement multicritère.

Pages : tableau de bord, fiche projet, conseiller IA, configuration (import d'un jeu de projets, critères, poids,
filtre, seuils), robustesse, modèle de risque, méthode. Toutes les pages travaillent sur l'« analyse active » :
la démonstration (40 prêts SBA) au départ, ou le jeu importé et configuré par l'utilisateur.
Lancer : streamlit run app/app.py
"""
import streamlit as st

st.set_page_config(page_title="Décision d'investissement multicritère", page_icon="📊", layout="wide")

import common  # noqa: E402
from views import conseiller, configurer, dashboard, methode, modele, projet, robustesse  # noqa: E402

st.markdown(common.CSS, unsafe_allow_html=True)
act = common.active()

pages = [
    st.Page(dashboard.render, title="Tableau de bord", icon="📋", url_path="tableau-de-bord", default=True),
    st.Page(projet.render, title="Fiche projet", icon="🔎", url_path="fiche-projet"),
    st.Page(conseiller.render, title="Conseiller IA", icon="💬", url_path="conseiller"),
    st.Page(configurer.render, title="Configurer l'analyse", icon="⚙️", url_path="configurer"),
    st.Page(robustesse.render, title="Robustesse", icon="🧪", url_path="robustesse"),
    st.Page(modele.render, title="Modèle de risque", icon="🧠", url_path="modele"),
    st.Page(methode.render, title="Méthode", icon="📐", url_path="methode"),
]
common.PAGES.update(dict(zip(["tableau-de-bord", "fiche-projet", "conseiller", "configurer", "robustesse", "modele",
                               "methode"], pages)))
nav = st.navigation({"Analyse": pages[:3], "Outil": pages[3:5], "Référence": pages[5:]})

with st.sidebar:
    st.markdown("### Analyse active")
    res = act["ev"]["results"]
    st.markdown(f"**{act['config'].get('nom', 'Analyse')}**")
    counts = res["categorie_finale"].value_counts()
    st.markdown(f"{len(res)} projets · 🟢 {counts.get('Financer', 0)} · 🟠 {counts.get('Conditionner', 0)} · "
                f"🔴 {counts.get('Refuser', 0)}")
    if not act["demo"]:
        if st.button("↩️ Revenir à la démonstration", width="stretch"):
            st.session_state.pop("active", None)
            st.rerun()
    st.divider()
    st.markdown("**ML** estime le risque · **MCDA** classe et décide · **LLM** rédige et répond · "
                "**le comité** décide")
    st.markdown("🟢 LLM configuré" if common.llm_ready() else "⚪ Pas de clé LLM : notes de gabarit")

nav.run()
