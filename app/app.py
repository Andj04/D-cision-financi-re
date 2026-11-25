"""Application Streamlit : outil réutilisable d'aide à la décision d'investissement multicritère.

Pages : tableau de bord, fiche projet, conseiller IA, configuration (import d'un jeu de projets, critères, poids,
filtre, seuils), robustesse, modèle de risque, méthode. Toutes les pages travaillent sur l'« analyse active » :
la démonstration (40 prêts SBA) au départ, ou le jeu importé et configuré par l'utilisateur.
Lancer : streamlit run app/app.py
"""
import sys
from pathlib import Path

import streamlit as st

if str(Path(__file__).resolve().parent) not in sys.path:      # common.py et views/ (dossier de l'application)
    sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="Aide à la décision d'investissement · Centrale Casablanca",
                   page_icon=str(Path(__file__).resolve().parent / "assets/logo-ecc-icone.png"), layout="wide")

import common  # noqa: E402
from views import conseiller, configurer, dashboard, methode, modele, projet, robustesse  # noqa: E402

st.markdown(common.CSS, unsafe_allow_html=True)
st.logo(str(common.ASSETS / "logo-ecc-horizontal.png"), size="large",
        icon_image=str(common.ASSETS / "logo-ecc-icone.png"))
act = common.active()

pages = [
    st.Page(dashboard.render, title="Tableau de bord", icon=":material/space_dashboard:", url_path="tableau-de-bord",
            default=True),
    st.Page(projet.render, title="Fiche projet", icon=":material/description:", url_path="fiche-projet"),
    st.Page(conseiller.render, title="Conseiller IA", icon=":material/forum:", url_path="conseiller"),
    st.Page(configurer.render, title="Configurer l'analyse", icon=":material/tune:", url_path="configurer"),
    st.Page(robustesse.render, title="Robustesse", icon=":material/verified:", url_path="robustesse"),
    st.Page(modele.render, title="Modèle de risque", icon=":material/monitoring:", url_path="modele"),
    st.Page(methode.render, title="Méthode", icon=":material/menu_book:", url_path="methode"),
]
common.PAGES.update(dict(zip(["tableau-de-bord", "fiche-projet", "conseiller", "configurer", "robustesse", "modele",
                               "methode"], pages)))
nav = st.navigation({"Analyse": pages[:3], "Outil": pages[3:5], "Référence": pages[5:]})

with st.sidebar:
    res = act["ev"]["results"]
    counts = res["categorie_finale"].value_counts()
    st.markdown(
        f"<div class='side-card'><div class='t'>Analyse active</div>"
        f"<div class='n'>{common.h(act['config'].get('nom', 'Analyse'))}</div>"
        f"{len(res)} projets &nbsp;·&nbsp; {common.dot('Financer')}{counts.get('Financer', 0)} &nbsp;"
        f"{common.dot('Conditionner')}{counts.get('Conditionner', 0)} &nbsp;{common.dot('Refuser')}{counts.get('Refuser', 0)}"
        f"</div>", unsafe_allow_html=True)
    if not act["demo"]:
        if st.button("Revenir à la démonstration", icon=":material/undo:", width="stretch"):
            st.session_state.pop("active", None)
            st.rerun()
    st.markdown(
        f"<div class='side-foot' style='margin-top:14px'>"
        f"<span class='tag {'ok' if common.llm_ready() else 'info'}'>"
        f"{'IA générative active' if common.llm_ready() else 'IA générative : mode gabarit'}</span><br><br>"
        "Le modèle ML estime le risque, le MCDA classe et décide, l'IA rédige et répond ; "
        "la décision finale revient au comité.<br><br>"
        "<b>Projet de machine learning</b> · École Centrale Casablanca · 2026<br>"
        "Mathieu · Adingra · Serge · Sarah</div>", unsafe_allow_html=True)

nav.run()
