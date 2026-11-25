"""Éléments partagés par les pages : identité visuelle (École Centrale Casablanca), état de l'analyse active,
composants d'interface (en-tête de page, cartes d'indicateurs, titres de section, badges)."""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "assets"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _reload_src_if_changed():
    """Streamlit Cloud relance le script après un push SANS redémarrer Python : un module de src/ déjà importé
    resterait dans son ancienne version (ex. « mcda has no attribute OPS »). On recharge donc les modules
    locaux, dans l'ordre des dépendances, dès que les fichiers de src/ ont changé."""
    import importlib
    import types
    files = sorted((ROOT / "src").glob("*.py"))
    stamp = tuple((f.name, f.stat().st_mtime, f.stat().st_size) for f in files)
    reg = sys.modules.setdefault("_src_reload_registry", types.ModuleType("_src_reload_registry"))
    if getattr(reg, "stamp", None) != stamp:
        for name in ["damodaran", "naming", "features", "mcda", "agent", "engine", "advisor"]:
            if name in sys.modules:
                importlib.reload(sys.modules[name])
        reg.stamp = stamp
        try:
            st.cache_data.clear()                       # résultats en cache calculés avec l'ancien code
        except Exception:
            pass


_reload_src_if_changed()
import engine  # noqa: E402

D, R, P = ROOT / "data/processed", ROOT / "reports", ROOT / "params"
PAGES: dict = {}          # rempli par app.py (objets st.Page, pour st.switch_page)

# ----------------------------------------------------------------------------------------------- identité visuelle
GREEN, GREEN_DARK, GREEN_LIGHT = "#02987D", "#01715D", "#E3F3EF"
GREY, GREY_DARK, LINE = "#72777A", "#2B3134", "#E1E6E8"
CAT_COLOR = {"Financer": GREEN, "Conditionner": "#D39B2A", "Refuser": "#C0503F"}
CAT_TINT = {"Financer": GREEN_LIGHT, "Conditionner": "#FBF1DD", "Refuser": "#F8E3DF"}
CAT_ORDER = ["Financer", "Conditionner", "Refuser"]
CAT_TEXT = {"Financer": "Financer", "Conditionner": "Financer sous conditions", "Refuser": "Refuser"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stMarkdown, .stMarkdown span {{
  font-family: 'Inter', 'Segoe UI', sans-serif; }}
/* les icônes gardent leur police (sinon leur nom s'affiche en texte : « space_dashboard ») */
[data-testid="stIconMaterial"], span[class*="material-symbols"], .material-symbols-rounded {{
  font-family: 'Material Symbols Rounded' !important; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }}
.stAppDeployButton, footer {{ display: none !important; }}
h1, h2, h3, h4 {{ color: {GREY_DARK}; letter-spacing: -0.01em; }}

/* barre latérale */
section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {LINE}; }}
img[data-testid="stSidebarLogo"], [data-testid="stSidebarHeader"] img {{ height: 2.9rem !important; max-width: 100%; }}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {{
  background: {GREEN_LIGHT}; border-left: 3px solid {GREEN}; }}
.side-card {{ background: #F7F9F9; border: 1px solid {LINE}; border-radius: 10px; padding: 12px 14px; font-size: .85rem; }}
.side-card .t {{ font-size: .7rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: {GREY}; }}
.side-card .n {{ font-weight: 600; color: {GREY_DARK}; margin: 4px 0 8px; line-height: 1.3; }}
.side-foot {{ font-size: .74rem; color: {GREY}; line-height: 1.5; }}

/* en-tête de page */
.page-head {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 16px;
  border-bottom: 1px solid {LINE}; padding-bottom: 14px; margin-bottom: 20px; }}
.page-head .eyebrow {{ font-size: .72rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: {GREEN}; }}
.page-head h1 {{ font-size: 1.75rem; font-weight: 700; margin: 2px 0 4px; padding: 0; color: {GREY_DARK}; }}
.page-head p {{ margin: 0; color: {GREY}; font-size: .95rem; max-width: 820px; }}
.page-head .rule {{ width: 56px; height: 3px; background: {GREEN}; border-radius: 2px; margin-top: 10px; }}
.pill {{ display: inline-block; font-size: .75rem; font-weight: 500; color: {GREY_DARK}; background: #FFFFFF;
  border: 1px solid {LINE}; border-radius: 999px; padding: 3px 10px; margin: 0 6px 6px 0; }}
.pill b {{ color: {GREEN}; font-weight: 700; margin-right: 4px; }}

/* cartes d'indicateurs */
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 2px 0 20px; }}
.kpi {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 10px; padding: 14px 16px 12px;
  border-top: 3px solid var(--accent, {GREEN}); box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
.kpi .l {{ font-size: .72rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: {GREY}; }}
.kpi .v {{ font-size: 1.65rem; font-weight: 700; color: {GREY_DARK}; line-height: 1.25; margin-top: 4px; }}
.kpi .s {{ font-size: .78rem; color: {GREY}; margin-top: 2px; }}

/* titres de section, cartes */
.sec {{ font-size: 1.02rem; font-weight: 600; color: {GREY_DARK}; border-left: 3px solid {GREEN};
  padding-left: 10px; margin: 22px 0 10px; }}
.sec small {{ font-weight: 400; color: {GREY}; margin-left: 6px; }}
.muted {{ color: {GREY}; font-size: .88rem; }}
.badge {{ display: inline-block; border-radius: 999px; padding: 4px 12px; font-weight: 600; font-size: .85rem;
  border: 1px solid currentColor; vertical-align: middle; }}
.step {{ background: #FFFFFF; border: 1px solid {LINE}; border-left: 3px solid {GREEN}; border-radius: 8px;
  padding: 10px 14px; margin-bottom: 8px; }}
.step b {{ color: {GREY_DARK}; }}
.stepnum {{ display: inline-block; background: {GREEN}; color: #fff; border-radius: 999px; width: 26px; height: 26px;
  text-align: center; line-height: 26px; font-weight: 700; margin-right: 10px; font-size: .85rem; }}
.tag {{ display: inline-block; font-size: .74rem; border-radius: 6px; padding: 2px 8px; margin-right: 6px; }}
.tag.ok {{ background: {GREEN_LIGHT}; color: {GREEN_DARK}; }}
.tag.warn {{ background: #FBF1DD; color: #8A5D00; }}
.tag.info {{ background: #EEF1F2; color: {GREY_DARK}; }}

/* composants Streamlit */
div[data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 10px; padding: 10px 14px; }}
div[data-testid="stExpander"] details {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 10px; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {LINE} !important; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 10px; }}
div[data-testid="stChatMessage"] {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 12px; padding: 12px 16px; }}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {{ font-weight: 600; }}
</style>
"""


def md(text) -> str:
    """Échappe « $ » dans du Markdown (sinon Streamlit l'interprète comme une formule LaTeX)."""
    return str(text).replace("$", "\\$")


def h(text) -> str:
    """Texte inséré dans un bloc HTML : échappé, et « $ » en entité (l'échappement Markdown y resterait visible)."""
    return html.escape(str(text)).replace("$", "&#36;")


def badge(cat: str) -> str:
    return (f"<span class='badge' style='color:{CAT_COLOR[cat]};background:{CAT_TINT[cat]}'>"
            f"{CAT_TEXT[cat]}</span>")


def dot(cat: str) -> str:
    return f"<span style='display:inline-block;width:9px;height:9px;border-radius:50%;background:{CAT_COLOR[cat]};" \
           f"margin:0 5px 1px 0'></span>"


def page_header(eyebrow: str, title: str, subtitle: str = "", pills: list[str] | None = None):
    p = "".join(f"<span class='pill'><b>{i + 1}</b>{html.escape(x)}</span>" for i, x in enumerate(pills or []))
    st.markdown(f"<div class='page-head'><div><div class='eyebrow'>{html.escape(eyebrow)}</div>"
                f"<h1>{h(title)}</h1><p>{h(subtitle)}</p>"
                f"{f'<div style=margin-top:10px>{p}</div>' if p else ''}</div></div>", unsafe_allow_html=True)


def section(title: str, note: str = ""):
    st.markdown(f"<div class='sec'>{html.escape(title)}{f'<small>{html.escape(note)}</small>' if note else ''}</div>",
                unsafe_allow_html=True)


def kpis(items: list[tuple]):
    """items : (libellé, valeur, sous-titre facultatif, couleur d'accent facultative)."""
    cards = []
    for it in items:
        label, value = it[0], it[1]
        sub = it[2] if len(it) > 2 and it[2] else ""
        accent = it[3] if len(it) > 3 and it[3] else GREEN
        cards.append(f"<div class='kpi' style='--accent:{accent}'><div class='l'>{html.escape(str(label))}</div>"
                     f"<div class='v'>{h(value)}</div>"
                     f"{f'<div class=s>{h(sub)}</div>' if sub else ''}</div>")
    st.markdown(f"<div class='kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def cat_scale():
    import altair as alt
    return alt.Scale(domain=CAT_ORDER, range=[CAT_COLOR[c] for c in CAT_ORDER])


# ----------------------------------------------------------------------------------------------- analyse active
@st.cache_data(show_spinner="Évaluation multicritère en cours…")
def cached_evaluate(df: pd.DataFrame, config_json: str, sensitivity: bool = True) -> dict:
    return engine.evaluate(df, json.loads(config_json), sensitivity=sensitivity)


@st.cache_data
def demo_data() -> tuple[pd.DataFrame, dict]:
    return engine.demo_projects(), engine.demo_config()


def set_active(df: pd.DataFrame, config: dict, demo: bool, sensitivity: bool = True):
    ev = cached_evaluate(df, engine.config_to_json(config), sensitivity)
    st.session_state["active"] = {"ev": ev, "demo": demo, "df": df, "config": config}
    st.session_state.pop("chat", None)                 # nouvelle analyse : nouvelle conversation


def active() -> dict:
    if "active" not in st.session_state:
        df, cfg = demo_data()
        set_active(df, cfg, demo=True)
    return st.session_state["active"]


def column_config_for(criteria: list[dict], df: pd.DataFrame) -> dict:
    """Formats d'affichage des colonnes de critères (st.dataframe)."""
    cc = {}
    for c in criteria:
        col, f, lab = c["colonne"], c.get("format", "dec2"), c["libelle"]
        if f == "pct":
            cc[col] = st.column_config.ProgressColumn(lab, format="percent", min_value=0,
                                                      max_value=float(max(df[col].max(), 1e-9)))
        elif f == "money":
            cc[col] = st.column_config.NumberColumn(lab, format="localized")
        else:
            cc[col] = st.column_config.NumberColumn(lab, format={"int": "%.0f", "dec1": "%.1f"}.get(f, "%.2f"))
    return cc


def llm_ready() -> bool:
    import agent
    return bool(agent._secret("GEMINI_API_KEY") or agent._secret("GROQ_API_KEY"))
