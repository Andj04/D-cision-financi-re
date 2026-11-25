"""Éléments partagés par les pages : style, couleurs, état de l'analyse active, petits composants."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import engine  # noqa: E402

D, R, P = ROOT / "data/processed", ROOT / "reports", ROOT / "params"
CAT_COLOR = {"Financer": "#1a7f5a", "Conditionner": "#c98a00", "Refuser": "#b03a2e"}
CAT_ORDER = ["Financer", "Conditionner", "Refuser"]
PAGES: dict = {}          # rempli par app.py (objets st.Page, pour st.switch_page)
CAT_TEXT = {"Financer": "Financer", "Conditionner": "Financer sous conditions", "Refuser": "Refuser"}

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1350px;}
.hero {background: linear-gradient(120deg, #1f4e79 0%, #2e6f9e 60%, #3f8fb5 100%); color: #fff;
       padding: 20px 26px; border-radius: 14px; margin-bottom: 16px;}
.hero h1 {color: #fff; font-size: 1.6rem; margin: 0 0 4px 0; padding: 0;}
.hero p {margin: 0; opacity: .92;}
.chips {margin-top: 10px;}
.chip {display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.35);
       border-radius: 999px; padding: 3px 12px; margin: 0 6px 6px 0; font-size: .85rem;}
.badge {display: inline-block; color: #fff; border-radius: 8px; padding: 4px 12px; font-weight: 600; font-size: 1rem;}
.muted {color: #5b6675; font-size: .9rem;}
.step {border-left: 4px solid #1f4e79; padding: 6px 12px; margin-bottom: 8px; background: #fff; border-radius: 6px;}
.stepnum {display:inline-block; background:#1f4e79; color:#fff; border-radius:999px; width:26px; height:26px;
          text-align:center; line-height:26px; font-weight:700; margin-right:8px;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #e3e7ee; border-radius: 12px; padding: 10px 14px;}
</style>
"""


def md(text) -> str:
    """Échappe « $ » (sinon Streamlit l'interprète comme une formule LaTeX)."""
    return str(text).replace("$", "\\$")


def badge(cat: str) -> str:
    return f"<span class='badge' style='background:{CAT_COLOR[cat]}'>{CAT_TEXT[cat]}</span>"


def hero(title: str, subtitle: str, chips: list[str] | None = None):
    c = "".join(f"<span class='chip'>{x}</span>" for x in chips or [])
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{md(subtitle)}</p>"
                f"{f'<div class=chips>{c}</div>' if c else ''}</div>", unsafe_allow_html=True)


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
    res = ev["results"]
    extra = [c for c in df.columns if c not in res.columns]
    res = res.join(df[extra]) if extra else res
    st.session_state["active"] = {"ev": {**ev, "results": res}, "demo": demo, "df": df, "config": config}
    st.session_state.pop("chat", None)                 # nouvelle analyse : nouvelle conversation


def active() -> dict:
    if "active" not in st.session_state:
        df, cfg = demo_data()
        set_active(df, cfg, demo=True)
    return st.session_state["active"]


def criteria_meta() -> list[dict]:
    return active()["config"]["criteres"]


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
