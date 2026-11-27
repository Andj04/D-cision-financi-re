"""Conseiller IA : questions en langage naturel sur le jeu de projets, les résultats et la méthode."""
import re

import streamlit as st

import advisor
import common
from common import md

AVATAR = ":material/support_agent:"
TOOL_FR = {"capacites": "Capacités du conseiller", "resume_resultats": "Vue d'ensemble des résultats", "lister_projets": "Liste de projets",
           "fiche_projet": "Fiche projet", "comparer_projets": "Comparaison de projets", "statistiques": "Statistiques",
           "simuler_scenario": "Scénario « et si »", "parametres_methode": "Paramètres de la méthode",
           "donnees_historiques": "Données historiques SBA", "arbitrage_budget": "Arbitrage budgétaire"}


def _workspace() -> advisor.Workspace:
    act = common.active()
    return advisor.Workspace(act["ev"], demo=act["demo"])


def _suggestions(act) -> list[str]:
    res = act["ev"]["results"]
    best = res[~res.ecarte].sort_values("rang_electre3")["nom"].head(2).tolist()
    s = ["Résume-moi les résultats de l'analyse.",
         "Quels projets financer en priorité, et lesquels sont fragiles ?"]
    if len(best) == 2:
        s.append(f"Compare {best[0]} et {best[1]}.")
    crit = act["config"]["criteres"]
    s.append(f"Que se passe-t-il si le critère « {crit[0]['libelle']} » pèse 40 % ?")
    s.append("Explique-moi simplement comment fonctionne ELECTRE TRI.")
    if act["demo"]:
        s.append("Le modèle de risque est-il fiable ? Quels sont ses points faibles ?")
    return s


def _clean(text: str) -> str:
    """Retire la notation LaTeX courte que le modèle glisse parfois ($q$, $\\lambda$) sans toucher aux montants en $."""
    text = re.sub(r"\$\\?(lambda)\$", "λ", text)
    return re.sub(r"\$([A-Za-zλ_{}]{1,12})\$", r"\1", text)


def _show_answer(msg: dict):
    st.markdown(md(_clean(msg["content"])))
    if msg.get("tools"):
        with st.expander(f"Données consultées ({len(msg['tools'])})", icon=":material/database:"):
            for t in msg["tools"]:
                st.markdown(f"**{TOOL_FR.get(t['name'], t['name'])}** `{t['args']}`")
                st.json(t["result"], expanded=False)
    tags = []
    if msg.get("foreign_methods"):
        tags.append(("warn", f"Méthode citée mais non utilisée : {', '.join(msg['foreign_methods'])}"))
    if msg.get("unverified"):
        tags.append(("warn", f"Chiffres non retrouvés dans les données : {', '.join(msg['unverified'])}"))
    elif msg.get("tools"):
        tags.append(("ok", "Chiffres vérifiés"))
    html_tags = "".join(f"<span class='tag {k}'>{common.h(t)}</span>" for k, t in tags)
    st.markdown(f"{html_tags}<span class='muted'>{common.h(msg.get('source', ''))}</span>", unsafe_allow_html=True)


def _ask(question: str):
    hist = st.session_state.setdefault("chat", [])
    hist.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(md(question))
    with st.chat_message("assistant", avatar=AVATAR):
        with st.spinner("Le conseiller consulte les données…"):
            past = [{"role": m["role"], "content": m["content"]} for m in hist[:-1]]
            r = advisor.ask(question, past, _workspace())
        if r["answer"] is None:
            msg = {"role": "assistant", "content": f"Je ne peux pas répondre pour l'instant : {r['source']}. "
                   "Les questions rapides ci-dessous fonctionnent sans LLM.", "source": "indisponible", "tools": []}
        else:
            msg = {"role": "assistant", "content": r["answer"], "source": r["source"], "tools": r["tools"],
                   "unverified": r["unverified"], "foreign_methods": r.get("foreign_methods", [])}
        _show_answer(msg)
    hist.append(msg)


def render():
    act = common.active()
    common.page_header("Conseiller IA", "Votre conseiller financier virtuel",
                       f"Questions en langage naturel sur « {act['config'].get('nom', 'l’analyse')} » : projets, "
                       "résultats, scénarios « et si », méthode.",
                       ["Répond à partir des données calculées", "Ne calcule rien lui-même", "Chiffres vérifiés",
                        "La décision reste au comité"])
    ready = common.llm_ready()
    hist = st.session_state.setdefault("chat", [])

    top = st.columns([4, 1])
    if hist and top[1].button("Effacer", width="stretch", icon=":material/delete:"):
        st.session_state["chat"] = []
        st.rerun()
    if not ready:
        st.warning("Aucune clé LLM configurée (GEMINI_API_KEY ou GROQ_API_KEY dans les secrets) : la conversation libre "
                   "est désactivée, mais les questions rapides ci-dessous interrogent directement les données.")

    if not hist:
        with st.chat_message("assistant", avatar=AVATAR):
            st.markdown("Bonjour ! Je suis votre conseiller financier virtuel. Je consulte les résultats de l'analyse "
                        "(catégories, rangs, critères, sensibilité) et j'explique la méthode. Je ne calcule rien "
                        "moi-même : chaque chiffre vient des données, et vous pouvez voir lesquelles j'ai consultées. "
                        "Quelques idées de questions :")
            cols = st.columns(2)
            for i, s in enumerate(_suggestions(act)):
                if cols[i % 2].button(s, key=f"sugg{i}", width="stretch", disabled=not ready):
                    st.session_state["chat_prefill"] = s
                    st.rerun()
    for m in hist:
        with st.chat_message(m["role"], avatar=AVATAR if m["role"] == "assistant" else None):
            _show_answer(m) if m["role"] == "assistant" else st.markdown(md(m["content"]))

    pre = st.session_state.pop("chat_prefill", None)
    if pre and ready:
        _ask(pre)
    q = st.chat_input("Votre question (ex. « Pourquoi ce projet est-il refusé ? »)", disabled=not ready)
    if q:
        _ask(q)

    with st.expander("Questions rapides (réponse directe des outils, sans IA)", expanded=not ready, icon=":material/bolt:"):
        ws = _workspace()
        cols = st.columns(len(advisor.QUICK))
        for (label, (tool, args)), c in zip(advisor.QUICK.items(), cols):
            if c.button(label, width="stretch"):
                st.session_state["quick"] = (label, advisor.run_tool(ws, tool, args))
        if "quick" in st.session_state:
            label, out = st.session_state["quick"]
            st.markdown(f"**{label}**")
            st.json(out, expanded=2)
