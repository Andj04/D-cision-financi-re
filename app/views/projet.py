"""Fiche projet : position par rapport aux seuils de décision, facteurs de risque, note rédigée par l'agent."""
import altair as alt
import pandas as pd
import streamlit as st

import agent
import common
import engine
from common import badge, md


def render():
    act = common.active()
    res, crit = act["ev"]["results"], act["config"]["criteres"]
    order = res.sort_values(["ecarte", "rang_electre3"])
    names = order["nom"].tolist()
    default = st.session_state.get("projet_choisi")
    pid = st.selectbox("Projet", names, index=names.index(default) if default in names else 0,
                       format_func=lambda n: f"{n} · {res.set_index('nom').loc[n, 'categorie_finale']}")
    st.session_state["projet_choisi"] = pid
    row = res[res.nom == pid].iloc[0]

    h1, h2 = st.columns([3, 2])
    with h1:
        st.markdown(f"### {md(pid)} &nbsp; {badge(row.categorie_finale)}", unsafe_allow_html=True)
        names = {"secteur": "Secteur", "etat": "État", "montant_pret": "Prêt demandé",
                 "investissement": "Investissement simulé", "duree_standard_annees": "Durée standard (ans)"}
        info = [f"{names.get(c, c.replace('_', ' ').capitalize())} : "
                f"{engine.fmt(row[c], 'money') if c in ('montant_pret', 'investissement') else row[c]}"
                for c in act["config"].get("colonnes_info", []) if c in row and pd.notna(row[c])]
        if info:
            st.markdown(md(f"<span class='muted'>{' · '.join(map(str, info))}</span>"), unsafe_allow_html=True)
    with h2:
        if row.ecarte:
            st.error(md(f"Écarté par le filtre : {row.motif_ecarte}"))
        else:
            st.markdown(f"**Rang ELECTRE III** : {row.rang_electre3:.0f} / {int((~res.ecarte).sum())} · "
                        f"**PROMETHEE II** : {row.rang_promethee2:.0f}")
            if row.get("position_ambigue"):
                st.warning("Rang ambigu : les deux distillations d'ELECTRE III divergent "
                           f"({row.rang_distillation_descendante:.0f} contre {row.rang_distillation_ascendante:.0f}).")

    m = st.columns(min(4, len(crit)))
    for box, c in zip(m, crit[:4]):
        box.metric(c["libelle"], engine.fmt(row[c["colonne"]], c.get("format", "dec2")))

    a, b = st.columns([1.15, 1])
    with a:
        st.markdown("#### Position par rapport aux seuils de décision")
        recs, pts = [], []
        for c in crit:
            col, s = c["colonne"], (1 if c["sens"] == "max" else -1)
            vals = (s * res[col].astype(float)).to_numpy()

            def rank(v, vals=vals, s=s):       # rang percentile parmi les projets (robuste aux valeurs extrêmes)
                return float(((vals < s * v).sum() + 0.5 * (vals == s * v).sum()) / len(vals))
            b1, b2 = rank(float(c["b1"])), rank(float(c["b2"]))
            b2 = max(b1, b2)
            recs += [{"critere": c["libelle"], "zone": "Refuser", "x0": 0.0, "x1": b1},
                     {"critere": c["libelle"], "zone": "Conditionner", "x0": b1, "x1": b2},
                     {"critere": c["libelle"], "zone": "Financer", "x0": b2, "x1": 1.0}]
            pts.append({"critere": c["libelle"], "pos": rank(float(row[col])),
                        "valeur": engine.fmt(row[col], c.get("format", "dec2"))})
        ylab = alt.Y("critere:N", sort=[c["libelle"] for c in crit], title=None, axis=alt.Axis(labelLimit=230))
        zl = alt.Chart(pd.DataFrame(recs)).mark_bar(height=16, opacity=.28).encode(
            y=ylab, x=alt.X("x0:Q", title="← moins bon · rang parmi les projets · meilleur →",
                            scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(labels=False, ticks=False)),
            x2="x1:Q", color=alt.Color("zone:N", title="Zone ELECTRE TRI", scale=common.cat_scale()))
        pl = alt.Chart(pd.DataFrame(pts)).mark_point(filled=True, size=170, color="#1c2430", shape="diamond").encode(
            y=ylab, x="pos:Q", tooltip=[alt.Tooltip("critere", title="Critère"), alt.Tooltip("valeur", title="Valeur")])
        st.altair_chart((zl + pl).properties(height=34 * len(crit) + 40), width="stretch")
        st.caption("Bandes : zones délimitées par les profils ELECTRE TRI b1 et b2. Losange : le projet. "
                   "Les critères à minimiser sont retournés (à droite = meilleur).")
        with st.expander("Valeurs détaillées"):
            st.dataframe(pd.DataFrame({
                "Critère": [c["libelle"] for c in crit],
                "Valeur": [engine.fmt(row[c["colonne"]], c.get("format", "dec2")) for c in crit],
                "Sens": ["à maximiser" if c["sens"] == "max" else "à minimiser" for c in crit],
                "Seuil b1 → b2": [f"{engine.fmt(c['b1'], c.get('format', 'dec2'))} → "
                                  f"{engine.fmt(c['b2'], c.get('format', 'dec2'))}" for c in crit]}),
                hide_index=True, width="stretch")
        if isinstance(row.get("facteurs_risque"), str):
            st.markdown(md(f"**Facteurs qui augmentent le risque (modèle EBM)** : {row.facteurs_risque}"))
        r1, r2 = st.columns(2)
        if "part_scenarios_categorie_differente" in row and pd.notna(row.part_scenarios_categorie_differente):
            r1.metric("Catégorie modifiée", f"{row.part_scenarios_categorie_differente:.0%}",
                      help="Part des 200 scénarios MCDA (poids ± 30 %, seuils q/p/v, coupure λ entre 0,5 et 0,8) "
                           "où la catégorie change")
        if "part_graines_meme_categorie" in row and pd.notna(row.part_graines_meme_categorie):
            r2.metric("Stable à la simulation", f"{row.part_graines_meme_categorie:.0%}",
                      help="20 tirages des indicateurs financiers simulés pour ce même prêt")

    with b:
        st.markdown("#### Note de l'agent")
        key = f"note_{act['config'].get('nom')}_{row['id']}"
        if st.button("✍️ Rédiger la note avec le LLM", type="primary", width="stretch"):
            with st.spinner("Rédaction puis vérification des chiffres…"):
                st.session_state[key] = agent.write_note(row.to_dict(), crit)
        res_note = st.session_state.get(key) or {
            "note": agent.template_note(agent.build_facts(row.to_dict(), crit)), "source": "gabarit",
            "verification": "note par défaut (cliquer sur le bouton pour faire rédiger le LLM)"}
        with st.container(border=True):
            st.markdown(md(res_note["note"]))
        ok = res_note["source"].startswith("LLM")
        st.markdown(md(f"{'✅' if ok else 'ℹ️'} **Source** : {res_note['source']}  \n"
                       f"**Contrôle** : {res_note['verification']}"))
        st.info("L'agent recopie les résultats déjà calculés ; un vérificateur rejette toute note contenant un "
                "chiffre absent des données. La décision finale revient au comité.", icon="🛡️")
        if st.button("💬 Poser une question sur ce projet au conseiller", width="stretch"):
            st.session_state["chat_prefill"] = f"Que penses-tu du projet {pid} ? Quels sont ses points forts et ses risques ?"
            st.switch_page(common.PAGES["conseiller"])
