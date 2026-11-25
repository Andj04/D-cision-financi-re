"""Robustesse : effet de la coupure λ, accord entre méthodes, stabilité de chaque décision, poids."""
import altair as alt
import pandas as pd
import streamlit as st

import common
from common import cat_scale


def render():
    act = common.active()
    ev, res, crit = act["ev"], act["ev"]["results"], act["config"]["criteres"]
    lab = {c["colonne"]: c["libelle"] for c in crit}
    common.hero("Robustesse des décisions", "Les catégories tiennent-elles si l'on change les paramètres ?")
    sens = ev.get("sensibilite")
    if sens is None:
        st.info("La robustesse n'a pas été calculée pour cette analyse (option désactivée ou moins de 3 projets "
                "retenus). Activez-la dans « Configurer l'analyse ».")
    else:
        meth = sens["comparaison_methodes"].iloc[0]
        k = st.columns(4)
        k[0].metric("Spearman E-III / P-II", f"{meth.spearman:.3f}", help="Corrélation de rangs ELECTRE III / PROMETHEE II")
        k[1].metric("Kendall E-III / P-II", f"{meth.kendall:.3f}")
        k[2].metric("Décisions fragiles", f"{int(res.decision_fragile.sum())} / {len(res)}")
        if "part_graines_meme_categorie" in res:
            k[3].metric("Stabilité face à la simulation", f"{res.part_graines_meme_categorie.mean():.0%}",
                        help="Part moyenne des 20 tirages des indicateurs simulés donnant la même catégorie")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Effet de la coupure λ (ELECTRE TRI)")
            t = sens["seuils_et_coupure"]
            lam = t[t.parametre == "coupure lambda"].assign(valeur=lambda d: d.valeur.astype(float))
            st.altair_chart(alt.Chart(lam).mark_line(point=True, color="#1f4e79").encode(
                x=alt.X("valeur:Q", title="Coupure λ"),
                y=alt.Y("projets_categorie_change:Q", title="Projets qui changent de catégorie"),
                tooltip=[alt.Tooltip("valeur", title="λ"), alt.Tooltip("projets_categorie_change", title="Changements")]
            ).properties(height=280), width="stretch")
            st.caption("Scénarios combinés : λ entre 0,5 et 0,8 ; 0,9-1 (quasi-unanimité) est montré ici comme cas extrême.")
        with c2:
            st.markdown("#### ELECTRE III contre PROMETHEE II")
            al = res[~res.ecarte]
            st.altair_chart(alt.Chart(al).mark_circle(size=90).encode(
                x=alt.X("rang_electre3:Q", title="Rang ELECTRE III"), y=alt.Y("rang_promethee2:Q", title="Rang PROMETHEE II"),
                color=alt.Color("categorie_finale:N", legend=None, scale=cat_scale()),
                tooltip=[alt.Tooltip("nom", title="Projet"), "rang_electre3", "rang_promethee2"]
            ).properties(height=280), width="stretch")

        st.markdown("#### Stabilité de chaque décision")
        stab = res[["nom", "categorie_finale"]].copy()
        stab["stab_mcda"] = 1 - res.part_scenarios_categorie_differente
        cols = [("stab_mcda", "Face aux paramètres MCDA (200 scénarios)")]
        if "part_graines_meme_categorie" in res:
            stab["stab_sim"] = res.part_graines_meme_categorie
            cols.append(("stab_sim", "Face au tirage des indicateurs simulés (20 graines)"))
        for (col, title), box in zip(cols, st.columns(len(cols))):
            with box:
                st.markdown(f"**{title}** · part des cas où la catégorie est inchangée")
                st.altair_chart(alt.Chart(stab).mark_bar(cornerRadiusEnd=2).encode(
                    x=alt.X("nom:N", sort=alt.EncodingSortField(col, order="descending"), title=None,
                            axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y(f"{col}:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("categorie_finale:N", title="Décision", scale=cat_scale()),
                    tooltip=[alt.Tooltip("nom", title="Projet"), alt.Tooltip("categorie_finale", title="Décision"),
                             alt.Tooltip(f"{col}:Q", format=".0%", title="Part identique")]
                ).properties(height=220), width="stretch")
        st.caption("Barre haute = décision robuste.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Poids des critères")
        w = ev["weights"]
        st.altair_chart(alt.Chart(pd.DataFrame({"critere": [lab[c] for c in w.index], "poids": w.values})).mark_bar(
            color="#1f4e79", cornerRadiusEnd=3).encode(
            y=alt.Y("critere:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("poids:Q", axis=alt.Axis(format="%"), title=None),
            tooltip=[alt.Tooltip("critere", title="Critère"), alt.Tooltip("poids", format=".1%")]
        ).properties(height=28 * len(w) + 30), width="stretch")
        if ev.get("cr") is not None:
            st.caption(f"AHP : ratio de cohérence {ev['cr']:.4f} (< 0,10 = cohérent).")
    with c4:
        if sens is not None:
            st.markdown("#### Un paramètre à la fois")
            st.dataframe(sens["seuils_et_coupure"].round(3), hide_index=True, width="stretch")
