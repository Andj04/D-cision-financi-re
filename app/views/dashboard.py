"""Tableau de bord : chiffres clés, carte des projets, classement, tableau filtrable."""
import altair as alt
import pandas as pd
import streamlit as st

import common
from common import CAT_ORDER, cat_scale


def render():
    act = common.active()
    res, crit = act["ev"]["results"], act["config"]["criteres"]
    lab = {c["colonne"]: c["libelle"] for c in crit}
    common.hero("Aide à la décision d'investissement multicritère",
                act["config"].get("nom", ""),
                ["① Risque (modèle ML)", "② Filtre éliminatoire", "③ ELECTRE III / TRI · PROMETHEE II",
                 "④ Conseiller IA vérifié", "⑤ Décision du comité"])

    k = st.columns(6)
    k[0].metric("Projets", len(res))
    for i, c in enumerate(CAT_ORDER):
        k[i + 1].metric({"Financer": "🟢 Financer", "Conditionner": "🟠 Sous conditions", "Refuser": "🔴 Refuser"}[c],
                        int((res.categorie_finale == c).sum()))
    k[4].metric("⛔ Écartés", int(res.ecarte.sum()), help="Projets écartés par le filtre éliminatoire")
    if "decision_fragile" in res:
        k[5].metric("⚠️ Fragiles", int(res.decision_fragile.sum()),
                    help="Catégorie différente dans au moins 25 % des scénarios de sensibilité")

    with st.expander("Filtres", expanded=False):
        f1, f2, f3 = st.columns([2, 2, 1])
        cats = f1.multiselect("Catégorie", CAT_ORDER, default=CAT_ORDER)
        group_col = next((c for c in ("secteur", "etat") if c in res), None)
        groups = f2.multiselect(f"Filtrer par {group_col}", sorted(res[group_col].dropna().astype(str).unique())) \
            if group_col else []
        show_out = f3.toggle("Projets écartés", value=True)
    view = res[res.categorie_finale.isin(cats)]
    if groups:
        view = view[view[group_col].astype(str).isin(groups)]
    if not show_out:
        view = view[~view.ecarte]

    left, right = st.columns([1.25, 1])
    with left:
        cols = [c["colonne"] for c in crit]
        dx = "pd_ml" if "pd_ml" in cols else cols[0]
        dy = "dscr_min" if "dscr_min" in cols else cols[1]
        a1, a2 = st.columns(2)
        x = a1.selectbox("Axe horizontal", cols, index=cols.index(dx), format_func=lab.get)
        y = a2.selectbox("Axe vertical", cols, index=cols.index(dy), format_func=lab.get)
        size = "montant_pret" if "montant_pret" in view else None
        enc = dict(
            x=alt.X(f"{x}:Q", title=lab[x], scale=alt.Scale(zero=False)),
            y=alt.Y(f"{y}:Q", title=lab[y], scale=alt.Scale(zero=False)),
            color=alt.Color("categorie_finale:N", title="Décision", scale=cat_scale()),
            tooltip=[alt.Tooltip("nom", title="Projet"), alt.Tooltip("categorie_finale", title="Décision"),
                     alt.Tooltip(f"{x}:Q", title=lab[x], format=",.3f"), alt.Tooltip(f"{y}:Q", title=lab[y], format=",.3f"),
                     alt.Tooltip("motif_ecarte", title="Motif d'exclusion")])
        if size:
            enc["size"] = alt.Size(f"{size}:Q", legend=None, scale=alt.Scale(range=[60, 700]))
        chart = alt.Chart(view).mark_circle(opacity=.85, stroke="#fff", strokeWidth=1).encode(**enc)
        rules = [alt.Chart(pd.DataFrame({"v": [float(r["seuil"])]})).mark_rule(strokeDash=[5, 4], color="#b03a2e")
                 .encode(**{("x" if r["critere"] == x else "y"): "v:Q"})
                 for r in act["config"].get("filtre", []) if r["critere"] in (x, y)]
        st.altair_chart(alt.layer(chart, *rules).properties(height=360), width="stretch")
        if rules:
            st.caption("Lignes pointillées : seuils du filtre éliminatoire.")
    with right:
        st.markdown("#### Classement ELECTRE III")
        top = view[~view.ecarte].sort_values("rang_electre3")
        st.altair_chart(alt.Chart(top).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("nom:N", sort=None, title=None, axis=alt.Axis(labelLimit=230)),
            x=alt.X("credibilite_nette:Q", title="Crédibilité nette (surclasse − est surclassé)"),
            color=alt.Color("categorie_finale:N", legend=None, scale=cat_scale()),
            tooltip=[alt.Tooltip("nom", title="Projet"), alt.Tooltip("rang_electre3", title="Rang ELECTRE III"),
                     alt.Tooltip("rang_promethee2", title="Rang PROMETHEE II"),
                     alt.Tooltip("categorie_finale", title="Décision")]
        ).properties(height=max(360, 16 * len(top))), width="stretch")

    st.markdown("#### Tableau des projets")
    table = view.sort_values(["ecarte", "rang_electre3"])
    show = ["nom", "categorie_finale", "rang_electre3", "rang_promethee2"] + [c["colonne"] for c in crit]
    show += [c for c in ("decision_fragile", "position_ambigue", "motif_ecarte") if c in table]
    cc = {"nom": st.column_config.TextColumn("Projet", width="medium"), "categorie_finale": "Décision",
          "rang_electre3": st.column_config.NumberColumn("Rang E-III", format="%d"),
          "rang_promethee2": st.column_config.NumberColumn("Rang P-II", format="%d"),
          "decision_fragile": st.column_config.CheckboxColumn("Fragile"),
          "position_ambigue": st.column_config.CheckboxColumn("Rang ambigu"), "motif_ecarte": "Motif d'exclusion",
          **common.column_config_for(crit, res)}
    st.dataframe(table[show], hide_index=True, width="stretch", height=430, column_config=cc)
    st.download_button("⬇️ Télécharger les résultats (CSV)", res.to_csv(index=False).encode("utf-8"),
                       "resultats_mcda.csv", "text/csv")
    st.caption(f"Décision = ELECTRE TRI pessimiste (λ = {act['config']['coupure']:g}). « Fragile » : la catégorie change "
               "dans au moins 25 % des scénarios. « Rang ambigu » : les deux distillations d'ELECTRE III divergent.")
