"""Arbitrage budgétaire : quels projets financer avec le budget disponible ? (agent d'arbitrage, src/arbitrage.py)"""
import altair as alt
import pandas as pd
import streamlit as st

import arbitrage
import common
from common import CAT_COLOR


def render():
    act = common.active()
    res = act["ev"]["results"]
    common.page_header("Arbitrage budgétaire", "Quels projets financer avec le budget disponible ?",
                       "Le MCDA classe et catégorise les projets ; l'arbitrage choisit lesquels financer sous un budget "
                       "global, des montants minimum et maximum par projet et un plafond par secteur.",
                       ["Budget global", "Bornes par projet", "Diversification sectorielle", "Ordre du comité"])
    p = arbitrage.params_with_defaults(act["config"].get("arbitrage"))
    money_cols = [c for c in res.columns if pd.api.types.is_numeric_dtype(res[c]) and str(c).startswith("montant")]
    if p["colonne_montant"] not in res.columns:
        if not money_cols:
            st.info("Ce jeu de projets n'a pas de colonne de montant (par exemple « montant_pret ») : "
                    "l'arbitrage budgétaire n'est pas possible. Ajoutez-la au fichier importé.")
            return
        p["colonne_montant"] = money_cols[0]

    with st.container(border=True):
        common.section("Paramètres du comité")
        c1, c2, c3 = st.columns(3)
        budget = c1.number_input("Budget global", min_value=0.0, value=float(p["budget"]), step=100_000.0,
                                 format="%.0f", help="Montant total disponible sur la période")
        mmin = c2.number_input("Montant minimum par projet", min_value=0.0, value=float(p["montant_min"]),
                               step=10_000.0, format="%.0f")
        mmax = c3.number_input("Montant maximum par projet", min_value=0.0, value=float(p["montant_max"]),
                               step=100_000.0, format="%.0f")
        c4, c5, c6 = st.columns(3)
        has_sector = p["colonne_secteur"] in res.columns
        part = c4.slider("Part maximale d'un secteur", 5, 100, int(round(100 * float(p["part_max_secteur"]))), 5,
                         format="%d %%", disabled=not has_sector,
                         help="Diversification : un secteur ne peut pas recevoir plus que cette part du budget")
        incl = c5.toggle("Compléter avec les projets « sous conditions »", value=bool(p["inclure_conditionner"]))
        mode = c6.radio("Règle de sélection", ["classement", "nombre"], index=0 if p["mode"] == "classement" else 1,
                        format_func={"classement": "Suivre le classement",
                                     "nombre": "Maximiser le nombre de projets"}.get)
    new = {**p, "budget": budget, "montant_min": mmin, "montant_max": mmax, "part_max_secteur": part / 100,
           "inclure_conditionner": incl, "mode": mode}
    if new != arbitrage.params_with_defaults(act["config"].get("arbitrage")) or "selection_budget" not in res:
        common.apply_arbitrage(new)
        act = common.active()
        res = act["ev"]["results"]
    summ = act["ev"]["arbitrage"]
    dev = act["config"].get("devise", "$")

    def money(v):                       # format compact pour les cartes : 4,98 M $
        v = float(v)
        txt = f"{v / 1e6:.2f} M" if abs(v) >= 1e6 else (f"{v / 1e3:.0f} k" if abs(v) >= 1e3 else f"{v:.0f}")
        return f"{txt} {dev}".replace(".", ",")
    st.caption("« Suivre le classement » : on finance dans l'ordre (d'abord les « Financer » par rang ELECTRE III, puis "
               "les « sous conditions ») tant que le budget, les bornes et le plafond du secteur le permettent. "
               "« Maximiser le nombre » : optimisation exacte en nombres entiers qui retient le plus de projets, quitte "
               "à écarter un projet très bien classé mais coûteux.")

    common.kpis([
        ("Budget", money(summ["budget"]), "disponible sur la période", common.GREY),
        ("Montant engagé", money(summ["montant_engage"]),
         f"{100 * summ['taux_utilisation']:.1f} % du budget".replace(".", ",")),
        ("Budget restant", money(summ["budget_restant"]), "non alloué", common.GREY),
        ("Projets retenus", summ["projets_selectionnes"],
         f"{summ['dont_financer']} à financer · {summ['dont_conditionner']} sous conditions"),
        ("Secteurs financés", summ["secteurs_finances"], "diversification"),
        ("« Financer » non retenus", summ["financer_non_retenus"], "faute de budget, de bornes ou de plafond",
         CAT_COLOR["Conditionner"]),
    ])

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        common.section("Montant engagé par secteur", f"plafond {part} % du budget")
        if summ["montant_par_secteur"]:
            sect = pd.DataFrame([{"secteur": k, "montant": v} for k, v in summ["montant_par_secteur"].items()])
            bars = alt.Chart(sect).mark_bar(color=common.GREEN, cornerRadiusEnd=3).encode(
                y=alt.Y("secteur:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
                x=alt.X("montant:Q", title="Montant engagé"),
                tooltip=[alt.Tooltip("secteur", title="Secteur"), alt.Tooltip("montant", format=",.0f", title="Montant")])
            layers = [bars]
            if has_sector and part < 100:
                layers.append(alt.Chart(pd.DataFrame({"v": [part / 100 * budget]})).mark_rule(
                    strokeDash=[5, 4], color=CAT_COLOR["Refuser"]).encode(x="v:Q"))
            st.altair_chart(alt.layer(*layers).properties(height=30 * len(sect) + 40), width="stretch")
        else:
            st.info("Aucun projet retenu avec ces paramètres.")
    with right:
        common.section("Pourquoi un projet n'est pas retenu")
        mot = res[~res["selection_budget"]]["motif_arbitrage"].value_counts().rename_axis("motif").reset_index(name="projets")
        st.altair_chart(alt.Chart(mot).mark_bar(color=common.GREY, cornerRadiusEnd=3).encode(
            y=alt.Y("motif:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("projets:Q", title="Projets", axis=alt.Axis(tickMinStep=1)),
            tooltip=["motif", "projets"]).properties(height=max(150, 36 * len(mot) + 40)), width="stretch")

    common.section("Décision projet par projet")
    view = res.assign(_o=res["categorie_finale"].map({"Financer": 0, "Conditionner": 1, "Refuser": 2}))
    view = view.sort_values(["selection_budget", "_o", "rang_electre3"], ascending=[False, True, True])
    cols = ["nom", "selection_budget", "categorie_finale", "rang_electre3", p["colonne_montant"]]
    cols += [c for c in (p["colonne_secteur"],) if c in view] + ["motif_arbitrage"]
    st.dataframe(view[cols], hide_index=True, width="stretch", height=430, column_config={
        "nom": st.column_config.TextColumn("Projet", width="medium"),
        "selection_budget": st.column_config.CheckboxColumn("Retenu"),
        "categorie_finale": "Catégorie MCDA",
        "rang_electre3": st.column_config.NumberColumn("Rang E-III", format="%d"),
        p["colonne_montant"]: st.column_config.NumberColumn("Montant", format="localized"),
        p["colonne_secteur"]: "Secteur", "motif_arbitrage": "Motif"})
    st.download_button("Télécharger la sélection (CSV)", view[cols].to_csv(index=False).encode("utf-8"),
                       "selection_budget.csv", "text/csv", icon=":material/download:")
