"""Étape 10 : application Streamlit : classement MCDA, fiche projet, modèle de risque, robustesse, méthode.

Lecture seule des fichiers produits par le pipeline (data/processed, reports, params). Aucun calcul de décision ici :
le MCDA est déterministe (src/mcda.py) et l'agent LLM ne fait que rédiger (src/agent.py).
Lancer : streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import agent  # noqa: E402
from mcda import CRITERIA, SENSE  # noqa: E402

D, R, P = ROOT / "data/processed", ROOT / "reports", ROOT / "params"
st.set_page_config(page_title="Décision d'investissement multicritère", page_icon="📊", layout="wide")

# ----------------------------------------------------------------------------------------------- style
CAT_COLOR = {"Financer": "#1a7f5a", "Conditionner": "#c98a00", "Refuser": "#b03a2e"}
CAT_ORDER = ["Financer", "Conditionner", "Refuser"]
LABEL = {"pd_ml": "Probabilité de défaut", "van": "VAN", "delai_recuperation": "Délai de récupération",
         "dscr_min": "DSCR minimum", "dette_fonds_propres": "Dette / fonds propres",
         "couverture_garanties": "Couverture garanties", "esg_score": "Score ESG",
         "emplois_par_million": "Emplois par M$"}
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1350px;}
.hero {background: linear-gradient(120deg, #1f4e79 0%, #2e6f9e 60%, #3f8fb5 100%); color: #fff;
       padding: 22px 28px; border-radius: 14px; margin-bottom: 18px;}
.hero h1 {color: #fff; font-size: 1.7rem; margin: 0 0 4px 0; padding: 0;}
.hero p {margin: 0; opacity: .92;}
.chips {margin-top: 12px;}
.chip {display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.35);
       border-radius: 999px; padding: 3px 12px; margin: 0 6px 6px 0; font-size: .86rem;}
.badge {display: inline-block; color: #fff; border-radius: 8px; padding: 4px 12px; font-weight: 600; font-size: 1rem;}
.muted {color: #5b6675; font-size: .9rem;}
.step {border-left: 4px solid #1f4e79; padding: 6px 12px; margin-bottom: 8px; background: #fff; border-radius: 6px;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #e3e7ee; border-radius: 12px; padding: 10px 14px;}
</style>
""", unsafe_allow_html=True)


def md(text: str) -> str:
    """Échappe « $ » (sinon Streamlit l'interprète comme une formule LaTeX)."""
    return str(text).replace("$", "\\$")


def badge(cat: str) -> str:
    txt = {"Financer": "Financer", "Conditionner": "Financer sous conditions", "Refuser": "Refuser"}[cat]
    return f"<span class='badge' style='background:{CAT_COLOR[cat]}'>{txt}</span>"


# ----------------------------------------------------------------------------------------------- données
@st.cache_data
def load():
    res = pd.read_csv(D / "mcda_resultats.csv")
    sim = pd.read_csv(D / "simulation_detail.csv")[["loan_id", "secteur_damodaran", "etat", "montant_pret",
                                                     "investissement", "duree_standard_annees"]]
    expl = pd.read_csv(D / "explications_projets.csv")[["loan_id", "facteurs_risque"]]
    df = res.merge(sim, on="loan_id").merge(expl, on="loan_id")
    seed_file = D / "sensibilite_graine_projets.csv"
    if seed_file.exists():
        df = df.merge(pd.read_csv(seed_file)[["loan_id", "part_graines_meme_categorie"]], on="loan_id", how="left")
    df["pd_pct"] = 100 * df["pd_ml"]
    return df


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


df = load()
filtre = read_json(P / "filtre_eliminatoire.json")
coh = read_json(P / "ahp_coherence.json")
profils = pd.read_csv(P / "profils_electre_tri.csv").set_index("critere")

# ----------------------------------------------------------------------------------------------- barre latérale
with st.sidebar:
    st.markdown("### Filtres")
    cats = st.multiselect("Catégorie", CAT_ORDER, default=CAT_ORDER)
    show_out = st.toggle("Afficher les projets écartés par le filtre", value=True)
    secteurs = st.multiselect("Secteur", sorted(df.secteur_damodaran.unique()))
    st.divider()
    st.markdown("### Rôles")
    st.markdown("**ML** estime le risque de défaut  \n**MCDA** classe et décide  \n**LLM** rédige la note  \n"
                "**Comité** prend la décision finale")
    st.divider()
    llm_on = bool(agent._secret("GEMINI_API_KEY") or agent._secret("GROQ_API_KEY"))
    st.markdown("### Agent rédacteur")
    st.markdown("🟢 LLM configuré" if llm_on else "⚪ Pas de clé : note de gabarit")

view = df[df.categorie_finale.isin(cats)]
if not show_out:
    view = view[~view.ecarte]
if secteurs:
    view = view[view.secteur_damodaran.isin(secteurs)]

# ----------------------------------------------------------------------------------------------- en-tête
st.markdown("""
<div class="hero">
  <h1>IA agentique pour la décision d'investissement multicritère</h1>
  <p>40 demandes de crédit d'investissement de PME (prêts SBA 7(a) réels, indicateurs financiers simulés)</p>
  <div class="chips"><span class="chip">① Modèle ML : probabilité de défaut</span>
  <span class="chip">② Filtre éliminatoire</span><span class="chip">③ ELECTRE III / TRI, PROMETHEE II</span>
  <span class="chip">④ Agent LLM : rédaction vérifiée</span><span class="chip">⑤ Décision du comité</span></div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Vue d'ensemble", "🔎 Fiche projet", "🧠 Modèle de risque",
                                        "🧪 Robustesse", "📐 Méthode"])

# ============================================================================================ 1. vue d'ensemble
with tab1:
    k = st.columns(6)
    k[0].metric("Projets candidats", len(df))
    k[1].metric("🟢 Financer", int((df.categorie_finale == "Financer").sum()))
    k[2].metric("🟠 Sous conditions", int((df.categorie_finale == "Conditionner").sum()))
    k[3].metric("🔴 Refuser", int((df.categorie_finale == "Refuser").sum()))
    k[4].metric("⛔ Écartés", int(df.ecarte.sum()), help="Filtre éliminatoire (DSCR < 1 ou risque trop élevé)")
    k[5].metric("⚠️ Fragiles", int(df.decision_fragile.sum()),
                help="Catégorie différente dans au moins 25 % des scénarios MCDA")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### Risque et capacité de remboursement")
        base = alt.Chart(view).mark_circle(opacity=.85, stroke="#fff", strokeWidth=1).encode(
            x=alt.X("pd_pct:Q", title="Probabilité de défaut (%)"),
            y=alt.Y("dscr_min:Q", title="DSCR minimum", scale=alt.Scale(zero=False)),
            size=alt.Size("montant_pret:Q", title="Montant ($)", scale=alt.Scale(range=[60, 700]), legend=None),
            color=alt.Color("categorie_finale:N", title="Décision",
                            scale=alt.Scale(domain=CAT_ORDER, range=[CAT_COLOR[c] for c in CAT_ORDER])),
            tooltip=[alt.Tooltip("loan_id", title="Projet"), alt.Tooltip("categorie_finale", title="Décision"),
                     alt.Tooltip("secteur_damodaran", title="Secteur"), alt.Tooltip("pd_pct", format=".1f", title="pd (%)"),
                     alt.Tooltip("dscr_min", format=".2f", title="DSCR"), alt.Tooltip("van", format=",.0f", title="VAN"),
                     alt.Tooltip("motif_ecarte", title="Motif d'exclusion")])
        rules = [alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(strokeDash=[5, 4], color="#b03a2e").encode(y="y:Q")]
        if filtre:
            rules.append(alt.Chart(pd.DataFrame({"x": [100 * filtre["seuil_pd"]]}))
                         .mark_rule(strokeDash=[5, 4], color="#b03a2e").encode(x="x:Q"))
        st.altair_chart(alt.layer(base, *rules).properties(height=360), width="stretch")
        st.caption("Lignes pointillées : seuils éliminatoires (DSCR = 1"
                   + (f", probabilité de défaut = {100 * filtre['seuil_pd']:.1f} %" if filtre else "") + ").")
    with right:
        st.markdown("#### Classement ELECTRE III (projets retenus)")
        top = view[~view.ecarte].sort_values("rang_electre3")
        bars = alt.Chart(top).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("loan_id:N", sort=None, title=None),
            x=alt.X("credibilite_nette:Q", title="Crédibilité nette (surclasse − est surclassé)"),
            color=alt.Color("categorie_finale:N", legend=None,
                            scale=alt.Scale(domain=CAT_ORDER, range=[CAT_COLOR[c] for c in CAT_ORDER])),
            tooltip=["loan_id", "rang_electre3", "rang_promethee2", "categorie_finale"])
        st.altair_chart(bars.properties(height=max(360, 13 * len(top))), width="stretch")

    st.markdown("#### Tableau des projets")
    table = view.sort_values(["ecarte", "rang_electre3"])
    st.dataframe(
        table[["loan_id", "categorie_finale", "rang_electre3", "rang_promethee2", "pd_ml", "dscr_min", "van",
               "delai_recuperation", "dette_fonds_propres", "couverture_garanties", "esg_score",
               "emplois_par_million", "decision_fragile", "position_ambigue", "motif_ecarte"]],
        hide_index=True, width="stretch", height=430,
        column_config={
            "loan_id": "Projet", "categorie_finale": "Décision",
            "rang_electre3": st.column_config.NumberColumn("Rang ELECTRE III", format="%d"),
            "rang_promethee2": st.column_config.NumberColumn("Rang PROMETHEE II", format="%d"),
            "pd_ml": st.column_config.ProgressColumn("Proba. défaut", format="percent", min_value=0,
                                                     max_value=float(df.pd_ml.max())),
            "dscr_min": st.column_config.NumberColumn("DSCR min", format="%.2f"),
            "van": st.column_config.NumberColumn("VAN ($)", format="localized"),
            "delai_recuperation": st.column_config.NumberColumn("Délai (ans)", format="%.1f"),
            "dette_fonds_propres": st.column_config.NumberColumn("Dette/FP", format="%.2f"),
            "couverture_garanties": st.column_config.NumberColumn("Garanties", format="%.2f"),
            "esg_score": st.column_config.NumberColumn("ESG", format="%.0f"),
            "emplois_par_million": st.column_config.NumberColumn("Emplois/M$", format="%.1f"),
            "decision_fragile": st.column_config.CheckboxColumn("Fragile"),
            "position_ambigue": st.column_config.CheckboxColumn("Rang ambigu"),
            "motif_ecarte": "Motif d'exclusion"})
    st.caption("Décision = ELECTRE TRI pessimiste (λ = 0,6). « Fragile » : la catégorie change dans ≥ 25 % des "
               "scénarios de poids, seuils et coupure. « Rang ambigu » : les deux distillations d'ELECTRE III placent "
               "le projet très différemment (incomparable à une partie des autres).")

# ============================================================================================ 2. fiche projet
with tab2:
    order = df.sort_values(["ecarte", "rang_electre3"])
    pid = st.selectbox("Projet", order.loan_id,
                       format_func=lambda i: f"{i} · {df.set_index('loan_id').loc[i, 'categorie_finale']}")
    row = df[df.loan_id == pid].iloc[0]

    h1, h2 = st.columns([3, 2])
    with h1:
        st.markdown(f"### Projet `{pid}` &nbsp; {badge(row.categorie_finale)}", unsafe_allow_html=True)
        st.markdown(md(f"<span class='muted'>{row.secteur_damodaran} · {row.etat} · prêt de {row.montant_pret:,.0f} $ "
                       f"· investissement simulé {row.investissement:,.0f} $ · durée standard "
                       f"{row.duree_standard_annees:.0f} ans</span>"), unsafe_allow_html=True)
    with h2:
        if row.ecarte:
            st.error(f"Écarté par le filtre : {row.motif_ecarte}")
        else:
            st.markdown(f"**Rang ELECTRE III** : {row.rang_electre3:.0f} / {int((~df.ecarte).sum())} · "
                        f"**PROMETHEE II** : {row.rang_promethee2:.0f}")
            if row.position_ambigue:
                st.warning("Rang ambigu : les deux distillations d'ELECTRE III divergent "
                           f"({row.rang_distillation_descendante:.0f} contre {row.rang_distillation_ascendante:.0f}).")

    m = st.columns(4)
    m[0].metric("Probabilité de défaut", f"{row.pd_pct:.1f} %")
    m[1].metric("DSCR minimum", f"{row.dscr_min:.2f}")
    m[2].metric("VAN", f"{row.van:,.0f} $")
    m[3].metric("Délai de récupération", f"{row.delai_recuperation:.1f} ans")

    a, b = st.columns([1.15, 1])
    with a:
        st.markdown("#### Position par rapport aux seuils de décision")
        recs, pos = [], []
        for c in CRITERIA:
            vals = (SENSE[c] * df[c]).to_numpy(float)

            def rank(x, vals=vals, s=SENSE[c]):   # rang percentile dans les 40 projets (robuste aux extrêmes)
                return float(((vals < s * x).sum() + 0.5 * (vals == s * x).sum()) / len(vals))
            b1 = rank(profils.loc[c, "b1_refuser_conditionner"])
            b2 = rank(profils.loc[c, "b2_conditionner_financer"])
            recs += [{"critere": LABEL[c], "zone": "Refuser", "x0": 0.0, "x1": b1},
                     {"critere": LABEL[c], "zone": "Conditionner", "x0": b1, "x1": b2},
                     {"critere": LABEL[c], "zone": "Financer", "x0": b2, "x1": 1.0}]
            pos.append(rank(row[c]))
        zones = pd.DataFrame(recs)
        pts = pd.DataFrame({"critere": [LABEL[c] for c in CRITERIA], "pos": pos,
                            "valeur": [f"{100 * row[c]:.1f} %" if c == "pd_ml" else f"{row[c]:,.2f}" for c in CRITERIA]})
        ylab = alt.Y("critere:N", sort=[LABEL[c] for c in CRITERIA], title=None, axis=alt.Axis(labelLimit=220))
        zl = alt.Chart(zones).mark_bar(height=16, opacity=.28).encode(
            y=ylab, x=alt.X("x0:Q", title="← moins bon · rang parmi les 40 projets · meilleur →",
                            scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(labels=False, ticks=False)),
            x2="x1:Q", color=alt.Color("zone:N", title="Zone ELECTRE TRI",
                                       scale=alt.Scale(domain=CAT_ORDER, range=[CAT_COLOR[c] for c in CAT_ORDER])))
        pl = alt.Chart(pts).mark_point(filled=True, size=170, color="#1c2430", shape="diamond").encode(
            y=ylab, x="pos:Q", tooltip=[alt.Tooltip("critere", title="Critère"), alt.Tooltip("valeur", title="Valeur")])
        st.altair_chart((zl + pl).properties(height=300), width="stretch")
        st.caption("Bandes : zones délimitées par les profils ELECTRE TRI b1 et b2 (voir onglet Méthode). "
                   "Losange : le projet. Les critères à minimiser sont retournés (à droite = meilleur).")
        with st.expander("Valeurs détaillées et facteurs de risque"):
            st.dataframe(pd.DataFrame({
                "Critère": [LABEL[c] for c in CRITERIA],
                "Valeur": [f"{100 * row.pd_ml:.1f} %", f"{row.van:,.0f} $", f"{row.delai_recuperation:.1f} ans",
                           f"{row.dscr_min:.2f}", f"{row.dette_fonds_propres:.2f}", f"{row.couverture_garanties:.2f}",
                           f"{row.esg_score:.0f}", f"{row.emplois_par_million:.1f}"],
                "Seuil b1 → b2": [f"{profils.loc[c, 'b1_refuser_conditionner']:,.3g} → "
                                  f"{profils.loc[c, 'b2_conditionner_financer']:,.3g}" for c in CRITERIA]}),
                hide_index=True, width="stretch")
            st.markdown(f"**Facteurs qui augmentent le risque (EBM)** : {row.facteurs_risque}")
        r1, r2 = st.columns(2)
        r1.metric("Scénarios MCDA où la catégorie change", f"{row.part_scenarios_categorie_differente:.0%}",
                  help="200 scénarios : poids ± 30 %, seuils q/p/v, coupure λ entre 0,5 et 0,8")
        if "part_graines_meme_categorie" in row and pd.notna(row.part_graines_meme_categorie):
            r2.metric("Simulations donnant la même catégorie", f"{row.part_graines_meme_categorie:.0%}",
                      help="20 tirages des indicateurs financiers simulés pour ce même prêt")

    with b:
        st.markdown("#### Note de l'agent")
        if st.button("✍️ Rédiger la note avec le LLM", type="primary", width="stretch"):
            with st.spinner("Rédaction puis vérification des chiffres…"):
                st.session_state[f"note_{pid}"] = agent.write_note(row.to_dict())
        res = st.session_state.get(f"note_{pid}")
        if res is None:
            res = {"note": agent.template_note(agent.build_facts(row.to_dict())), "source": "gabarit",
                   "verification": "note par défaut (cliquer sur le bouton pour faire rédiger le LLM)"}
        with st.container(border=True):
            st.markdown(md(res["note"]))
        ok = res["source"].startswith("LLM")
        st.markdown(f"{'✅' if ok else 'ℹ️'} **Source** : {res['source']}  \n**Contrôle** : {res['verification']}")
        st.info("L'agent recopie les résultats déjà calculés ; un vérificateur rejette toute note contenant un "
                "chiffre absent des données. La décision finale revient au comité.", icon="🛡️")

# ============================================================================================ 3. modèle
with tab3:
    test = pd.read_csv(R / "metriques_test.csv")
    ebm = test[(test.modele == "ebm") & (test.version == "calibre")].iloc[0]
    lr = test[(test.modele == "logreg") & (test.version == "calibre")].iloc[0]
    k = st.columns(4)
    k[0].metric("AUROC test (EBM)", f"{ebm.AUROC:.3f}", f"{ebm.AUROC - lr.AUROC:+.3f} vs logistique")
    k[1].metric("AUPRC test", f"{ebm.AUPRC:.3f}", f"{ebm.AUPRC - lr.AUPRC:+.3f} vs logistique")
    k[2].metric("Brier test", f"{ebm.Brier:.4f}", f"{ebm.Brier - lr.Brier:+.4f}", delta_color="inverse")
    k[3].metric(f"pd moyenne (réel {100 * ebm.taux_defaut_reel:.2f} %)", f"{100 * ebm.pd_moyenne:.2f} %",
                f"{100 * (ebm.pd_moyenne - ebm.taux_defaut_reel):+.2f} pt", delta_color="off")
    fmt = {"AUROC": "%.4f", "AUPRC": "%.4f", "F1": "%.4f", "Brier": "%.4f", "taux_defaut_reel": "%.4f",
           "pd_moyenne": "%.4f", "seuil_F1": "%.4f"}
    cc = {c: st.column_config.NumberColumn(c, format=f) for c, f in fmt.items()}
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Validation (2013-2014) : choix du modèle")
        st.dataframe(pd.read_csv(R / "metriques_validation.csv"), hide_index=True, width="stretch", column_config=cc)
    with c2:
        st.markdown("#### Test (2015-2016) : lu une seule fois")
        st.dataframe(test, hide_index=True, width="stretch", column_config=cc)
    hp = R / "reglage_hyperparametres.csv"
    if hp.exists():
        with st.expander("Réglage des hyperparamètres (sur la validation)"):
            st.dataframe(pd.read_csv(hp).round(4), hide_index=True, width="stretch")
    st.image(str(R / "calibration.png"), caption="Calibration : logistique vs EBM (validation puis test)")
    i1, i2 = st.columns(2)
    i1.image(str(R / "importance_variables.png"), caption="Importance native EBM et par permutation (validation)")
    i2.image(str(R / "ebm_effets.png"), caption="Effet des variables principales (EBM)")
    with st.expander("Erreurs par secteur, État et procédure (validation)"):
        st.dataframe(pd.read_csv(R / "erreurs_par_segment_validation.csv").round(3), hide_index=True, width="stretch")

# ============================================================================================ 4. robustesse
with tab4:
    sens = pd.read_csv(D / "sensibilite_seuils_coupure.csv")
    meth = pd.read_csv(D / "comparaison_methodes.csv").iloc[0]
    k = st.columns(4)
    k[0].metric("Spearman E-III / P-II", f"{meth.spearman:.3f}",
                help="Corrélation de rangs entre ELECTRE III et PROMETHEE II")
    k[1].metric("Kendall E-III / P-II", f"{meth.kendall:.3f}")
    k[2].metric("Décisions fragiles (MCDA)", f"{int(df.decision_fragile.sum())} / {len(df)}")
    if "part_graines_meme_categorie" in df:
        k[3].metric("Stabilité moyenne à la simulation", f"{df.part_graines_meme_categorie.mean():.0%}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Effet de la coupure λ (ELECTRE TRI)")
        lam = sens[sens.parametre == "coupure lambda"].assign(valeur=lambda d: d.valeur.astype(float))
        st.altair_chart(alt.Chart(lam).mark_line(point=True, color="#1f4e79").encode(
            x=alt.X("valeur:Q", title="Coupure λ"), y=alt.Y("projets_categorie_change:Q",
                                                         title="Projets qui changent de catégorie"),
            tooltip=["valeur", "projets_categorie_change", alt.Tooltip("spearman_vs_base", format=".3f")]
        ).properties(height=280), width="stretch")
        st.caption("Les scénarios combinés limitent λ à 0,5-0,8 ; 0,9-1 (quasi-unanimité) est montré ici comme "
                   "cas extrême.")
    with c2:
        st.markdown("#### ELECTRE III contre PROMETHEE II")
        al = df[~df.ecarte]
        st.altair_chart(alt.Chart(al).mark_circle(size=90).encode(
            x=alt.X("rang_electre3:Q", title="Rang ELECTRE III"), y=alt.Y("rang_promethee2:Q", title="Rang PROMETHEE II"),
            color=alt.Color("categorie_finale:N", legend=None,
                            scale=alt.Scale(domain=CAT_ORDER, range=[CAT_COLOR[c] for c in CAT_ORDER])),
            tooltip=["loan_id", "rang_electre3", "rang_promethee2"]).properties(height=280), width="stretch")

    st.markdown("#### Stabilité de chaque décision")
    stab = df[["loan_id", "categorie_finale", "part_scenarios_categorie_differente"]].copy()
    stab["stab_mcda"] = 1 - stab.part_scenarios_categorie_differente
    cols = [("stab_mcda", "Face aux paramètres MCDA (200 scénarios)")]
    if "part_graines_meme_categorie" in df:
        stab["stab_sim"] = df.part_graines_meme_categorie
        cols.append(("stab_sim", "Face au tirage des indicateurs simulés (20 graines)"))
    for (col, title), box in zip(cols, st.columns(len(cols))):
        with box:
            st.markdown(f"**{title}** · part des cas où la catégorie est inchangée")
            st.altair_chart(alt.Chart(stab).mark_bar(cornerRadiusEnd=2).encode(
                x=alt.X("loan_id:N", sort=alt.EncodingSortField(col, order="descending"), title=None,
                        axis=alt.Axis(labels=False, ticks=False)),
                y=alt.Y(f"{col}:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("categorie_finale:N", title="Décision",
                                scale=alt.Scale(domain=CAT_ORDER, range=[CAT_COLOR[c] for c in CAT_ORDER])),
                tooltip=[alt.Tooltip("loan_id", title="Projet"), alt.Tooltip("categorie_finale", title="Décision"),
                         alt.Tooltip(f"{col}:Q", format=".0%", title="Part identique")]
            ).properties(height=220), width="stretch")
    st.caption("Barre haute = décision robuste. Une décision instable face au tirage des indicateurs simulés dépend "
               "de la simulation plus que du projet : à interpréter avec prudence.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Poids AHP")
        w = pd.read_csv(P / "poids_ahp.csv")
        wl = w.melt(id_vars="critere", value_vars=["poids_ahp", "poids_cible"], var_name="type", value_name="poids")
        st.altair_chart(alt.Chart(wl).mark_bar().encode(
            y=alt.Y("critere:N", sort="-x", title=None), x=alt.X("poids:Q", axis=alt.Axis(format="%")),
            color=alt.Color("type:N", scale=alt.Scale(range=["#1f4e79", "#9fb7cc"]), title=None),
            yOffset="type:N", tooltip=["critere", "type", alt.Tooltip("poids", format=".3f")]
        ).properties(height=300), width="stretch")
        st.caption(f"Ratio de cohérence AHP : {coh.get('CR', float('nan')):.4f} (< 0,10). Jugements : "
                   "params/ahp_jugements.csv.")
    with c4:
        st.markdown("#### Un paramètre à la fois")
        st.dataframe(sens.round(3), hide_index=True, width="stretch")

# ============================================================================================ 5. méthode
with tab5:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Chaîne de décision")
        for t, d in [("1. Données réelles", "Prêts SBA 7(a) FOIA ; cible `default_60m` (perte ≤ 60 mois après le "
                      "premier versement) ; séparation temporelle 2000-2012 / 2013-2014 / 2015-2016."),
                     ("2. Modèle de risque", "Régression logistique et EBM, mêmes variables, calibration isotonique "
                      "sur la validation ; EBM retenu. Sorties : `pd_ml` et facteurs de risque."),
                     ("3. Simulation", "Pour 40 prêts du test : VAN, délai, DSCR, dette/fonds propres, garanties, "
                      "ESG (paramètres Damodaran + hypothèses d'équipe). Indépendante de l'issue réelle."),
                     ("4. MCDA", "Poids AHP, filtre éliminatoire, ELECTRE III (classement), ELECTRE TRI "
                      "(catégorie), PROMETHEE II (comparaison), analyses de sensibilité."),
                     ("5. Agent LLM", "Rédige une note à partir des seuls chiffres calculés ; un vérificateur "
                      "rejette tout chiffre inventé."),
                     ("6. Comité", "Décision finale humaine.")]:
            st.markdown(f"<div class='step'><b>{t}</b><br><span class='muted'>{d}</span></div>",
                        unsafe_allow_html=True)
    with c2:
        st.markdown("#### Filtre éliminatoire")
        if filtre:
            st.markdown(f"- DSCR minimum < **{filtre['dscr_min_plancher']:g}**\n"
                        f"- Probabilité de défaut > **{100 * filtre['seuil_pd']:.1f} %** "
                        f"(2 × {100 * filtre['taux_defaut_reference']:.2f} %, taux de la validation, régime sur "
                        f"lequel `pd_ml` est calibrée ; le taux 2000-2012 de "
                        f"{100 * filtre['taux_entrainement_non_utilise']:.1f} %, gonflé par la crise de 2008, rendrait "
                        "ce filtre inopérant)")
        post = D / "controle_a_posteriori.csv"
        if post.exists():
            st.markdown("#### Contrôle a posteriori (illustration)")
            st.dataframe(pd.read_csv(post), hide_index=True, width="stretch")
            st.caption("Vrais défauts à 60 mois parmi les 40 projets, lus APRÈS les décisions et jamais utilisés dans "
                       "un calcul. Quelques cas seulement : aucune valeur de preuve.")
    st.markdown("#### Profils ELECTRE TRI (seuils entre catégories)")
    st.dataframe(profils.reset_index().assign(critere=lambda d: d.critere.map(LABEL)),
                 hide_index=True, width="stretch",
                 column_config={"critere": "Critère", "type": "Type",
                                "b1_refuser_conditionner": st.column_config.NumberColumn("b1 Refuser | Conditionner",
                                                                                         format="%.4g"),
                                "b2_conditionner_financer": st.column_config.NumberColumn("b2 Conditionner | Financer",
                                                                                          format="%.4g"),
                                "justification": st.column_config.TextColumn("Justification", width="large")})
    st.markdown("#### Limites connues")
    st.markdown(
        "- Les indicateurs financiers sont **simulés** (aucune donnée ouverte n'associe VAN/DSCR à l'issue d'un prêt) ; "
        "plusieurs hypothèses d'équipe ne sont pas sourcées (croissance, horizon, valeur résiduelle, ESG).\n"
        "- Le filtre DSCR découle de la simulation : la tranche « faible » du guide (0,8-1,0) est toujours sous 1, "
        "donc ses 8 projets sont écartés par construction.\n"
        "- La durée exacte des prêts (`TermInMonths`) est altérée après l'octroi pour les prêts en défaut : seules "
        "3 tranches sont utilisées, dans le modèle comme dans la simulation.\n"
        "- Seuils q/p/v et certains profils sont relatifs au lot de projets : ajouter un projet peut les déplacer.\n"
        "- Le vérificateur de l'agent contrôle les nombres, pas le sens des phrases : relecture humaine nécessaire.")
