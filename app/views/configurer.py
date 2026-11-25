"""Configurer l'analyse : importer un jeu de projets et régler toute la méthode, puis lancer l'évaluation."""
import io

import altair as alt
import pandas as pd
import streamlit as st

import common
import engine
import mcda

SAATY = ["9", "8", "7", "6", "5", "4", "3", "2", "1", "1/2", "1/3", "1/4", "1/5", "1/6", "1/7", "1/8", "1/9"]


def _inverse(j: str | None) -> str | None:
    """« 3 » -> « 1/3 », « 1/3 » -> « 3 », « 1 » -> « 1 »."""
    if not j:
        return None
    j = str(j).strip()
    return j.split("/")[1] if j.startswith("1/") else ("1" if j == "1" else f"1/{j}")


def _load_draft(df: pd.DataFrame, cfg: dict, source: str):
    st.session_state.update(draft_df=df, draft_cfg=cfg, draft_src=source,
                            draft_ver=st.session_state.get("draft_ver", 0) + 1)


def _candidates(df: pd.DataFrame, configured: list[str]) -> list[str]:
    """Critères possibles : ceux de la configuration d'abord, puis les autres colonnes numériques non techniques."""
    skip = {"id", "nom", "part_graines_meme_categorie", "secteur_naics2"} | set(engine.MODEL_COLUMNS)
    others = [c for c in df.columns if c not in skip and c not in configured and not c.startswith(engine.INFO_PREFIXES)
              and pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in configured if c in df.columns] + others


def _step(n: int, title: str, help_txt: str = ""):
    st.markdown(f"<div class='sec' style='border:none;padding-left:0;margin-top:26px'><span class='stepnum'>{n}</span>"
                f"{title}</div>", unsafe_allow_html=True)
    if help_txt:
        st.caption(help_txt)


def render():
    act = common.active()
    if "draft_df" not in st.session_state:
        _load_draft(act["df"], act["config"], "demo" if act["demo"] else "import")
    common.page_header("Outil réutilisable", "Configurer l'analyse",
                       "Importez vos propres projets et réglez chaque paramètre de la méthode ; toutes les pages se "
                       "mettent ensuite à jour sur cette nouvelle analyse.",
                       ["Données", "Critères et seuils", "Pondération", "Filtre", "ELECTRE", "Lancer"])
    ver = st.session_state["draft_ver"]
    if st.session_state.pop("launched", False):
        res = act["ev"]["results"]["categorie_finale"].value_counts()
        st.success(f"Analyse lancée : {res.get('Financer', 0)} Financer, {res.get('Conditionner', 0)} Conditionner, "
                   f"{res.get('Refuser', 0)} Refuser. Toutes les pages affichent maintenant ces résultats.")
        st.page_link(common.PAGES["tableau-de-bord"], label="Voir le tableau de bord", icon="📋")

    # ------------------------------------------------------------------------------------------ 1. données
    _step(1, "Jeu de projets", "Une ligne par projet. Colonnes conseillées : « id », « nom », puis une colonne numérique "
          "par critère. Si le fichier contient les champs d'un prêt SBA (GrossApproval, TermInMonths, NaicsCode…) "
          "et pas de colonne « pd_ml », la probabilité de défaut est calculée par le modèle de risque.")
    c1, c2, c3 = st.columns([1.3, 1.3, 1])
    with c1:
        if st.button("Charger la démonstration (40 prêts SBA)", width="stretch", icon=":material/inventory_2:"):
            df, cfg = common.demo_data()
            _load_draft(df, cfg, "demo")
            st.rerun()
        st.download_button("Modèle de fichier (CSV)", engine.template_projects().to_csv(index=False).encode("utf-8"),
                           "modele_projets.csv", "text/csv", width="stretch",
                           icon=":material/download:")
    with c2:
        up = st.file_uploader("Importer des projets (CSV ou Excel)", type=["csv", "xlsx", "xls"], key="up_data")
        if up is not None and st.session_state.get("last_upload") != (up.name, up.size):
            try:
                raw = (pd.read_csv(up, sep=None, engine="python") if up.name.lower().endswith(".csv")
                       else pd.read_excel(up))
                had_pd = "pd_ml" in raw.columns
                df = engine.prepare_projects(raw)
                cfg = engine.guess_config(df, name=f"Analyse : {up.name}")
                _load_draft(df, cfg, "import")
                st.session_state["last_upload"] = (up.name, up.size)
                if "pd_ml" in df and not had_pd:
                    st.session_state["upload_msg"] = "Probabilité de défaut calculée par le modèle de risque (pd_ml)."
                st.rerun()
            except Exception as e:  # fichier illisible : message clair, pas de plantage
                st.error(f"Fichier illisible : {e}")
    with c3:
        upc = st.file_uploader("Charger une configuration (JSON)", type=["json"], key="up_cfg")
        if upc is not None and st.session_state.get("last_cfg") != (upc.name, upc.size):
            try:
                _load_draft(st.session_state["draft_df"], engine.config_from_json(upc.read().decode("utf-8")),
                            st.session_state["draft_src"])
                st.session_state["last_cfg"] = (upc.name, upc.size)
                st.rerun()
            except Exception as e:
                st.error(f"Configuration invalide : {e}")
    if msg := st.session_state.pop("upload_msg", None):
        st.success(msg)
    df, base_cfg = st.session_state["draft_df"], st.session_state["draft_cfg"]
    src = "démonstration" if st.session_state["draft_src"] == "demo" else "fichier importé"
    st.markdown(f"**{len(df)} projets** ({src}) · aperçu :")
    st.dataframe(df.head(8), hide_index=True, width="stretch", height=200)
    name = st.text_input("Nom de l'analyse", base_cfg.get("nom", "Mon analyse"), key=f"name_{ver}")

    # ------------------------------------------------------------------------------------------ 2. critères
    _step(2, "Critères, sens, poids et seuils de catégorie",
          "Cochez les critères à utiliser. Sens : « max » si plus grand = meilleur, « min » sinon. b1 sépare Refuser et "
          "Conditionner, b2 sépare Conditionner et Financer (dans les unités du critère). Après un changement de sens, "
          "recalculez les profils.")
    in_cfg = {c["colonne"]: c for c in base_cfg["criteres"]}
    cand = _candidates(df, list(in_cfg))
    num = [c for c in cand if c not in in_cfg]
    sense = {c: -1 if any(h in c.lower() for h in engine.MIN_HINTS) else 1 for c in num}
    prof = mcda.tertile_profiles(df, sense) if num else {}
    g = {c: {"libelle": c.replace("_", " "), "sens": "max" if sense[c] > 0 else "min", "poids": 0.0,
             "format": "dec2", "b1": prof[c][0], "b2": prof[c][1]} for c in num}
    base = pd.DataFrame([{**{"utiliser": col in in_cfg}, "colonne": col,
                          **{k: (in_cfg.get(col) or g[col])[k] for k in ("libelle", "sens", "poids", "format", "b1", "b2")}}
                         for col in cand if col in in_cfg or col in g])
    edited = st.data_editor(
        base, key=f"crit_{ver}", hide_index=True, width="stretch", disabled=["colonne"],
        column_config={"utiliser": st.column_config.CheckboxColumn("Utiliser", width="small"),
                       "colonne": "Colonne", "libelle": st.column_config.TextColumn("Libellé", width="medium"),
                       "sens": st.column_config.SelectboxColumn("Sens", options=["max", "min"], required=True),
                       "poids": st.column_config.NumberColumn("Poids", min_value=0.0, step=0.01, format="%.3f"),
                       "format": st.column_config.SelectboxColumn("Format", options=list(engine.FORMATS), required=True),
                       "b1": st.column_config.NumberColumn("b1 Refuser | Conditionner", format="%.4g"),
                       "b2": st.column_config.NumberColumn("b2 Conditionner | Financer", format="%.4g")})
    sel = edited[edited["utiliser"]]
    crit = [{k: (float(r[k]) if k in ("poids", "b1", "b2") else r[k]) for k in
             ("colonne", "libelle", "sens", "poids", "format", "b1", "b2")} for _, r in sel.iterrows()]
    if st.button("Recalculer les profils b1/b2 (tertiles du jeu, selon le sens)", icon=":material/refresh:") and crit:
        prof = mcda.tertile_profiles(df, {c["colonne"]: 1 if c["sens"] == "max" else -1 for c in crit})
        for c in crit:
            c["b1"], c["b2"] = prof[c["colonne"]]
        _load_draft(df, {**base_cfg, "criteres": crit, "nom": name}, st.session_state["draft_src"])
        st.rerun()

    # ------------------------------------------------------------------------------------------ 3. poids
    _step(3, "Pondération", "Poids saisis directement (normalisés à 100 %) ou méthode AHP : comparez les critères deux "
          "à deux (« 3 » = A modérément plus important que B ; « 1/3 » = l'inverse) ; le ratio de cohérence doit rester "
          "inférieur à 0,10.")
    mode = st.radio("Mode", ["manuel", "ahp"], horizontal=True, key=f"mode_{ver}",
                    index=0 if base_cfg.get("mode_poids") != "ahp" else 1,
                    format_func={"manuel": "Poids saisis", "ahp": "AHP (comparaisons deux à deux)"}.get)
    cols = [c["colonne"] for c in crit]
    labels = {c["colonne"]: c["libelle"] for c in crit}
    judg = []
    weights, cr = None, None
    if mode == "ahp" and len(cols) >= 2:
        known = {(j["critere_a"], j["critere_b"]): j["jugement"] for j in base_cfg.get("ahp_jugements", [])}
        tw = {c["colonne"]: max(c["poids"], 1e-3) for c in crit}
        prop = mcda.judgments_table(mcda.ahp_matrix_from_targets(tw))
        rows = []
        for _, r in prop.iterrows():
            a, b = r["critere_a"], r["critere_b"]
            v = known.get((a, b)) or _inverse(known.get((b, a))) or r["jugement"]   # jugement déjà saisi, sinon proposé
            rows.append({"critere_a": a, "critere_b": b, "jugement": v if v in SAATY else r["jugement"]})
        ahp_base = pd.DataFrame(rows).assign(A=lambda d: d.critere_a.map(labels), B=lambda d: d.critere_b.map(labels))
        jed = st.data_editor(ahp_base[["A", "jugement", "B", "critere_a", "critere_b"]],
                             key=f"ahp_{ver}_{'-'.join(cols)}", hide_index=True, width="stretch",
                             disabled=["A", "B", "critere_a", "critere_b"],
                             column_config={"A": "Critère A", "B": "Critère B",
                                            "jugement": st.column_config.SelectboxColumn(
                                                "Importance de A par rapport à B", options=SAATY, required=True),
                                            "critere_a": None, "critere_b": None})
        judg = jed[["critere_a", "critere_b", "jugement"]].to_dict("records")
        try:
            weights, _, cr = mcda.ahp_weights(mcda.ahp_matrix_from_judgments(pd.DataFrame(judg), cols))
        except Exception as e:
            st.error(f"AHP : {e}")
    elif cols:
        w = pd.Series({c["colonne"]: c["poids"] for c in crit})
        weights = w / w.sum() if w.sum() > 0 else w
    if weights is not None and len(weights):
        wdf = pd.DataFrame({"critere": [labels[c] for c in weights.index], "poids": weights.values})
        a, b = st.columns([3, 1])
        a.altair_chart(alt.Chart(wdf).mark_bar(color="#1f4e79", cornerRadiusEnd=3).encode(
            y=alt.Y("critere:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("poids:Q", axis=alt.Axis(format="%"), title="Poids"),
            tooltip=[alt.Tooltip("critere", title="Critère"), alt.Tooltip("poids", format=".1%", title="Poids")]
        ).properties(height=26 * len(wdf) + 30), width="stretch")
        if cr is not None:
            b.metric("Ratio de cohérence", f"{cr:.3f}", "cohérent" if cr < 0.10 else "à revoir",
                     delta_color="normal" if cr < 0.10 else "inverse")

    # ------------------------------------------------------------------------------------------ 4. filtre
    _step(4, "Filtre éliminatoire", "Un projet qui vérifie AU MOINS une règle est écarté avant le classement "
          "(catégorie Refuser). Ajoutez ou supprimez des lignes librement ; laissez vide pour ne rien écarter.")
    fbase = pd.DataFrame(base_cfg.get("filtre", []), columns=["critere", "operateur", "seuil", "libelle"])
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    fed = st.data_editor(fbase, key=f"filt_{ver}", num_rows="dynamic", hide_index=True, width="stretch",
                         column_config={"critere": st.column_config.SelectboxColumn("Colonne", options=num_cols, required=True),
                                        "operateur": st.column_config.SelectboxColumn("Opérateur", options=engine.OPERATEURS,
                                                                                     required=True),
                                        "seuil": st.column_config.NumberColumn("Seuil", format="%.4g", required=True),
                                        "libelle": st.column_config.TextColumn("Libellé affiché (facultatif)")})
    rules = [{"critere": r["critere"], "operateur": r["operateur"], "seuil": float(r["seuil"]),
              "libelle": r["libelle"] if isinstance(r["libelle"], str) and r["libelle"] else
              f"{labels.get(r['critere'], r['critere'])} {r['operateur']} {float(r['seuil']):g}"}
             for _, r in fed.dropna(subset=["critere", "operateur", "seuil"]).iterrows()]
    if rules:
        hits = sum(mcda.apply_rules(df, rules)["ecarte"]) if all(r["critere"] in df for r in rules) else 0
        st.caption(f"Avec ces règles, {hits} projet(s) sur {len(df)} seraient écartés.")

    # ------------------------------------------------------------------------------------------ 5. electre
    _step(5, "Paramètres ELECTRE", "q (indifférence), p (préférence) et v (veto) en % de l'étendue de chaque critère ; "
          "λ = niveau de crédibilité exigé pour qu'un projet « surclasse » un profil (0,6 par défaut).")
    s = base_cfg.get("seuils", {"q": 0.05, "p": 0.20, "v": 0.50})
    e1, e2, e3, e4 = st.columns(4)
    q = e1.slider("q (indifférence)", 0.0, 0.30, float(s["q"]), 0.01, key=f"q_{ver}", format="%.2f")
    p = e2.slider("p (préférence)", 0.0, 0.60, float(s["p"]), 0.01, key=f"p_{ver}", format="%.2f")
    v = e3.slider("v (veto)", 0.0, 1.0, float(s["v"]), 0.01, key=f"v_{ver}", format="%.2f")
    lam = e4.slider("Coupure λ", 0.5, 1.0, float(base_cfg.get("coupure", 0.6)), 0.05, key=f"l_{ver}")
    sens_on = st.toggle("Calculer la robustesse (200 scénarios de sensibilité, quelques secondes)", value=True,
                        key=f"sens_{ver}")

    # ------------------------------------------------------------------------------------------ lancer
    config = {"nom": name, "criteres": crit, "mode_poids": mode, "ahp_jugements": judg, "filtre": rules,
              "seuils": {"q": q, "p": p, "v": v}, "coupure": lam,
              "colonnes_info": [c for c in base_cfg.get("colonnes_info", []) if c in df.columns]}
    errs = engine.validate(df, config)
    if cr is not None and cr >= 0.10:
        errs.append(f"AHP : ratio de cohérence {cr:.3f} ≥ 0,10, revoyez les jugements.")
    st.divider()
    l1, l2, l3 = st.columns([1.4, 1, 1])
    if errs:
        st.warning("À corriger avant de lancer :\n\n" + "\n".join(f"- {e}" for e in errs))
    if l1.button("Lancer l'évaluation", type="primary", disabled=bool(errs), width="stretch",
                 icon=":material/play_arrow:"):
        try:
            common.set_active(df, config, demo=st.session_state["draft_src"] == "demo", sensitivity=sens_on)
            st.session_state["launched"] = True
            st.rerun()
        except Exception as e:
            st.error(f"Évaluation impossible : {e}")
    l2.download_button("Enregistrer la configuration (JSON)", engine.config_to_json(config).encode("utf-8"),
                       "configuration_mcda.json", "application/json", width="stretch", icon=":material/save:")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    l3.download_button("Données utilisées (CSV)", buf.getvalue().encode("utf-8"), "projets.csv", "text/csv",
                       width="stretch", icon=":material/download:")
    st.caption("En ligne de commande : python src/evaluer.py projets.csv --config configuration_mcda.json")
