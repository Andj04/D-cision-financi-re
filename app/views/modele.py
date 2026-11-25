"""Modèle de risque (démonstration SBA) : métriques, calibration, importances, erreurs par segment."""
import pandas as pd
import streamlit as st

import common
from common import R


def render():
    common.page_header("Référence", "Modèle de risque",
                       "Probabilité de défaut à 60 mois des prêts SBA 7(a) : régression logistique contre EBM, entraînés "
                       "sur 2000-2012, calibrés sur 2013-2014, testés une seule fois sur 2015-2016.")
    test = pd.read_csv(R / "metriques_test.csv")
    ebm = test[(test.modele == "ebm") & (test.version == "calibre")].iloc[0]
    lr = test[(test.modele == "logreg") & (test.version == "calibre")].iloc[0]
    common.kpis([("AUROC test · EBM", f"{ebm.AUROC:.3f}", f"logistique : {lr.AUROC:.3f}"),
                 ("AUPRC test · EBM", f"{ebm.AUPRC:.3f}", f"logistique : {lr.AUPRC:.3f}"),
                 ("Brier test · EBM", f"{ebm.Brier:.4f}", f"logistique : {lr.Brier:.4f}"),
                 ("Probabilité moyenne prédite", f"{100 * ebm.pd_moyenne:.2f} %",
                  f"taux réel : {100 * ebm.taux_defaut_reel:.2f} %", common.GREY)])
    fmt = {c: st.column_config.NumberColumn(c, format="%.4f")
           for c in ["AUROC", "AUPRC", "F1", "Brier", "taux_defaut_reel", "pd_moyenne", "seuil_F1"]}
    c1, c2 = st.columns(2, gap="large")
    with c1:
        common.section("Validation (2013-2014) : choix du modèle")
        st.dataframe(pd.read_csv(R / "metriques_validation.csv"), hide_index=True, width="stretch", column_config=fmt)
    with c2:
        common.section("Test (2015-2016) : lu une seule fois")
        st.dataframe(test, hide_index=True, width="stretch", column_config=fmt)
    hp = R / "reglage_hyperparametres.csv"
    if hp.exists():
        with st.expander("Réglage des hyperparamètres (sur la validation)", icon=":material/tune:"):
            st.dataframe(pd.read_csv(hp).round(4), hide_index=True, width="stretch")
    st.image(str(R / "calibration.png"), caption="Calibration : logistique vs EBM (validation puis test)")
    i1, i2 = st.columns(2)
    i1.image(str(R / "importance_variables.png"), caption="Importance native EBM et par permutation (validation)")
    i2.image(str(R / "ebm_effets.png"), caption="Effet des variables principales (EBM)")
    with st.expander("Erreurs par secteur, État et procédure (validation)", icon=":material/error:"):
        st.dataframe(pd.read_csv(R / "erreurs_par_segment_validation.csv").round(3), hide_index=True, width="stretch")
