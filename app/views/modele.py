"""Modèle de risque (démonstration SBA) : métriques, calibration, importances, erreurs par segment."""
import pandas as pd
import streamlit as st

import common
from common import R


def render():
    common.hero("Modèle de risque", "Probabilité de défaut à 60 mois des prêts SBA 7(a) : régression logistique contre "
                "EBM, entraînés sur 2000-2012, calibrés sur 2013-2014, testés une seule fois sur 2015-2016.")
    test = pd.read_csv(R / "metriques_test.csv")
    ebm = test[(test.modele == "ebm") & (test.version == "calibre")].iloc[0]
    lr = test[(test.modele == "logreg") & (test.version == "calibre")].iloc[0]
    k = st.columns(4)
    k[0].metric("AUROC test (EBM)", f"{ebm.AUROC:.3f}", f"{ebm.AUROC - lr.AUROC:+.3f} vs logistique")
    k[1].metric("AUPRC test", f"{ebm.AUPRC:.3f}", f"{ebm.AUPRC - lr.AUPRC:+.3f} vs logistique")
    k[2].metric("Brier test", f"{ebm.Brier:.4f}", f"{ebm.Brier - lr.Brier:+.4f}", delta_color="inverse")
    k[3].metric(f"pd moyenne (réel {100 * ebm.taux_defaut_reel:.2f} %)", f"{100 * ebm.pd_moyenne:.2f} %",
                f"{100 * (ebm.pd_moyenne - ebm.taux_defaut_reel):+.2f} pt", delta_color="off")
    fmt = {c: st.column_config.NumberColumn(c, format="%.4f")
           for c in ["AUROC", "AUPRC", "F1", "Brier", "taux_defaut_reel", "pd_moyenne", "seuil_F1"]}
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Validation (2013-2014) : choix du modèle")
        st.dataframe(pd.read_csv(R / "metriques_validation.csv"), hide_index=True, width="stretch", column_config=fmt)
    with c2:
        st.markdown("#### Test (2015-2016) : lu une seule fois")
        st.dataframe(test, hide_index=True, width="stretch", column_config=fmt)
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
