"""Étape 2 : liste de variables du modèle de risque (décision 1) et prétraitement.

Les deux modèles (régression logistique, EBM) utilisent EXACTEMENT la même liste de variables.
JobsSupported est volontairement absent : il devient le critère MCDA « emplois par million »
(sinon double comptage). Aucune colonne connue après l'octroi n'est utilisée (statut, dates de
paiement/perte, montants de radiation : retirés dès data_prep.py).

Tout ce qui apprend quelque chose des données (médianes, échelles, modalités) est ajusté
sur l'ENTRAÎNEMENT uniquement : les transformateurs sont des Pipelines scikit-learn fit sur train.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "default_60m"
ID_COL = "loan_id"

NUMERIC = ["log_montant", "part_garantie", "duree_mois", "taux_initial", "taux_manquant"]
CATEGORICAL = ["taux_fixe_variable", "ligne_renouvelable", "secteur_naics2", "etat", "type_entreprise",
               "age_entreprise", "franchise", "procedure", "garantie_reelle"]
FEATURES = NUMERIC + CATEGORICAL
FORBIDDEN = {"JobsSupported", "LoanStatus", "ChargeOffDate", "PaidInFullDate", "GrossChargeOffAmount",
             "default_60m", "ApprovalFY"}


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transformations sans apprentissage (déterministes, ligne par ligne) : sûres avant le fit."""
    out = pd.DataFrame(index=df.index)
    approval = df["GrossApproval"].astype(float)
    out["log_montant"] = np.log1p(approval)
    out["part_garantie"] = (df["SBAGuaranteedApproval"].astype(float) / approval).clip(0, 1)
    out["duree_mois"] = df["TermInMonths"].astype(float)
    rate = pd.to_numeric(df["InitialInterestRate"], errors="coerce")
    out["taux_initial"] = rate
    out["taux_manquant"] = rate.isna().astype(float)
    out["taux_fixe_variable"] = df["FixedorVariableInterestInd"].astype("object").fillna("inconnu")
    out["ligne_renouvelable"] = df["RevolverStatus"].astype("object").fillna("inconnu")
    naics = df["NaicsCode"].astype("object")
    out["secteur_naics2"] = naics.str[:2].fillna("inconnu")
    out["etat"] = df["BorrState"].astype("object").fillna("inconnu")
    out["type_entreprise"] = df["BusinessType"].astype("object").fillna("inconnu")
    out["age_entreprise"] = df["BusinessAge"].astype("object").fillna("inconnu")
    out["franchise"] = np.where(df["FranchiseCode"].notna(), "oui", "non")
    out["procedure"] = df["ProcessingMethod"].astype("object").fillna("inconnu")
    out["garantie_reelle"] = df["CollateralInd"].astype("object").fillna("inconnu")
    assert not FORBIDDEN & set(out.columns)
    return out[FEATURES]


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    from pathlib import Path
    df = pd.read_parquet(Path(__file__).resolve().parents[1] / f"data/processed/{name}.parquet")
    return make_features(df), df[TARGET].astype(int), df[ID_COL]


def logreg_preprocessor() -> ColumnTransformer:
    """Imputation médiane + standardisation (numériques) ; one-hot (catégorielles) : ajustés sur train."""
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200, sparse_output=False)
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])
