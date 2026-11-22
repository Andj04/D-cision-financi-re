"""Étape 1 : construction de la cible default_60m et découpage train / validation / test.

Décisions du guide (3, 4, 5) :
  3. lignes sans FirstDisbursementDate écartées (comptées dans l'audit) ;
  4. défaut = CHGOFF (CLSLN et SOLDCO seraient regroupés s'ils existaient dans le fichier) ;
     default_60m = 1 si la perte survient au plus 60 mois après le premier versement ;
  5. séparation par ApprovalFY : entraînement 2000-2012, validation 2013-2014, test 2015-2016.

Le découpage se fait AVANT tout calcul qui apprend quelque chose des données.
Les colonnes connues seulement après l'octroi (statut, dates de paiement/perte, montants de
radiation) sont retirées des fichiers produits : elles ne peuvent pas fuiter dans le modèle.

Usage : python src/data_prep.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PARQUET = [
    ROOT / "data/processed/sba_7a_FY2000_FY2009.parquet",
    ROOT / "data/processed/sba_7a_FY2010_FY2019.parquet",
]
RAW_XLSX = [
    ROOT / "data/raw/FOIA_7a_FY2000_FY2009_asof_260930.xlsx",
    ROOT / "data/raw/FOIA_7a_FY2010_FY2019_asof_260930.xlsx",
]
OUT = ROOT / "data/processed"

DEFAULT_STATUSES = {"CHGOFF", "CLSLN", "SOLDCO"}   # décision 4
REPAID_STATUSES = {"P I F"}
KNOWN_STATUSES = DEFAULT_STATUSES | REPAID_STATUSES | {"CANCLD", "COMMIT", "EXEMPT", "NOT FUNDED", "LIQUID", "PURCH(NOT C/O)"}
HORIZON_MONTHS = 60
SPLITS = {"train": (2000, 2012), "val": (2013, 2014), "test": (2015, 2016)}

# Colonnes connues seulement après l'octroi, ou identifiantes : jamais dans les fichiers modèle.
LEAKY = ["LoanStatus", "PaidInFullDate", "ChargeOffDate", "GrossChargeOffAmount", "FirstDisbursementDate",
         "SoldSecMrktInd", "AsOfDate"]
PII = ["BorrName", "BorrStreet", "BorrCity", "BorrZip", "BankName", "BankStreet", "BankCity", "BankZip",
       "BankFDICNumber", "BankNCUANumber", "LocationID"]


def excel_serial_to_date(s: pd.Series) -> pd.Series:
    """Dates stockées comme numéros de série Excel (parfois en texte) -> datetime."""
    n = pd.to_numeric(s, errors="coerce")
    return pd.to_datetime("1899-12-30") + pd.to_timedelta(n, unit="D")


def load_raw() -> pd.DataFrame:
    """Lit les parquet (conversion des xlsx faite une fois) ; sinon reconvertit les xlsx."""
    if all(p.exists() for p in RAW_PARQUET):
        return pd.concat([pd.read_parquet(p) for p in RAW_PARQUET], ignore_index=True)
    frames = []
    for x, p in zip(RAW_XLSX, RAW_PARQUET):
        df = pd.read_excel(x, dtype=object)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.astype(str).replace({"nan": None}).to_parquet(p)  # types mixtes -> texte
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_loan_id(df: pd.DataFrame) -> pd.Series:
    """Identifiant haché stable (pas de nom d'emprunteur dans les fichiers produits)."""
    key = (df["BorrName"].astype(str) + "|" + df["ApprovalDate"].astype(str) + "|"
           + df["GrossApproval"].astype(str) + "|" + df["BankName"].astype(str) + "|"
           + df["BorrZip"].astype(str))
    return key.map(lambda k: hashlib.sha1(k.encode("utf-8")).hexdigest()[:12])


def build() -> dict:
    audit: dict = {}
    df = load_raw()
    audit["lignes_brutes"] = len(df)

    # --- audit des statuts (consigne : compter avant de construire la cible)
    audit["statuts_bruts"] = df["LoanStatus"].map(
        lambda v: v if v in KNOWN_STATUSES else "autre (ligne décalée / valeur parasite)").value_counts().to_dict()
    audit["ecriture_PIF_variantes"] = sorted({v for v in df["LoanStatus"].dropna().unique()
                                              if str(v).replace(" ", "").upper() == "PIF"})

    # lignes dont les colonnes sont décalées dans le fichier source (statut numérique, etc.)
    bad = ~df["LoanStatus"].isin(KNOWN_STATUSES)
    audit["lignes_statut_invalide_retirees"] = int(bad.sum())
    df = df[~bad]

    # doublons exacts
    n0 = len(df)
    df = df.drop_duplicates()
    audit["doublons_exacts_retires"] = n0 - len(df)

    # --- décision 3 : lignes sans FirstDisbursementDate
    df["first_disb"] = excel_serial_to_date(df["FirstDisbursementDate"])
    nodate = df["first_disb"].isna()
    audit["sans_FirstDisbursementDate_total"] = int(nodate.sum())
    audit["sans_FirstDisbursementDate_par_statut"] = df.loc[nodate, "LoanStatus"].value_counts().to_dict()
    df = df[~nodate]

    # --- statuts non résolus ou annulés : issue inconnue -> retirés
    resolved = df["LoanStatus"].isin(DEFAULT_STATUSES | REPAID_STATUSES)
    audit["retires_statut_non_resolu"] = df.loc[~resolved, "LoanStatus"].value_counts().to_dict()
    df = df[resolved].copy()

    # --- cible default_60m (décision 4)
    df["approval_fy"] = pd.to_numeric(df["ApprovalFY"], errors="coerce")
    df["co_date"] = excel_serial_to_date(df["ChargeOffDate"])
    as_of = excel_serial_to_date(pd.Series([df["AsOfDate"].iloc[0]])).iloc[0]
    audit["date_du_fichier"] = str(as_of.date())
    horizon_end = df["first_disb"] + pd.DateOffset(months=HORIZON_MONTHS)
    # prêts versés depuis au moins 60 mois à la date du fichier (issue observable)
    observable = horizon_end <= as_of
    audit["retires_moins_de_60_mois"] = int((~observable).sum())
    df = df[observable]
    horizon_end = horizon_end[observable]
    is_default_status = df["LoanStatus"].isin(DEFAULT_STATUSES)
    df["default_60m"] = (is_default_status & df["co_date"].notna() & (df["co_date"] <= horizon_end)).astype(int)
    audit["defauts_status_chgoff_apres_60m_comptes_zero"] = int(
        (is_default_status & (df["default_60m"] == 0)).sum())

    # --- séparation temporelle (décision 5), AVANT tout prétraitement
    df["loan_id"] = make_loan_id(df)
    dup_ids = df["loan_id"].duplicated(keep="first")
    audit["id_haches_en_double_retires"] = int(dup_ids.sum())
    df = df[~dup_ids]

    keep_years = df["approval_fy"].between(2000, 2016)
    audit["retires_hors_2000_2016"] = int((~keep_years).sum())
    df = df[keep_years]

    for c in LEAKY + PII + ["first_disb", "co_date", "ApprovalFY"]:
        if c in df.columns:
            df = df.drop(columns=c)
    df = df.rename(columns={"approval_fy": "ApprovalFY"})
    for c in ["InitialInterestRate", "JobsSupported", "SBAGuaranteedApproval", "GrossApproval", "TermInMonths"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("string")

    OUT.mkdir(parents=True, exist_ok=True)
    audit["blocs"] = {}
    for name, (lo, hi) in SPLITS.items():
        part = df[df["ApprovalFY"].between(lo, hi)].reset_index(drop=True)
        part.to_parquet(OUT / f"{name}.parquet", index=False)
        audit["blocs"][name] = {"annees": f"{lo}-{hi}", "lignes": len(part),
                                "taux_defaut": round(float(part["default_60m"].mean()), 4),
                                "defauts": int(part["default_60m"].sum())}
    # garde-fou : pas de recouvrement d'années ni d'identifiants
    ids = [set(pd.read_parquet(OUT / f"{n}.parquet", columns=["loan_id"])["loan_id"]) for n in SPLITS]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "recouvrement entre blocs"

    (OUT / "audit_donnees.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str),
                                            encoding="utf-8")
    pd.DataFrame(audit["blocs"]).T.to_csv(OUT / "taux_defaut_par_bloc.csv")
    return audit


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, ensure_ascii=False, default=str))
