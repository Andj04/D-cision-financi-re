"""Vérifie la présence des données et indique où les obtenir (les fichiers lourds ne sont pas dans Git).

Sources (vérifiées le 27/11/2026) :
  * SBA 7(a) FOIA, page « 7(a) & 504 FOIA » : https://data.sba.gov/en/dataset/7-a-504-foia
    (fiche data.gov : https://catalog.data.gov/dataset/sba-7a-and-504-loan-data-reports)
    Fichiers utilisés : « FOIA - 7(a) (FY2000-FY2009) » et « FOIA - 7(a) (FY2010-FY2019) », instantané au 30/09/2026
    (asof 260930), avec le dictionnaire de données « 7(a) and 504 FOIA Data Dictionary ». La SBA publie un nouvel
    instantané chaque trimestre : un fichier plus récent donne des chiffres légèrement différents (prêts mis à jour).
    À placer dans data/raw/ sous les noms ci-dessous (xlsx ; un csv du même contenu convient, data_prep.py le lit).
  * Damodaran, « Cost of Equity and Capital (US) » : https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/wacc.html
    tableur : https://www.stern.nyu.edu/~adamodar/pc/datasets/wacc.xls -> l'ouvrir et l'enregistrer en
    data/external/wacc.xlsx (format .xlsx, lu par src/damodaran.py). Données de janvier de chaque année.
Usage : python scripts/download_data.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "data/raw/FOIA_7a_FY2000_FY2009_asof_260930.xlsx": 172_673_904,
    "data/raw/FOIA_7a_FY2010_FY2019_asof_260930.xlsx": 142_802_089,
    "data/external/wacc.xlsx": 34_266,
}
SOURCES = {"data/raw": "https://data.sba.gov/en/dataset/7-a-504-foia",
           "data/external": "https://www.stern.nyu.edu/~adamodar/pc/datasets/wacc.xls (à enregistrer en .xlsx)"}
missing = False
for rel, size in FILES.items():
    p = ROOT / rel
    if not p.exists() and p.suffix == ".xlsx" and p.with_suffix(".csv").exists():
        p = p.with_suffix(".csv")                       # version csv du même fichier SBA
    if p.exists():
        print(f"OK       {rel}  ({p.stat().st_size:,} octets{'' if p.stat().st_size == size else f' ; attendu {size:,}'})")
    else:
        missing = True
        print(f"MANQUANT {rel}  <- {SOURCES[str(Path(rel).parent).replace(chr(92), '/')]}")
if missing:
    print("\nTélécharger les fichiers manquants (sources ci-dessus) puis relancer. Ensuite : python src/data_prep.py")
else:
    print("\nToutes les données sont présentes. Suite : python src/data_prep.py")
