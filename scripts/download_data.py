"""Vérifie la présence des données et rappelle où les obtenir (les fichiers lourds ne sont pas dans Git).

Sources (voir README, journal des données) :
  * SBA 7(a) FOIA : FOIA_7a_FY2000_FY2009_asof_260930.xlsx et FOIA_7a_FY2010_FY2019_asof_260930.xlsx
    -> page « 7(a) & 504 FOIA » du portail data.sba.gov, à télécharger dans data/raw/ ;
  * Damodaran « wacc.xlsx » (Cost of Capital by Industry, US) -> pages.stern.nyu.edu/~adamodar, dans data/external/.
Usage : python scripts/download_data.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "data/raw/FOIA_7a_FY2000_FY2009_asof_260930.xlsx": 172_673_904,
    "data/raw/FOIA_7a_FY2010_FY2019_asof_260930.xlsx": 142_802_089,
    "data/external/wacc.xlsx": 34_266,
}
missing = False
for rel, size in FILES.items():
    p = ROOT / rel
    if p.exists():
        print(f"OK       {rel}  ({p.stat().st_size:,} octets{'' if p.stat().st_size == size else f' ; attendu {size:,}'})")
    else:
        missing = True
        print(f"MANQUANT {rel}")
if missing:
    print("\nTélécharger les fichiers manquants (sources ci-dessus) puis relancer. Ensuite : python src/data_prep.py")
else:
    print("\nToutes les données sont présentes. Suite : python src/data_prep.py")
