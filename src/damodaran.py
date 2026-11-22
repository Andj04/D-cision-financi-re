"""Lecture de data/external/wacc.xlsx (Damodaran, feuille « Industry Averages »).

openpyxl échoue sur ce fichier (feuille de style non standard) ; on lit donc le XML directement.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
PATH = Path(__file__).resolve().parents[1] / "data/external/wacc.xlsx"


def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_sheet(path: Path = PATH, sheet_file: str = "xl/worksheets/sheet2.xml") -> list[list]:
    z = zipfile.ZipFile(path)
    shared = [("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
              for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)]
    rows = []
    for r in ET.fromstring(z.read(sheet_file)).iter("{%s}row" % NS["m"]):
        cells = {}
        for c in r.findall("m:c", NS):
            v = c.find("m:v", NS)
            if v is None:
                continue
            cells[_col(c.get("r"))] = shared[int(v.text)] if c.get("t") == "s" else float(v.text)
        if cells:
            rows.append([cells.get(i) for i in range(max(cells) + 1)])
    return rows


def industry_table(path: Path = PATH) -> pd.DataFrame:
    rows = read_sheet(path)
    hdr = next(i for i, r in enumerate(rows) if r and r[0] == "Industry Name")
    header = [str(h).strip() for h in rows[hdr]]
    width = len(header)
    data = [(r + [None] * width)[:width] for r in rows[hdr + 1:] if r and r[0]]
    return pd.DataFrame(data, columns=header)


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    t = industry_table()
    print(t.shape); print(t.head(12)); print(t.columns.tolist()); print(t["Industry Name"].tolist())
