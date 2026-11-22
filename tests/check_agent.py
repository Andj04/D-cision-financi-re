"""Vérifie le garde-fou de l'agent (sans appel réseau) : python tests/check_agent.py"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import agent  # noqa: E402

D = ROOT / "data/processed"
df = (pd.read_csv(D / "mcda_resultats.csv")
      .merge(pd.read_csv(D / "explications_projets.csv")[["loan_id", "facteurs_risque"]], on="loan_id")
      .merge(pd.read_csv(D / "simulation_detail.csv")[["loan_id", "secteur_damodaran", "etat", "montant_pret"]], on="loan_id"))
for _, r in df.iterrows():
    f = agent.build_facts(r.to_dict())
    ok, bad = agent.verify_note(agent.template_note(f), f)
    assert ok, (r.loan_id, bad)
ok, bad = agent.verify_note("La VAN est de 999 $ et la probabilité de défaut de 12.3 %.", agent.build_facts(df.iloc[0].to_dict()))
assert not ok and "999" in bad
print("agent OK : 40 notes de gabarit vérifiées ; un chiffre inventé est bien rejeté ->", bad)
print(agent.write_note(df.sort_values("rang_electre3").iloc[0].to_dict())["note"])
