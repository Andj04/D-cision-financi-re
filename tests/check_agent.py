"""Vérifie le garde-fou de l'agent sur les 40 projets de démonstration : python tests/check_agent.py
(appelle le LLM pour une note seulement si une clé est configurée ; sinon note de gabarit)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import agent  # noqa: E402
import engine  # noqa: E402

ev = engine.evaluate(engine.demo_projects(), engine.demo_config(), sensitivity=True)
df, crit = ev["results"], ev["config"]["criteres"]
for _, r in df.iterrows():
    f = agent.build_facts(r.to_dict(), crit)
    ok, bad = agent.verify_note(agent.template_note(f), f)
    assert ok, (r.nom, bad)
ok, bad = agent.verify_note("La VAN est de 999 $ et la probabilité de défaut de 12.3 %.", agent.build_facts(df.iloc[0].to_dict(), crit))
assert not ok and "999" in bad
print("agent OK : 40 notes de gabarit vérifiées ; un chiffre inventé est bien rejeté ->", bad)
out = agent.write_note(df.sort_values("rang_electre3").iloc[0].to_dict(), crit)
print(out["source"], "|", out["verification"])
print(out["note"])
