"""Évaluer un jeu de projets en ligne de commande (même moteur que l'application).

Exemples :
  python src/evaluer.py exemples/projets_exemple.csv                       # configuration devinée (poids égaux)
  python src/evaluer.py exemples/projets_exemple.csv --config exemples/configuration_exemple.json
  python src/evaluer.py --demo --sortie resultats_demo.csv                 # les 40 projets de démonstration
  python src/evaluer.py mes_projets.xlsx --ecrire-config ma_config.json    # écrit la configuration de départ à éditer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Évaluation multicritère (ELECTRE III / TRI, PROMETHEE II) d'un jeu de projets")
    ap.add_argument("projets", nargs="?", help="fichier CSV ou Excel (une ligne par projet)")
    ap.add_argument("--demo", action="store_true", help="utiliser les 40 projets de démonstration")
    ap.add_argument("--config", help="configuration JSON (sinon : devinée à partir des colonnes)")
    ap.add_argument("--ecrire-config", help="écrire la configuration utilisée dans ce fichier JSON")
    ap.add_argument("--sortie", default="resultats_mcda.csv", help="fichier CSV des résultats")
    ap.add_argument("--sans-sensibilite", action="store_true", help="ne pas calculer les 200 scénarios de robustesse")
    a = ap.parse_args(argv)

    if a.demo:
        df, cfg = engine.demo_projects(), engine.demo_config()
    elif a.projets:
        p = Path(a.projets)
        raw = pd.read_csv(p, sep=None, engine="python") if p.suffix.lower() == ".csv" else pd.read_excel(p)
        df = engine.prepare_projects(raw)
        cfg = engine.guess_config(df, name=p.stem)
    else:
        ap.error("indiquer un fichier de projets ou --demo")
    if a.config:
        cfg = engine.config_from_json(Path(a.config).read_text(encoding="utf-8"))
    if a.ecrire_config:
        Path(a.ecrire_config).write_text(engine.config_to_json(cfg), encoding="utf-8")
        print(f"configuration écrite dans {a.ecrire_config}")

    errs = engine.validate(df, cfg)
    if errs:
        print("Configuration invalide :\n- " + "\n- ".join(errs), file=sys.stderr)
        return 1
    out = engine.evaluate(df, cfg, sensitivity=not a.sans_sensibilite)
    res = out["results"]
    keep = ["id", "nom", "categorie_finale", "rang_electre3", "rang_promethee2", "ecarte", "motif_ecarte"]
    keep += [c["colonne"] for c in cfg["criteres"]]
    keep += [c for c in ("decision_fragile", "part_scenarios_categorie_differente", "position_ambigue") if c in res]
    res[keep].sort_values(["ecarte", "rang_electre3"]).to_csv(a.sortie, index=False)
    counts = res["categorie_finale"].value_counts()
    print(f"{cfg.get('nom')} : {len(res)} projets -> Financer {counts.get('Financer', 0)}, "
          f"Conditionner {counts.get('Conditionner', 0)}, Refuser {counts.get('Refuser', 0)} "
          f"(dont {int(res['ecarte'].sum())} écartés par le filtre)")
    if out["cr"] is not None:
        print(f"ratio de cohérence AHP : {out['cr']:.4f}")
    print("poids :", ", ".join(f"{c['libelle']} {100 * out['weights'][c['colonne']]:.1f} %" for c in cfg["criteres"]))
    print(f"résultats écrits dans {a.sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
