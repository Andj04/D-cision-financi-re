"""Agent d'arbitrage : contraintes de budget, de bornes et de plafond sectoriel, deux règles de sélection."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import arbitrage  # noqa: E402
import engine  # noqa: E402


def toy():
    # 5 projets : A (rang 1, cher), B, C, D « Financer » ; E « Conditionner » ; F « Refuser »
    return pd.DataFrame({
        "nom": list("ABCDEF"), "categorie_finale": ["Financer"] * 4 + ["Conditionner", "Refuser"],
        "rang_electre3": [1, 2, 3, 4, 5, None], "flux_net_promethee": [0.5, 0.3, 0.2, 0.1, 0.0, None],
        "montant_pret": [600, 300, 300, 100, 100, 50], "secteur": ["X", "X", "Y", "Z", "Z", "Y"]})


def test_ranking_rule_follows_committee_order():
    res, s = arbitrage.run(toy(), {"budget": 1000, "montant_min": 0, "montant_max": 10_000, "part_max_secteur": 0.7})
    sel = dict(zip(res.nom, res.selection_budget))
    assert sel["A"] and not sel["B"]                       # A (rang 1) passe ; B dépasserait le plafond du secteur X
    assert res.set_index("nom").loc["B", "motif_arbitrage"] == "plafond du secteur atteint"
    assert sel["C"] and sel["D"] and s["montant_engage"] == 1000          # 600 + 300 + 100 : budget plein
    assert res.set_index("nom").loc["E", "motif_arbitrage"] == "budget insuffisant"
    assert not sel["F"] and res.set_index("nom").loc["F", "motif_arbitrage"] == "refusé par le MCDA"


def test_constraints_always_hold():
    for mode in ("classement", "nombre"):
        for budget in (200, 500, 900, 2000):
            p = {"budget": budget, "montant_min": 80, "montant_max": 650, "part_max_secteur": 0.6, "mode": mode}
            res, s = arbitrage.run(toy(), p)
            chosen = res[res.selection_budget]
            assert chosen.montant_pret.sum() <= budget + 1e-9
            assert chosen.montant_pret.between(80, 650).all()
            assert (chosen.groupby("secteur").montant_pret.sum() <= 0.6 * budget + 1e-9).all()
            assert (chosen.categorie_finale != "Refuser").all()


def test_count_mode_maximises_number_of_financer():
    p = {"budget": 700, "montant_min": 0, "montant_max": 10_000, "part_max_secteur": 1.0}
    n_rank = arbitrage.run(toy(), {**p, "mode": "classement"})[1]["dont_financer"]
    n_count = arbitrage.run(toy(), {**p, "mode": "nombre"})[1]["dont_financer"]
    assert n_count >= n_rank and n_count == 3                  # B + C + D (700) plutôt que A seul (600)


def test_exclude_conditional_and_missing_amount():
    res, _ = arbitrage.run(toy(), {"budget": 5000, "inclure_conditionner": False})
    assert not res.set_index("nom").loc["E", "selection_budget"]
    res, s = arbitrage.run(toy().drop(columns="montant_pret"), {})
    assert not s["actif"] and not res.selection_budget.any()


@pytest.fixture(scope="module")
def demo():
    return engine.evaluate(engine.demo_projects(), engine.demo_config(), sensitivity=False)


def test_demo_selection(demo):
    a, r = demo["arbitrage"], demo["results"].set_index("nom")
    assert a["projets_selectionnes"] == 11 and a["montant_engage"] <= a["budget"]
    assert r.loc["Studio de développement logiciel Murphy", "selection_budget"]      # le n° 1 est financé
    assert not r.loc["Garde-meubles Grant", "selection_budget"]                         # 5 M$ : hors bornes


def test_validation_of_budget_parameters():
    df = engine.prepare_projects(pd.read_csv(ROOT / "exemples/test1_pme_casablanca.csv"))
    cfg = engine.config_from_json((ROOT / "exemples/test1_configuration_ahp.json").read_text(encoding="utf-8"))
    assert engine.validate(df, cfg) == []
    bad = {**cfg, "arbitrage": {**cfg["arbitrage"], "budget": -1, "montant_min": 9e9}}
    errs = engine.validate(df, bad)
    assert any("budget" in e for e in errs) and any("minimum" in e for e in errs)
