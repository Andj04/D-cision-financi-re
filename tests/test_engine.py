"""Outil réutilisable : moteur générique, configuration, conseiller (outils sans LLM), noms des projets."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import advisor  # noqa: E402
import agent  # noqa: E402
import engine  # noqa: E402
import naming  # noqa: E402

EX = ROOT / "exemples"


@pytest.fixture(scope="module")
def demo_eval():
    return engine.evaluate(engine.demo_projects(), engine.demo_config(), sensitivity=True)


def test_engine_reproduces_pipeline(demo_eval):
    r = demo_eval["results"].set_index("id")
    b = pd.read_csv(ROOT / "data/processed/mcda_resultats.csv").set_index("loan_id").loc[r.index]
    assert (r["categorie_finale"] == b["categorie_finale"]).all()
    assert (r["rang_electre3"].fillna(-1) == b["rang_electre3"].fillna(-1)).all()
    assert (r["decision_fragile"] == b["decision_fragile"]).all()


def test_generic_dataset_and_config_roundtrip():
    df = engine.prepare_projects(pd.read_csv(EX / "projets_exemple.csv"))
    cfg = engine.config_from_json((EX / "configuration_exemple.json").read_text(encoding="utf-8"))
    assert engine.validate(df, cfg) == []
    out = engine.evaluate(df, cfg, sensitivity=False)
    res = out["results"]
    assert len(res) == len(df) and res["categorie_finale"].isin(["Financer", "Conditionner", "Refuser"]).all()
    assert res.loc[res["dscr_min"] < 1, "ecarte"].all()                       # règle du filtre appliquée
    assert engine.config_from_json(engine.config_to_json(cfg)) == json.loads(engine.config_to_json(cfg))


def test_example_datasets():
    d1 = engine.prepare_projects(pd.read_csv(EX / "test1_pme_casablanca.csv"))
    c1 = engine.config_from_json((EX / "test1_configuration_ahp.json").read_text(encoding="utf-8"))
    r1 = engine.evaluate(d1, c1, sensitivity=False)["results"]
    assert r1["categorie_finale"].value_counts().to_dict() == {"Financer": 6, "Conditionner": 6, "Refuser": 3}
    d2 = engine.prepare_projects(pd.read_csv(EX / "test2_prets_format_sba.csv"))   # pd_ml calculée par le modèle
    assert "pd_ml" in d2 and d2["pd_ml"].between(0, 1).all() and d2["pd_ml"].nunique() > 3
    d3 = engine.prepare_projects(pd.read_excel(EX / "test3_projets_energie.xlsx"))
    sens = {c["colonne"]: c["sens"] for c in engine.guess_config(d3)["criteres"]}
    assert sens["cout_par_mw_mdh"] == sens["risque_pays_note"] == sens["delai_construction_mois"] == "min"
    assert sens["co2_evite_t_an"] == sens["rendement_tri_pct"] == "max"
    d4 = engine.prepare_projects(pd.read_csv(EX / "test4_fichier_avec_erreurs.csv"))
    errs = engine.validate(d4, engine.guess_config(d4))
    assert any("van" in e for e in errs) and any("dscr_min" in e for e in errs)


def test_french_numbers():
    p = engine.prepare_projects(pd.DataFrame({"nom": ["A", "B"], "van": ["1 234,5", "-200"], "risque": ["2,5 %", "4 %"]}))
    assert p["van"].tolist() == [1234.5, -200.0] and p["risque"].tolist() == [2.5, 4.0]


def test_guess_config_and_validation_errors():
    df = engine.prepare_projects(pd.DataFrame({"risque": [0.1, 0.2, 0.05], "rendement": [5, 3, 8], "montant": [1, 2, 3]}))
    cfg = engine.guess_config(df)
    sens = {c["colonne"]: c["sens"] for c in cfg["criteres"]}
    assert sens == {"risque": "min", "rendement": "max"}                     # « montant » = descriptif
    assert df["nom"].is_unique and df["id"].tolist() == ["P001", "P002", "P003"]
    bad = {**cfg, "filtre": [{"critere": "absente", "operateur": "<", "seuil": 1}]}
    assert any("absente" in e for e in engine.validate(df, bad))


def test_ahp_mode_from_config():
    df = engine.prepare_projects(pd.read_csv(EX / "projets_exemple.csv"))
    cfg = engine.config_from_json((EX / "configuration_exemple.json").read_text(encoding="utf-8"))
    import mcda
    tw = {c["colonne"]: c["poids"] for c in cfg["criteres"]}
    cfg["mode_poids"] = "ahp"
    cfg["ahp_jugements"] = mcda.judgments_table(mcda.ahp_matrix_from_targets(tw)).to_dict("records")
    w, cr = engine.weights_from_config(cfg)
    assert cr < 0.10 and abs(w.sum() - 1) < 1e-9


def test_advisor_tools_without_llm(demo_eval):
    ws = advisor.Workspace(demo_eval, demo=True)
    name = demo_eval["results"]["nom"].iloc[0]
    assert ws.fiche_projet(name.lower())["faits"]["projet"] == name               # recherche tolérante
    assert ws.lister_projets(categorie="Financer")["nombre_total"] == 9
    sc = ws.simuler_scenario(critere="esg_score", nouveau_poids=0.3)
    assert abs(sum(float(v.rstrip(" %")) for v in sc["poids_utilises"].values()) - 100) < 0.5
    st = ws.statistiques("pd_ml", "categorie")["statistiques"]
    assert set(st) == {"Financer", "Conditionner", "Refuser"}
    for spec in advisor.tool_specs(ws):                                          # chaque outil déclaré existe
        assert callable(getattr(ws, spec["name"]))


def test_scenario_several_weights_and_capabilities(demo_eval):
    ws = advisor.Workspace(demo_eval, demo=True)
    sc = ws.simuler_scenario(poids=[{"critere": "esg_score", "poids": 0.2}, {"critere": "van", "poids": 30}])
    w = {k: float(v.rstrip(" %")) for k, v in sc["poids_utilises"].items()}
    assert w["Score ESG"] == 20.0 and w["VAN"] == 30.0 and abs(sum(w.values()) - 100) < 0.5
    assert "erreur" in ws.simuler_scenario(poids=[{"critere": "van", "poids": 0.8}, {"critere": "pd_ml", "poids": 0.5}])
    cap = ws.capacites()
    assert cap["ce_que_je_ne_fais_pas"] and "ELECTRE TRI" in cap["methodes_utilisees"]


def test_project_sheet_under_scenario(demo_eval):
    ws = advisor.Workspace(demo_eval, demo=True)
    r = ws.fiche_projet("Garde-meubles Silva", poids=[{"critere": "esg_score", "poids": 0.2},
                                                      {"critere": "van", "poids": 0.3}])
    ch = r["changements_par_rapport_a_l_analyse_de_reference"]
    assert ch["categorie"] == "Conditionner -> Financer" and r["faits"]["categorie"] == "Financer"
    assert r["scenario"]["poids_utilises"]["Score ESG"] == "20.0 %"
    assert "stabilite_simulation" not in r["faits"]                  # non recalculée dans un scénario
    assert "fiche_projet" in [t["name"] for t in advisor.tool_specs(ws)]
    assert "poids" in next(t for t in advisor.tool_specs(ws) if t["name"] == "fiche_projet")["parameters"]["properties"]


def test_foreign_method_names_flagged():
    assert advisor.foreign_methods("On utilise ELECTRE Tri-C et TOPSIS.") == ["ELECTRE Tri-C", "TOPSIS"]
    ok = "AHP, ELECTRE III, ELECTRE TRI (c'est la règle pessimiste), PROMETHEE II et l'EBM."
    assert advisor.foreign_methods(ok) == []


def test_verifier_handles_thousands_and_flags_invented():
    src = [json.dumps({"prets": 10601, "taux": "3.4 %"})]
    assert advisor._verify("10 601 prêts, taux de 3,4 %", src) == []
    assert advisor._verify("taux de 12,7 %", src) == ["12.7"]


def test_agent_generic_criteria():
    crit = [{"colonne": "risque", "libelle": "Risque", "format": "pct"}, {"colonne": "rendement", "libelle": "Rendement"}]
    row = {"id": "P1", "nom": "Atelier Test", "risque": 0.05, "rendement": 3.2, "categorie_finale": "Financer",
           "ecarte": False, "rang_electre3": 1, "rang_promethee2": 2, "motif_ecarte": ""}
    f = agent.build_facts(row, crit)
    note = agent.template_note(f)
    assert "Atelier Test" in note and "5.0 %" in note and agent.verify_note(note, f)[0]


def test_names_unique_and_deterministic():
    ids = pd.Series([f"x{i}" for i in range(60)])
    a, b = naming.human_names(ids, pd.Series(["722511"] * 60)), naming.human_names(ids, pd.Series(["722511"] * 60))
    assert a.equals(b) and a.is_unique
    sim = pd.read_csv(ROOT / "data/processed/simulation_detail.csv")
    assert sim["nom_projet"].is_unique and sim["nom_projet"].str.len().min() > 5
