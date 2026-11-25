"""Garde-fous du pipeline (ne demandent que les fichiers versionnés ou le code)."""
import inspect
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features  # noqa: E402


def test_no_leaky_or_double_counted_feature():
    # JobsSupported et CollateralInd servent au MCDA ; TermInMonths exact est altéré après l'octroi
    assert not {"JobsSupported", "CollateralInd", "TermInMonths"} & set(features.FEATURES)
    assert "garantie_reelle" not in features.FEATURES


def test_simulation_does_not_read_target_or_exact_term():
    import simulate
    src = inspect.getsource(simulate.simulate)
    assert 'drop(columns=["default_60m"]' in src
    assert "TermInMonths\"] / 12" not in src and "duree_classe(" in src


def test_project_scores_shape():
    s = pd.read_csv(ROOT / "data/processed/project_scores.csv")
    assert s.shape == (40, 9) and not s.isna().any().any()
