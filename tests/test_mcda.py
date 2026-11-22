"""Tests du MCDA sur un exemple à trois projets calculé à la main (voir commentaires)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mcda  # noqa: E402

# Projets A=(10,10), B=(8,10), C=(10,4) ; poids 0,5/0,5 ; q=1, p=3, v=6 sur les deux critères (max)
PERF = np.array([[10.0, 10.0], [8.0, 10.0], [10.0, 4.0]])
W = np.array([0.5, 0.5])
TH = mcda.Thresholds(q=np.array([1.0, 1.0]), p=np.array([3.0, 3.0]), v=np.array([6.0, 6.0]))


def test_credibility_hand_computed():
    s = mcda.credibility(PERF, PERF, W, TH)
    assert np.isclose(s[0, 1], 1.0)    # A surclasse B : B n'est pas meilleur au-delà de q
    assert np.isclose(s[1, 0], 0.75)   # B vs A : c1 = (3-2)/(3-1) = 0,5 ; c2 = 1 -> 0,75 ; pas de veto
    assert np.isclose(s[0, 2], 1.0)    # A surclasse C
    assert np.isclose(s[2, 0], 0.0)    # C vs A : concordance 0,5 mais écart de 6 = v sur le critère 2 -> veto


def test_veto_blocks_outranking():
    s = mcda.credibility(PERF, PERF, W, TH)
    assert s[2, 0] < 0.5 <= 0.75       # la concordance seule (0,5) aurait donné un surclassement faible ; le veto l'annule


def test_electre3_dominant_first():
    res = mcda.electre_iii(PERF, W, TH)
    assert res["rank"][0] == 1


def test_ahp_target_weights_consistent():
    m = mcda.ahp_matrix_from_targets(mcda.TARGET_WEIGHTS)
    w, lam, cr = mcda.ahp_weights(m)
    assert m.shape == (8, 8) and cr < 0.10
    assert abs(w.sum() - 1) < 1e-9
    assert (w - pd.Series(mcda.TARGET_WEIGHTS)).abs().max() < 0.03   # proches des poids cibles de la décision 13


def test_filter():
    df = pd.DataFrame({"dscr_min": [0.9, 1.2, 1.5], "pd_ml": [0.05, 0.40, 0.05]})
    out = mcda.eliminatory_filter(df, train_default_rate=0.13)
    assert out["ecarte"].tolist() == [True, True, False]
