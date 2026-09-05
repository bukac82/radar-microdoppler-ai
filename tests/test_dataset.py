"""
Unit tests for Dataset module.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Dataset.loader import get_iq_matrix, get_labels, get_train_test_split
import pandas as pd


def _make_dummy_df(n=100):
    """Create a small dummy DataFrame mimicking the real CSV structure."""
    cols = ["num_blades", "radius_m", "rpm", "tip_velocity_m_s", "snr_db"]
    for i in range(50):
        cols += [f"I_{i}", f"Q_{i}"]
    data = np.random.default_rng(0).standard_normal((n, len(cols)))
    df = pd.DataFrame(data, columns=cols)
    df["num_blades"] = np.random.choice([2, 3, 4], size=n)
    return df


def test_get_iq_matrix_shape():
    df = _make_dummy_df(50)
    X = get_iq_matrix(df)
    assert X.shape == (50, 50), f"Expected (50, 50), got {X.shape}"
    assert np.iscomplexobj(X), "IQ matrix should be complex"


def test_get_labels():
    df = _make_dummy_df(30)
    y = get_labels(df)
    assert y.shape == (30,)
    assert set(y).issubset({2, 3, 4})


def test_train_test_split_sizes():
    df = _make_dummy_df(100)
    X_tr, X_te, y_tr, y_te = get_train_test_split(df, test_size=0.2)
    assert len(X_tr) == 80
    assert len(X_te) == 20
    assert len(y_tr) == 80
    assert len(y_te) == 20
