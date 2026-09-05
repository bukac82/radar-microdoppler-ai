"""
Dataset Loader Utilities
========================
Utilities for loading and splitting the helicopter micro-Doppler datasets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Default data paths (relative to project root)
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "helicopter_microdoppler_dataset.csv"
EXTENDED_DATASET_PATH = BASE_DIR / "helicopter_microdoppler_extended_dataset.csv"

IQ_COLS_START = 5      # Index where I_0, Q_0, ... columns begin (base dataset)
EXT_IQ_COLS_START = 8  # Index for extended dataset (has 3 extra metadata cols)


def load_dataset(
    filepath: str | Path = DATASET_PATH,
    nrows: int | None = None,
    extended: bool = False,
) -> pd.DataFrame:
    """Load the micro-Doppler CSV dataset.

    Args:
        filepath: Path to the CSV file. Defaults to the base dataset.
        nrows: If set, load only the first `nrows` rows (useful for dev/debug).
        extended: Set True when loading the extended dataset — adjusts column offset.

    Returns:
        pd.DataFrame with metadata columns + IQ signal columns.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Run generate_dataset.py or generate_dataset_extended.py first."
        )
    df = pd.read_csv(filepath, nrows=nrows)
    return df


def get_iq_matrix(df: pd.DataFrame, extended: bool = False) -> np.ndarray:
    """Extract the raw IQ signal matrix from a loaded DataFrame.

    Args:
        df: DataFrame from load_dataset().
        extended: True if the DataFrame comes from the extended dataset.

    Returns:
        Complex numpy array of shape (n_samples, n_timesteps).
    """
    start = EXT_IQ_COLS_START if extended else IQ_COLS_START
    iq_cols = df.columns[start:]
    n_timesteps = len(iq_cols) // 2  # half I, half Q

    I = df[[f"I_{i}" for i in range(n_timesteps)]].values
    Q = df[[f"Q_{i}" for i in range(n_timesteps)]].values
    return I + 1j * Q


def get_labels(df: pd.DataFrame) -> np.ndarray:
    """Return integer labels (num_blades: 2, 3, or 4)."""
    return df["num_blades"].values.astype(int)


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    extended: bool = False,
):
    """Split into train/test sets.

    Returns:
        X_train, X_test (complex IQ matrices), y_train, y_test (int labels).
    """
    X = get_iq_matrix(df, extended=extended)
    y = get_labels(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


if __name__ == "__main__":
    print("Loading first 1000 rows from base dataset...")
    df = load_dataset(nrows=1000)
    print(f"  Shape : {df.shape}")
    print(f"  Classes: {df['num_blades'].value_counts().to_dict()}")
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    print(f"  Train : {X_train.shape}, Test: {X_test.shape}")
