"""
Preprocessing Pipeline
======================
Sklearn-compatible end-to-end preprocessing pipeline for the micro-Doppler dataset.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import numpy as np

from Preprocessing.features import extract_features
from Preprocessing.normalize import fit_feature_scaler


def build_feature_pipeline(scaler_type: str = "standard") -> Pipeline:
    """Build a feature-extraction + normalization pipeline.

    The pipeline:
      1. extract_features: IQ array → 14-D feature vector per sample
      2. StandardScaler / MinMaxScaler

    Args:
        scaler_type: 'standard' or 'minmax'

    Returns:
        Unfitted sklearn Pipeline.
    """
    feature_extractor = FunctionTransformer(
        extract_features, validate=False
    )
    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

    pipeline = Pipeline([
        ("features", feature_extractor),
        ("scaler", scaler),
    ])
    return pipeline


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X_iq = rng.standard_normal((50, 500)) + 1j * rng.standard_normal((50, 500))
    pipe = build_feature_pipeline()
    X_out = pipe.fit_transform(X_iq)
    print(f"Pipeline output shape: {X_out.shape}")
    print(f"Mean: {X_out.mean():.4f}, Std: {X_out.std():.4f}")
