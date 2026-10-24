"""
AdverScan — Data Pipeline
Loads, cleans, and preprocesses CICIDS2017 or sample_data.csv.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "data")
FULL_DATASET = os.path.join(DATA_DIR, "cicids2017_wednesday.csv")
SAMPLE_DATASET = os.path.join(DATA_DIR, "sample_data.csv")

_NON_NUMERIC = [
    "Flow ID", "Source IP", "Destination IP", "Src IP", "Dst IP",
    "Timestamp", "timestamp", "src_ip", "dst_ip",
]


def auto_detect_dataset():
    """Return (path, type) for the best available dataset."""
    if os.path.exists(FULL_DATASET):
        size_mb = os.path.getsize(FULL_DATASET) / (1024 * 1024)
        print(f"  Found full CICIDS2017 dataset ({size_mb:.0f} MB)")
        return FULL_DATASET, "cicids2017"
    print("  Full dataset not found — using sample_data.csv (demo mode)")
    print("  WARNING: Results are illustrative only. See data/README.md to download the full dataset.")
    return SAMPLE_DATASET, "sample"


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()  # CICIDS2017 has leading/trailing spaces
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop non-numeric / identifier columns
    drop_cols = [c for c in _NON_NUMERIC if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Ensure Label is last
    if "Label" in df.columns:
        label = df.pop("Label")
        df["Label"] = label

    # Replace inf / -inf with NaN then drop
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def _encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Label"] = (df["Label"].astype(str).str.upper() != "BENIGN").astype(int)
    return df


def _select_features(X: np.ndarray, n: int = 20):
    """Keep the n highest-variance features (after removing zero-variance)."""
    selector = VarianceThreshold(threshold=1e-8)
    X_var = selector.fit_transform(X)
    variances = X_var.var(axis=0)
    top_idx = np.argsort(variances)[-n:][::-1]
    return X_var[:, top_idx], selector, top_idx


def get_feature_bounds(X_train: np.ndarray):
    """Return (X_min, X_max) per-feature from training data."""
    return X_train.min(axis=0), X_train.max(axis=0)


def load_data(path: str = None, n_features: int = 20, test_size: float = 0.2,
              random_state: int = 42):
    """
    Main entry point.  Returns:
        X_train, X_test, y_train, y_test  (float32 numpy arrays, standardised)
        scaler                             (fitted StandardScaler)
    """
    if path is None:
        path, _ = auto_detect_dataset()

    print(f"  Loading {os.path.basename(path)} …")
    df = _load_csv(path)
    df = _clean(df)
    df = _encode_labels(df)

    n_benign = int((df["Label"] == 0).sum())
    n_attack = int((df["Label"] == 1).sum())
    print(f"  Rows: {len(df):,}  |  Benign: {n_benign:,}  |  Attack: {n_attack:,}")

    X = df.drop("Label", axis=1).values.astype(np.float32)
    y = df["Label"].values.astype(np.int32)

    X_sel, var_selector, feat_idx = _select_features(X, n=n_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler
