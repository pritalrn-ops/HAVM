"""HAVM Framework - Synthetic Data Generation & Loading.

Generates synthetic heart-disease data that mimics the UCI Heart Disease
Dataset.  Two datasets are produced:

* **Training data** (700 samples) — drawn from realistic baseline
  distributions.
* **Incoming / production data** (200 samples) — contains subtle
  distribution drift *and* a handful of clearly anomalous records so that
  drift-detection and OOD-detection modules have something to find.

All random generation is seeded (``numpy.random.seed(42)``) for full
reproducibility.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


# ── helpers ────────────────────────────────────────────────────────────────

def _generate_base_features(
    n_samples: int,
    rng: np.random.RandomState,
    *,
    age_mean: float = 54.0,
    chol_mean: float = 246.0,
    trestbps_mean: float = 131.0,
) -> pd.DataFrame:
    """Return a DataFrame of ``n_samples`` rows with realistic feature
    distributions.  Caller can override means for drift injection."""

    age = np.clip(rng.normal(age_mean, 9, n_samples), 29, 77).astype(int)
    sex = (rng.random(n_samples) < 0.68).astype(int)
    cp = rng.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.17, 0.28, 0.08])
    trestbps = np.clip(
        rng.normal(trestbps_mean, 17, n_samples), 94, 200
    ).astype(int)
    chol = np.clip(
        rng.normal(chol_mean, 52, n_samples), 126, 564
    ).astype(int)
    fbs = (rng.random(n_samples) < 0.15).astype(int)
    restecg = rng.choice([0, 1, 2], n_samples, p=[0.50, 0.35, 0.15])
    thalach = np.clip(rng.normal(149, 23, n_samples), 71, 202).astype(int)
    exang = (rng.random(n_samples) < 0.33).astype(int)
    oldpeak = np.clip(rng.exponential(1.0, n_samples), 0, 6.2).round(1)
    slope = rng.choice([0, 1, 2], n_samples, p=[0.35, 0.45, 0.20])
    ca = rng.choice([0, 1, 2, 3], n_samples, p=[0.55, 0.25, 0.13, 0.07])
    thal = rng.choice([0, 1, 2], n_samples, p=[0.06, 0.52, 0.42])

    return pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    })


def _assign_target(df: pd.DataFrame, rng: np.random.RandomState) -> pd.Series:
    """Compute a correlated binary target using a simple logistic-like
    rule based on clinically-meaningful features."""

    logit = (
        0.04 * (df["age"] - 54)
        + 0.005 * (df["chol"] - 246)
        + 0.02 * (df["trestbps"] - 131)
        - 0.03 * (df["thalach"] - 149)
        + 0.6 * (df["cp"] == 0).astype(float)
        + 0.4 * df["exang"]
        + 0.3 * df["ca"]
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    # Add a little noise so the boundary isn't perfectly deterministic
    noisy_prob = np.clip(prob + rng.normal(0, 0.05, len(df)), 0.01, 0.99)
    return (rng.random(len(df)) < noisy_prob).astype(int)


def _create_anomalous_records(
    n_records: int, rng: np.random.RandomState
) -> pd.DataFrame:
    """Create obviously anomalous records with extreme / impossible
    feature values so that OOD detectors can flag them."""

    records = []
    for _ in range(n_records):
        records.append({
            "age": rng.choice([18, 19, 95, 99]),          # far outside 29-77
            "sex": rng.choice([0, 1]),
            "cp": rng.choice([0, 1, 2, 3]),
            "trestbps": rng.choice([60, 65, 250, 260]),    # extreme BP
            "chol": rng.choice([80, 90, 600, 700]),        # extreme chol
            "fbs": rng.choice([0, 1]),
            "restecg": rng.choice([0, 1, 2]),
            "thalach": rng.choice([50, 55, 220, 230]),     # extreme heart rate
            "exang": rng.choice([0, 1]),
            "oldpeak": rng.choice([0.0, 6.0, 6.2]),
            "slope": rng.choice([0, 1, 2]),
            "ca": rng.choice([0, 1, 2, 3]),
            "thal": rng.choice([0, 1, 2]),
            "target": rng.choice([0, 1]),
        })
    return pd.DataFrame(records)


# ── public API ─────────────────────────────────────────────────────────────

def generate_and_save_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate training and incoming datasets, persist them as CSV,
    and return the two DataFrames.

    Returns
    -------
    train_df : pd.DataFrame
        700-sample training set with target column.
    incoming_df : pd.DataFrame
        200-sample production set (190 drifted + 10 anomalous).
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)
    rng = np.random.RandomState(config.RANDOM_STATE)

    # ---- training data (baseline distributions) ----
    train_df = _generate_base_features(700, rng)
    train_df[config.TARGET_COLUMN] = _assign_target(train_df, rng)

    # ---- incoming / production data (with drift) ----
    incoming_df = _generate_base_features(
        190,
        rng,
        age_mean=54.0 + 5,       # +5 year age drift
        chol_mean=246.0 + 30,     # +30 cholesterol drift
        trestbps_mean=131.0 + 10, # +10 BP drift
    )
    incoming_df[config.TARGET_COLUMN] = _assign_target(incoming_df, rng)

    # inject 10 clearly anomalous records
    anomalies = _create_anomalous_records(10, rng)
    incoming_df = pd.concat([incoming_df, anomalies], ignore_index=True)

    # ---- persist ----
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    incoming_df.to_csv(config.INCOMING_DATA_PATH, index=False)

    print(f"[data_loader] Training data   : {len(train_df):>5} rows  -> {config.TRAIN_DATA_PATH}")
    print(f"[data_loader] Incoming data    : {len(incoming_df):>5} rows  -> {config.INCOMING_DATA_PATH}")

    return train_df, incoming_df


def load_training_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the training CSV and return an 80/20 train-test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if not os.path.isfile(config.TRAIN_DATA_PATH):
        print("[data_loader] Training CSV not found -- generating data ...")
        generate_and_save_data()

    df = pd.read_csv(config.TRAIN_DATA_PATH)
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def load_incoming_data() -> pd.DataFrame:
    """Load the incoming / production dataset.

    Returns
    -------
    pd.DataFrame  with feature columns **and** the target column.
    """
    if not os.path.isfile(config.INCOMING_DATA_PATH):
        print("[data_loader] Incoming CSV not found -- generating data ...")
        generate_and_save_data()

    return pd.read_csv(config.INCOMING_DATA_PATH)


# ── CLI entry-point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_df, incoming_df = generate_and_save_data()
    print(f"\nTraining data preview:\n{train_df.head()}")
    print(f"\nIncoming data preview:\n{incoming_df.head()}")
