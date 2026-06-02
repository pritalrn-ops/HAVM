"""
Out-of-Distribution (OOD) Detection
====================================
Identifies incoming samples that fall outside the training distribution
by fitting an Isolation Forest on the reference data and scoring new
observations.

Usage:
    python ood_detection.py
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import (
    FEATURE_COLUMNS,
    INCOMING_DATA_PATH,
    OOD_CONTAMINATION,
    RANDOM_STATE,
    TRAIN_DATA_PATH,
)

# ── Constants ────────────────────────────────────────────────────────────────
_OOD_ALERT_PCT_THRESHOLD: float = 5.0  # fire alert when OOD % exceeds this


# ── Core Functions ───────────────────────────────────────────────────────────

def train_ood_detector(
    train_df: pd.DataFrame,
    features: list[str] | None = None,
    contamination: float = OOD_CONTAMINATION,
) -> IsolationForest:
    """Fit an Isolation Forest on the training feature matrix.

    Parameters
    ----------
    train_df : pd.DataFrame
        Reference training data.
    features : list[str] | None
        Columns to use as features.  Defaults to ``FEATURE_COLUMNS``.
    contamination : float
        Expected proportion of outliers in the training set.

    Returns
    -------
    IsolationForest
        The fitted detector.

    Raises
    ------
    ValueError
        If the DataFrame is empty or required columns are missing.
    """
    if train_df.empty:
        raise ValueError("train_df must be a non-empty DataFrame.")

    features = features or FEATURE_COLUMNS

    missing = [f for f in features if f not in train_df.columns]
    if missing:
        raise ValueError(f"Features missing from training data: {missing}")

    model = IsolationForest(
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_estimators=100,
    )
    model.fit(train_df[features])
    return model


def detect_ood(
    ood_model: IsolationForest,
    incoming_df: pd.DataFrame,
    features: list[str] | None = None,
) -> dict[str, Any]:
    """Score incoming data against the trained OOD detector.

    Parameters
    ----------
    ood_model : IsolationForest
        A fitted Isolation Forest model.
    incoming_df : pd.DataFrame
        New data to evaluate.
    features : list[str] | None
        Feature columns.  Defaults to ``FEATURE_COLUMNS``.

    Returns
    -------
    dict
        Keys: ``predictions`` (1 = inlier, -1 = outlier),
        ``anomaly_scores``, ``ood_indices``, ``ood_count``,
        ``total_count``, ``ood_percentage``, ``ood_detected``.

    Raises
    ------
    ValueError
        If the DataFrame is empty or required columns are missing.
    """
    if incoming_df.empty:
        raise ValueError("incoming_df must be a non-empty DataFrame.")

    features = features or FEATURE_COLUMNS

    missing = [f for f in features if f not in incoming_df.columns]
    if missing:
        raise ValueError(f"Features missing from incoming data: {missing}")

    X = incoming_df[features]
    predictions: np.ndarray = ood_model.predict(X)
    anomaly_scores: np.ndarray = ood_model.decision_function(X)

    ood_mask = predictions == -1
    ood_indices = list(incoming_df.index[ood_mask])
    ood_count = int(ood_mask.sum())
    total_count = len(incoming_df)
    ood_percentage = (ood_count / total_count * 100) if total_count else 0.0

    return {
        "predictions": predictions,
        "anomaly_scores": anomaly_scores,
        "ood_indices": ood_indices,
        "ood_count": ood_count,
        "total_count": total_count,
        "ood_percentage": round(ood_percentage, 2),
        "ood_detected": ood_percentage > _OOD_ALERT_PCT_THRESHOLD,
    }


def get_ood_status(
    train_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
) -> dict[str, Any]:
    """Convenience wrapper: train detector then score incoming data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Reference training data.
    incoming_df : pd.DataFrame
        Incoming data to evaluate.

    Returns
    -------
    dict
        Same structure as :func:`detect_ood`.
    """
    detector = train_ood_detector(
        train_df,
        features=FEATURE_COLUMNS,
        contamination=OOD_CONTAMINATION,
    )
    return detect_ood(detector, incoming_df, features=FEATURE_COLUMNS)


# ── Standalone Execution ─────────────────────────────────────────────────────

def _print_report(report: dict[str, Any]) -> None:
    """Pretty-print the OOD report to stdout."""
    width = 50
    print("\n" + "=" * width)
    print("  OUT-OF-DISTRIBUTION DETECTION REPORT")
    print("=" * width)
    print(f"  Total samples      : {report['total_count']}")
    print(f"  OOD samples        : {report['ood_count']}")
    print(f"  OOD percentage     : {report['ood_percentage']:.2f}%")
    status = "OOD DETECTED" if report["ood_detected"] else "No significant OOD"
    print(f"  Status             : {status}")

    if report["ood_indices"]:
        shown = report["ood_indices"][:20]
        suffix = f"  ... and {len(report['ood_indices']) - 20} more" if len(report["ood_indices"]) > 20 else ""
        print(f"  OOD sample indices : {shown}{suffix}")

    scores = report["anomaly_scores"]
    print(f"\n  Anomaly score stats:")
    print(f"    Mean  : {float(np.mean(scores)):.4f}")
    print(f"    Std   : {float(np.std(scores)):.4f}")
    print(f"    Min   : {float(np.min(scores)):.4f}")
    print(f"    Max   : {float(np.max(scores)):.4f}")
    print("=" * width + "\n")


if __name__ == "__main__":
    try:
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        incoming_df = pd.read_csv(INCOMING_DATA_PATH)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    report = get_ood_status(train_df, incoming_df)
    _print_report(report)
