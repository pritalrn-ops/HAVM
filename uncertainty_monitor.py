"""
Uncertainty Monitoring
======================
Flags predictions where the model's confidence (max class probability) falls
below a configurable threshold, indicating epistemic or aleatoric uncertainty.

Usage:
    python uncertainty_monitor.py
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import (
    FEATURE_COLUMNS,
    INCOMING_DATA_PATH,
    MODEL_PATH,
    UNCERTAINTY_CONFIDENCE_THRESHOLD,
)

# ── Constants ────────────────────────────────────────────────────────────────
_HIGH_UNCERTAINTY_PCT_THRESHOLD: float = 15.0


# ── Core Functions ───────────────────────────────────────────────────────────

def assess_uncertainty(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    threshold: float = UNCERTAINTY_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate prediction uncertainty for a set of samples.

    Confidence is defined as the maximum predicted class probability for each
    sample.  Samples whose confidence is below *threshold* are flagged as
    uncertain.

    Parameters
    ----------
    model : RandomForestClassifier
        A fitted scikit-learn classifier that exposes ``predict_proba``.
    X : pd.DataFrame
        Feature matrix (must match the model's expected features).
    threshold : float
        Confidence level below which a prediction is considered uncertain.

    Returns
    -------
    dict
        Keys: ``confidences``, ``predictions``, ``uncertain_indices``,
        ``uncertain_count``, ``total_count``, ``uncertain_percentage``,
        ``mean_confidence``, ``min_confidence``, ``high_uncertainty``.

    Raises
    ------
    ValueError
        If ``X`` is empty.
    """
    if X.empty:
        raise ValueError("Feature matrix X must be a non-empty DataFrame.")

    probas: np.ndarray = model.predict_proba(X)
    confidences: np.ndarray = probas.max(axis=1)
    predictions: np.ndarray = model.predict(X)

    uncertain_mask = confidences < threshold
    uncertain_indices = list(X.index[uncertain_mask])
    uncertain_count = int(uncertain_mask.sum())
    total_count = len(X)
    uncertain_percentage = (uncertain_count / total_count * 100) if total_count else 0.0

    return {
        "confidences": confidences,
        "predictions": predictions,
        "uncertain_indices": uncertain_indices,
        "uncertain_count": uncertain_count,
        "total_count": total_count,
        "uncertain_percentage": round(uncertain_percentage, 2),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "min_confidence": round(float(np.min(confidences)), 4),
        "high_uncertainty": uncertain_percentage > _HIGH_UNCERTAINTY_PCT_THRESHOLD,
    }


def get_uncertainty_status(
    model: RandomForestClassifier,
    incoming_df: pd.DataFrame,
) -> dict[str, Any]:
    """Convenience wrapper using config defaults.

    Extracts ``FEATURE_COLUMNS`` from *incoming_df* and delegates to
    :func:`assess_uncertainty`.

    Parameters
    ----------
    model : RandomForestClassifier
        Fitted classifier.
    incoming_df : pd.DataFrame
        Incoming dataset (must contain ``FEATURE_COLUMNS``).

    Returns
    -------
    dict
        Same structure as :func:`assess_uncertainty`.

    Raises
    ------
    ValueError
        If required feature columns are missing from *incoming_df*.
    """
    missing = [f for f in FEATURE_COLUMNS if f not in incoming_df.columns]
    if missing:
        raise ValueError(f"Features missing from incoming data: {missing}")

    X = incoming_df[FEATURE_COLUMNS]
    return assess_uncertainty(
        model,
        X,
        threshold=UNCERTAINTY_CONFIDENCE_THRESHOLD,
    )


# ── Standalone Execution ─────────────────────────────────────────────────────

def _print_report(report: dict[str, Any]) -> None:
    """Pretty-print the uncertainty report to stdout."""
    width = 52
    print("\n" + "=" * width)
    print("  PREDICTION UNCERTAINTY REPORT")
    print("=" * width)
    print(f"  Total samples         : {report['total_count']}")
    print(f"  Uncertain samples     : {report['uncertain_count']}")
    print(f"  Uncertain percentage  : {report['uncertain_percentage']:.2f}%")
    print(f"  Mean confidence       : {report['mean_confidence']:.4f}")
    print(f"  Min confidence        : {report['min_confidence']:.4f}")
    status = "HIGH UNCERTAINTY" if report["high_uncertainty"] else "Acceptable uncertainty"
    print(f"  Status                : {status}")

    if report["uncertain_indices"]:
        shown = report["uncertain_indices"][:20]
        extra = len(report["uncertain_indices"]) - 20
        suffix = f"  ... and {extra} more" if extra > 0 else ""
        print(f"  Uncertain indices     : {shown}{suffix}")

    # Confidence distribution summary
    confs = report["confidences"]
    print(f"\n  Confidence distribution:")
    bins = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        count = int(((confs >= lo) & (confs < hi)).sum())
        label = f"[{lo:.1f}, {hi:.1f})" if hi < 1.01 else f"[{lo:.1f}, 1.0]"
        if hi == 1.01:
            label = f"[{lo:.1f}, 1.0]"
            count = int(((confs >= lo) & (confs <= 1.0)).sum())
        print(f"    {label} : {count}")

    print("=" * width + "\n")


if __name__ == "__main__":
    import joblib

    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        incoming_df = pd.read_csv(INCOMING_DATA_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Incoming data not found at {INCOMING_DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    report = get_uncertainty_status(model, incoming_df)
    _print_report(report)
