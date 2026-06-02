"""
Data Drift Detection
====================
Detects distribution shifts between training and incoming data using the
two-sample Kolmogorov-Smirnov test on each feature column.

Usage:
    python drift_detection.py
"""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd
from scipy.stats import ks_2samp

from config import (
    DRIFT_P_VALUE_THRESHOLD,
    FEATURE_COLUMNS,
    INCOMING_DATA_PATH,
    TRAIN_DATA_PATH,
)

# ── Constants ────────────────────────────────────────────────────────────────
_OVERALL_DRIFT_PCT_THRESHOLD: float = 30.0  # percent of features that must drift


# ── Core Functions ───────────────────────────────────────────────────────────

def detect_drift(
    train_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    features: list[str] | None = None,
    p_threshold: float = DRIFT_P_VALUE_THRESHOLD,
) -> dict[str, Any]:
    """Run a KS test on each feature and summarise drift.

    Parameters
    ----------
    train_df : pd.DataFrame
        Reference (training) dataset.
    incoming_df : pd.DataFrame
        New incoming dataset to compare against the reference.
    features : list[str] | None
        Feature columns to test.  Defaults to ``FEATURE_COLUMNS``.
    p_threshold : float
        Significance level; a feature is flagged if its p-value falls below
        this threshold.

    Returns
    -------
    dict
        Keys: ``results`` (per-feature details), ``summary`` (aggregate
        counts), ``overall_drift`` (bool).

    Raises
    ------
    ValueError
        If either DataFrame is empty or a requested feature is missing.
    """
    if train_df.empty or incoming_df.empty:
        raise ValueError("Both train_df and incoming_df must be non-empty DataFrames.")

    features = features or FEATURE_COLUMNS

    missing_train = [f for f in features if f not in train_df.columns]
    missing_incoming = [f for f in features if f not in incoming_df.columns]
    if missing_train:
        raise ValueError(f"Features missing from training data: {missing_train}")
    if missing_incoming:
        raise ValueError(f"Features missing from incoming data: {missing_incoming}")

    results: list[dict[str, Any]] = []
    drifted_count = 0

    for feature in features:
        stat, p_value = ks_2samp(
            train_df[feature].dropna(),
            incoming_df[feature].dropna(),
        )
        drift_detected = bool(p_value < p_threshold)
        drifted_count += int(drift_detected)
        results.append(
            {
                "feature": feature,
                "statistic": round(float(stat), 6),
                "p_value": round(float(p_value), 6),
                "drift_detected": drift_detected,
            }
        )

    total_features = len(features)
    drift_percentage = (drifted_count / total_features * 100) if total_features else 0.0

    return {
        "results": results,
        "summary": {
            "total_features": total_features,
            "drifted_features": drifted_count,
            "drift_percentage": round(drift_percentage, 2),
        },
        "overall_drift": drift_percentage > _OVERALL_DRIFT_PCT_THRESHOLD,
    }


def get_drift_status(
    train_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
) -> dict[str, Any]:
    """Convenience wrapper that runs :func:`detect_drift` with config defaults.

    Parameters
    ----------
    train_df : pd.DataFrame
        Reference (training) dataset.
    incoming_df : pd.DataFrame
        Incoming dataset.

    Returns
    -------
    dict
        Same structure as :func:`detect_drift`.
    """
    return detect_drift(
        train_df,
        incoming_df,
        features=FEATURE_COLUMNS,
        p_threshold=DRIFT_P_VALUE_THRESHOLD,
    )


# ── Standalone Execution ─────────────────────────────────────────────────────

def _print_report(report: dict[str, Any]) -> None:
    """Pretty-print the drift report to stdout."""
    header = f"{'Feature':<14} {'KS Stat':>10} {'p-value':>10} {'Drift?':>8}"
    print("\n" + "=" * len(header))
    print("  DATA DRIFT DETECTION REPORT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for row in report["results"]:
        flag = "YES" if row["drift_detected"] else "no"
        print(
            f"{row['feature']:<14} {row['statistic']:>10.6f} "
            f"{row['p_value']:>10.6f} {flag:>8}"
        )

    summary = report["summary"]
    print("-" * len(header))
    print(
        f"Drifted features : {summary['drifted_features']} / "
        f"{summary['total_features']}  ({summary['drift_percentage']:.1f}%)"
    )
    overall = "DRIFT DETECTED" if report["overall_drift"] else "No significant overall drift"
    print(f"Overall status   : {overall}")
    print("=" * len(header) + "\n")


if __name__ == "__main__":
    try:
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        incoming_df = pd.read_csv(INCOMING_DATA_PATH)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    report = get_drift_status(train_df, incoming_df)
    _print_report(report)
