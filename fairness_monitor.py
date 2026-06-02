"""
Fairness Monitoring
===================
Evaluates classification performance across demographic groups defined by a
sensitive attribute and flags metric disparities that exceed a configurable
threshold.

Usage:
    python fairness_monitor.py
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import (
    FAIRNESS_DISPARITY_THRESHOLD,
    FEATURE_COLUMNS,
    INCOMING_DATA_PATH,
    MODEL_PATH,
    SENSITIVE_ATTRIBUTE,
    TARGET_COLUMN,
)

# ── Group Label Mapping ──────────────────────────────────────────────────────
_SEX_LABELS: dict[int, str] = {0: "Female (sex=0)", 1: "Male (sex=1)"}


# ── Core Functions ───────────────────────────────────────────────────────────

def assess_fairness(
    model: RandomForestClassifier,
    df: pd.DataFrame,
    sensitive_attr: str = SENSITIVE_ATTRIBUTE,
    features: list[str] | None = None,
    target: str = TARGET_COLUMN,
    disparity_threshold: float = FAIRNESS_DISPARITY_THRESHOLD,
) -> dict[str, Any]:
    """Compute per-group metrics and cross-group disparities.

    Parameters
    ----------
    model : RandomForestClassifier
        Fitted classifier.
    df : pd.DataFrame
        Dataset containing features, the sensitive attribute, and ground-truth
        labels.
    sensitive_attr : str
        Column name of the sensitive/protected attribute.
    features : list[str] | None
        Feature columns for prediction.  Defaults to ``FEATURE_COLUMNS``.
    target : str
        Column name of the ground-truth label.
    disparity_threshold : float
        Maximum allowed absolute difference between any two groups on any
        metric before a fairness violation is raised.

    Returns
    -------
    dict
        Keys: ``group_metrics``, ``disparities``, ``fairness_violated``,
        ``max_disparity_metric``, ``max_disparity_value``.

    Raises
    ------
    ValueError
        If the DataFrame is empty, required columns are missing, or fewer
        than two groups exist for the sensitive attribute.
    """
    if df.empty:
        raise ValueError("DataFrame must be non-empty.")

    features = features or FEATURE_COLUMNS

    required_cols = set(features) | {sensitive_attr, target}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {sorted(missing)}")

    groups = sorted(df[sensitive_attr].unique())
    if len(groups) < 2:
        raise ValueError(
            f"Need at least 2 groups for fairness comparison; "
            f"found {len(groups)} unique value(s) in '{sensitive_attr}'."
        )

    # ── Per-group metrics ─────────────────────────────────────────────────
    group_metrics: dict[str, dict[str, Any]] = {}

    for group_val in groups:
        mask = df[sensitive_attr] == group_val
        group_df = df.loc[mask]
        X_group = group_df[features]
        y_true = group_df[target]
        y_pred = model.predict(X_group)

        # Use zero_division=0 to handle edge cases where a class is absent
        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "count": int(len(group_df)),
        }

        label = _SEX_LABELS.get(group_val, f"{sensitive_attr}={group_val}")
        group_metrics[label] = metrics

    # ── Disparities (pairwise for exactly two groups) ─────────────────────
    metric_names = ["accuracy", "precision", "recall", "f1"]
    group_labels = list(group_metrics.keys())

    disparities: dict[str, float] = {}
    for metric in metric_names:
        vals = [group_metrics[g][metric] for g in group_labels]
        disparities[metric] = round(float(abs(vals[0] - vals[1])), 4)

    max_metric = max(disparities, key=disparities.get)  # type: ignore[arg-type]
    max_value = disparities[max_metric]

    return {
        "group_metrics": group_metrics,
        "disparities": disparities,
        "fairness_violated": any(v > disparity_threshold for v in disparities.values()),
        "max_disparity_metric": max_metric,
        "max_disparity_value": max_value,
    }


def get_fairness_status(
    model: RandomForestClassifier,
    incoming_df: pd.DataFrame,
) -> dict[str, Any]:
    """Convenience wrapper using config defaults.

    Parameters
    ----------
    model : RandomForestClassifier
        Fitted classifier.
    incoming_df : pd.DataFrame
        Incoming dataset with features, sensitive attribute, and labels.

    Returns
    -------
    dict
        Same structure as :func:`assess_fairness`.
    """
    return assess_fairness(
        model,
        incoming_df,
        sensitive_attr=SENSITIVE_ATTRIBUTE,
        features=FEATURE_COLUMNS,
        target=TARGET_COLUMN,
        disparity_threshold=FAIRNESS_DISPARITY_THRESHOLD,
    )


# ── Standalone Execution ─────────────────────────────────────────────────────

def _print_report(report: dict[str, Any]) -> None:
    """Pretty-print the fairness report to stdout."""
    width = 64
    print("\n" + "=" * width)
    print("  FAIRNESS MONITORING REPORT")
    print("=" * width)

    # ── Group Metrics Table ───────────────────────────────────────────────
    header = (
        f"  {'Group':<20} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
        f"{'F1':>7} {'Count':>7}"
    )
    print(header)
    print("  " + "-" * (width - 4))

    for group, metrics in report["group_metrics"].items():
        print(
            f"  {group:<20} {metrics['accuracy']:>7.4f} "
            f"{metrics['precision']:>7.4f} {metrics['recall']:>7.4f} "
            f"{metrics['f1']:>7.4f} {metrics['count']:>7}"
        )

    # ── Disparities Table ─────────────────────────────────────────────────
    print("\n  " + "-" * (width - 4))
    print(f"  {'Metric':<14} {'Disparity':>10} {'Exceeds?':>10}")
    print("  " + "-" * (width - 4))

    for metric, value in report["disparities"].items():
        exceeds = "YES" if value > FAIRNESS_DISPARITY_THRESHOLD else "no"
        print(f"  {metric:<14} {value:>10.4f} {exceeds:>10}")

    # ── Overall status ────────────────────────────────────────────────────
    print("  " + "-" * (width - 4))
    status = "FAIRNESS VIOLATION" if report["fairness_violated"] else "Fair"
    print(f"  Overall status        : {status}")
    print(
        f"  Max disparity         : {report['max_disparity_metric']} "
        f"({report['max_disparity_value']:.4f})"
    )
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

    report = get_fairness_status(model, incoming_df)
    _print_report(report)
