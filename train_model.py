"""HAVM Framework - Model Training Script.

Trains a ``RandomForestClassifier`` on synthetic heart-disease data and
persists the fitted model to disk.  Run as a standalone script::

    python train_model.py

The script will:
1. Generate synthetic data if it does not already exist.
2. Load and split the training data (80/20 stratified).
3. Fit a Random Forest with the hyper-parameters defined in ``config.py``.
4. Print accuracy and a full classification report.
5. Save the model via ``joblib`` to ``models/rf_model.joblib``.
"""

from __future__ import annotations

import os
import sys

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import config
import data_loader


def train_and_save_model() -> RandomForestClassifier:
    """End-to-end training pipeline.

    Returns
    -------
    RandomForestClassifier
        The fitted model instance.
    """
    # ---- 1. Ensure data exists ----
    if not os.path.isfile(config.TRAIN_DATA_PATH):
        print("[train_model] Data not found — generating synthetic datasets …")
        data_loader.generate_and_save_data()

    # ---- 2. Load data ----
    X_train, X_test, y_train, y_test = data_loader.load_training_data()
    print(f"[train_model] Training samples : {len(X_train)}")
    print(f"[train_model] Test samples     : {len(X_test)}")

    # ---- 3. Train ----
    clf = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ---- 4. Evaluate ----
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"])

    print(f"\n{'='*60}")
    print(f"  Training Accuracy : {accuracy:.4f}")
    print(f"{'='*60}")
    print(report)

    # ---- 5. Persist ----
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(clf, config.MODEL_PATH)
    print(f"[train_model] Model saved to {config.MODEL_PATH}")

    return clf


# ── CLI entry-point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        train_and_save_model()
    except Exception as exc:
        print(f"[train_model] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
