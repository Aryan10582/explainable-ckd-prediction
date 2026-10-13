"""Small shared helpers: environment capture, JSON I/O, metric scoring."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

CORE_PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost", "shap",
                 "matplotlib", "streamlit", "joblib", "pyyaml"]
METRICS = ["roc_auc", "accuracy", "precision", "recall", "f1", "specificity"]
METRIC_LABELS = {"roc_auc": "ROC-AUC", "accuracy": "Accuracy", "precision": "Precision",
                 "recall": "Recall", "f1": "F1-score", "specificity": "Specificity"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def env_info() -> dict:
    versions = {}
    for pkg in CORE_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor(), "packages": versions}


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def metrics_from_scores(y_true, p_ckd, threshold: float = 0.5) -> dict:
    """All project metrics, with CKD (label 1) as the positive class.

    Every model is thresholded on its CKD probability, so labels and scores are
    always consistent with each other.
    """
    y_true = np.asarray(y_true).astype(int)
    p_ckd = np.asarray(p_ckd, dtype=float)
    y_pred = (p_ckd >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, p_ckd),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
    }


def ckd_probability(estimator, X) -> np.ndarray:
    """Probability of the positive class (CKD = 1), checking the class order."""
    classes = list(estimator.classes_)
    if classes != [0, 1]:
        raise ValueError(f"Unexpected class order {classes}; CKD must be label 1.")
    return estimator.predict_proba(X)[:, 1]


def cv_scorer(estimator, X, y, threshold: float = 0.5) -> dict:
    """Multi-metric scorer for GridSearchCV / cross_validate (one predict_proba call)."""
    return metrics_from_scores(y, ckd_probability(estimator, X), threshold)
