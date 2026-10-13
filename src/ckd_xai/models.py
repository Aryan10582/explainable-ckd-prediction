"""The four classifiers compared in the project."""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODEL_NAMES = ["logistic_regression", "svm", "random_forest", "xgboost"]
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "svm": "SVM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
# What the SHAP values of each model are expressed in.
SHAP_OUTPUT = {
    "logistic_regression": "log-odds of CKD",
    "svm": "probability of CKD",
    "random_forest": "probability of CKD",
    "xgboost": "log-odds of CKD",
}


def make_classifier(name: str, seed: int):
    if name == "logistic_regression":
        # liblinear supports both the L2 (l1_ratio=0) and L1 (l1_ratio=1) penalties in the grid.
        return LogisticRegression(solver="liblinear", max_iter=5000, random_state=seed)
    if name == "svm":
        # SVC itself gives decision scores, not probabilities. Platt (sigmoid) calibration is
        # fitted with an internal 5-fold CV on the *training* data only; ensemble=False keeps a
        # single SVC refitted on all training data plus one calibrator (scikit-learn >= 1.9 way of
        # writing the deprecated SVC(probability=True)).
        return CalibratedClassifierCV(SVC(kernel="rbf", random_state=seed), method="sigmoid",
                                      cv=5, ensemble=False)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
    if name == "xgboost":
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                             device="cpu", random_state=seed, n_jobs=1, verbosity=0)
    raise ValueError(f"Unknown model: {name}")
