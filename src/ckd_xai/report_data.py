"""One place that reads the saved experiment outputs for the paper, slides and docs.

Every number quoted in the written deliverables comes from here, so the text
cannot drift from results/ and models/.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import schema
from .config import LOGS_DIR, PROCESSED_DIR, SELECTED_META_PATH, SELECTION_PATH, SPLIT_MANIFEST_PATH, TABLES_DIR
from .models import DISPLAY_NAMES, MODEL_NAMES

ABBR = {"logistic_regression": "LR", "svm": "SVM", "random_forest": "RF", "xgboost": "XGBoost"}


def _j(path):
    return json.loads(path.read_text(encoding="utf-8"))


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def load_results() -> dict:
    R: dict = {}
    clean = _j(PROCESSED_DIR / "cleaning_summary.json")
    split = _j(SPLIT_MANIFEST_PATH)
    sel = _j(SELECTION_PATH)
    meta = _j(SELECTED_META_PATH)
    shapc = _j(LOGS_DIR / "shap_checks.json")
    lat = _j(LOGS_DIR / "latency.json")
    times = _j(LOGS_DIR / "training_times.json")
    manifest = _j(LOGS_DIR / "run_manifest.json")

    R["clean"], R["split"], R["selection"], R["meta"] = clean, split, sel, meta
    R["shap_checks"], R["latency"], R["train_times"], R["manifest"] = shapc, lat, times, manifest
    n_cells = clean["n_records_clean"] * len(schema.FEATURES)
    R["missing_pct_cells"] = 100 * clean["missing_cells_total"] / n_cells
    R["whitespace_cells"] = sum(clean["whitespace_fixed_cells"].values())

    cv = pd.read_csv(TABLES_DIR / "cv_model_comparison.csv").set_index("model")
    test = pd.read_csv(TABLES_DIR / "test_metrics.csv").set_index("model")
    abl = pd.read_csv(TABLES_DIR / "cv_resampling_ablation.csv")
    R["cv"], R["test"] = cv, test
    R["winner"] = sel["selected_model"]
    R["winner_name"] = DISPLAY_NAMES[R["winner"]]
    R["cv_auc_min"] = cv["mean_roc_auc"].min()
    R["cv_auc_max"] = cv["mean_roc_auc"].max()
    R["n_candidates"] = sum(t["n_candidates"] for t in times.values())
    R["n_fits"] = sum(t["n_fits"] for t in times.values())
    R["grid_seconds"] = sum(t["grid_search_seconds"] for t in times.values())

    piv = abl.pivot(index="model", columns="resampling")
    R["ablation"] = {m: {k: piv.loc[m, (f"mean_{k}", "SMOTENC")] - piv.loc[m, (f"mean_{k}", "none")]
                         for k in ("roc_auc", "accuracy", "recall", "f1", "specificity")} for m in MODEL_NAMES}
    R["ablation_table"] = abl
    probe = pd.read_csv(TABLES_DIR / "cv_missingness_probe.csv").iloc[0]
    R["probe_auc"], R["probe_auc_sd"] = probe["mean_roc_auc"], probe["std_roc_auc"]
    R["perm_auc"] = pd.read_csv(TABLES_DIR / "cv_permutation_check.csv")["mean_roc_auc"].mean()
    audit = pd.read_csv(TABLES_DIR / "leakage_audit.csv")
    R["audit_passed"], R["audit_total"] = int(audit.passed.sum()), len(audit)
    R["audit"] = audit
    oof = pd.read_csv(TABLES_DIR / "cv_oof_predictions.csv")
    from sklearn.metrics import roc_auc_score
    R["oof_auc"] = {m: roc_auc_score(oof.true_label, oof[f"{m}_p_ckd"]) for m in MODEL_NAMES}

    preds = pd.read_csv(TABLES_DIR / "test_predictions.csv").set_index(schema.ID_COLUMN)
    R["preds"] = preds
    errors = pd.read_csv(TABLES_DIR / "test_errors.csv")
    R["errors"] = errors
    non = preds[preds.true_label == 0]
    R["near_miss_id"] = int(non[f"{R['winner']}_p_ckd"].idxmax())
    R["near_miss_p"] = {m: float(preds.loc[R["near_miss_id"], f"{m}_p_ckd"]) for m in MODEL_NAMES}
    R["min_ckd_p"] = {m: float(preds.loc[preds.true_label == 1, f"{m}_p_ckd"].min()) for m in MODEL_NAMES}

    imp = pd.read_csv(TABLES_DIR / "shap_global_importance.csv")
    R["shap_imp"] = imp
    R["shap_top5"] = imp.label.head(5).tolist()
    R["shap_top5_share"] = imp.share_of_total_pct.head(5).sum()
    R["local"] = pd.read_csv(TABLES_DIR / "shap_local_examples.csv").set_index("example")
    sv = pd.read_csv(TABLES_DIR / "shap_values_test.csv", index_col=0)
    R["shap_values"] = sv

    cc = pd.read_csv(TABLES_DIR / "class_comparison_train.csv").set_index("feature")
    R["class_cmp"] = cc
    R["binary_zero_in_noncdk"] = [c for c in schema.BINARY if cc.loc[c, "Non-CKD"].startswith("0.0%")]
    mb = pd.read_csv(TABLES_DIR / "missingness_by_class_train.csv").set_index("feature")
    R["miss_by_class"] = mb
    corr = pd.read_csv(TABLES_DIR / "spearman_correlation_train.csv", index_col=0)
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    R["corr_top"] = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(3)

    data = pd.read_csv(PROCESSED_DIR / "ckd_clean.csv").set_index(schema.ID_COLUMN)
    R["records"] = data
    n_miss = data[schema.RAW_FEATURE_ORDER].isna().sum(axis=1)
    R["median_missing_by_class_train"] = None
    R["env"] = manifest["environment"]
    R["stage_seconds"] = manifest["stage_seconds"]
    R["pipeline_seconds"] = sum(manifest["stage_seconds"].values())
    R["n_missing_per_record"] = n_miss
    return R


def median_iqr(R, feature, cls):
    """'2.3 [1.4-4.6]' style text from the training class comparison table, without the n."""
    return R["class_cmp"].loc[feature, cls].split(" (n=")[0]
