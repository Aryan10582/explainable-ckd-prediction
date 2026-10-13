"""Cross-validated tuning and predeclared model selection (training data only).

Nothing in this module can see the test partition: it receives the training
records only. The selection is written to models/selection.json before the
evaluation module ever loads the test records.
"""
from __future__ import annotations

import time
from functools import partial

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate

from . import schema
from .config import (MODELS_DIR, SELECTED_META_PATH, SELECTED_MODEL_PATH, SELECTION_PATH, TABLES_DIR,
                     model_path)
from .data import sha256
from .models import DISPLAY_NAMES, MODEL_NAMES, SHAP_OUTPUT
from .preprocessing import build_pipeline
from .utils import METRICS, cv_scorer, env_info, now_utc, write_json


def make_cv(cfg: dict) -> StratifiedKFold:
    return StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True,
                           random_state=cfg["project"]["random_seed"])


def _scorer(cfg):
    return partial(cv_scorer, threshold=cfg["selection"]["threshold"])


def tune_model(name: str, X: pd.DataFrame, y: pd.Series, cfg: dict) -> tuple[GridSearchCV, float]:
    grid = cfg["grids"][name]
    search = GridSearchCV(build_pipeline(name, cfg, resample=True), grid, scoring=_scorer(cfg),
                          refit=cfg["selection"]["primary_metric"], cv=make_cv(cfg),
                          n_jobs=cfg["cv"]["n_jobs"], error_score="raise")
    t0 = time.perf_counter()
    search.fit(X, y)
    return search, time.perf_counter() - t0


def _search_table(name: str, search: GridSearchCV) -> pd.DataFrame:
    res = pd.DataFrame(search.cv_results_)
    keep = ["params"] + [f"{s}_test_{m}" for m in METRICS for s in ("mean", "std")]
    out = res[keep].copy()
    out.insert(0, "model", name)
    out["params"] = out["params"].map(lambda p: {k.split("__")[-1]: v for k, v in p.items()})
    out["rank_by_roc_auc"] = res["rank_test_roc_auc"]
    out["mean_fit_time_s"] = res["mean_fit_time"]
    return out.sort_values("rank_by_roc_auc")


def _best_fold_scores(name: str, search: GridSearchCV, n_splits: int) -> pd.DataFrame:
    i = search.best_index_
    rows = []
    for k in range(n_splits):
        row = {"model": name, "fold": k + 1}
        for m in METRICS:
            row[m] = search.cv_results_[f"split{k}_test_{m}"][i]
        rows.append(row)
    return pd.DataFrame(rows)


def select_model(summary: pd.DataFrame, cfg: dict) -> tuple[str, pd.DataFrame, str]:
    """Apply the predeclared selection rule from config.yaml."""
    sel = cfg["selection"]
    df = summary.set_index("model")
    best_auc = df["mean_roc_auc"].max()
    df["within_auc_tolerance"] = df["mean_roc_auc"] >= best_auc - sel["auc_tie_tolerance"]
    df["simplicity_rank"] = [sel["simplicity_order"].index(m) for m in df.index]
    # Tied models (inside the tolerance band) are ordered by the tie-breakers, then simplicity;
    # models outside the band are ordered by AUC alone and always rank below the tied group.
    sort_cols = [f"mean_{t}" for t in sel["tie_breakers"]] + ["simplicity_rank"]
    inside = df[df.within_auc_tolerance].sort_values(
        sort_cols, ascending=[False] * len(sel["tie_breakers"]) + [True])
    outside = df[~df.within_auc_tolerance].sort_values("mean_roc_auc", ascending=False)
    ranked = pd.concat([inside, outside])
    ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
    winner = ranked.index[0]
    tied = inside.index.tolist()
    if len(tied) == 1:
        reason = (f"{DISPLAY_NAMES[winner]} has the highest mean CV ROC-AUC "
                  f"({df.loc[winner, 'mean_roc_auc']:.4f}); no other model is within "
                  f"{sel['auc_tie_tolerance']} of it.")
    else:
        reason = (f"Models within {sel['auc_tie_tolerance']} of the best mean CV ROC-AUC "
                  f"({best_auc:.4f}) were treated as tied: {', '.join(DISPLAY_NAMES[m] for m in tied)}. "
                  f"Among them, {DISPLAY_NAMES[winner]} ranks first on the predeclared tie-breakers "
                  f"(mean CV {', then '.join(sel['tie_breakers'])}"
                  f"{', then simplicity' if len(tied) > 1 else ''}).")
    return winner, ranked.reset_index(), reason


def resampling_ablation(best_params: dict, X, y, cfg) -> pd.DataFrame:
    """Same tuned hyper-parameters and folds, with and without SMOTENC."""
    rows = []
    for name, params in best_params.items():
        for resample in (True, False):
            pipe = build_pipeline(name, cfg, resample=resample, params=params)
            cv = cross_validate(pipe, X, y, cv=make_cv(cfg), scoring=_scorer(cfg),
                                n_jobs=cfg["cv"]["n_jobs"])
            row = {"model": name, "resampling": "SMOTENC" if resample else "none"}
            for m in METRICS:
                row[f"mean_{m}"] = float(np.mean(cv[f"test_{m}"]))
                row[f"std_{m}"] = float(np.std(cv[f"test_{m}"]))
            rows.append(row)
    return pd.DataFrame(rows)


def missingness_probe(X, y, cfg) -> pd.DataFrame:
    """How well does *which values are missing* predict the class, ignoring the values?

    A logistic regression sees only 24 missing-value indicators. A high score
    means missingness itself carries class information (a possible shortcut).
    """
    M = X.isna().astype(float).add_suffix("_missing")
    M["n_missing"] = M.sum(axis=1)
    cv = cross_validate(LogisticRegression(max_iter=5000), M, y, cv=make_cv(cfg), scoring=_scorer(cfg))
    row = {"probe": "missing-value indicators only (logistic regression)"}
    for m in METRICS:
        row[f"mean_{m}"] = float(np.mean(cv[f"test_{m}"]))
        row[f"std_{m}"] = float(np.std(cv[f"test_{m}"]))
    return pd.DataFrame([row])


def permutation_check(name: str, params: dict, X, y, cfg, n_repeats: int = 5) -> pd.DataFrame:
    """Train on shuffled labels: a leak-free pipeline should fall to chance (AUC ~ 0.5)."""
    rng = np.random.default_rng(cfg["project"]["random_seed"])
    rows = []
    for r in range(n_repeats):
        y_perm = pd.Series(rng.permutation(y.values), index=y.index)
        cv = cross_validate(build_pipeline(name, cfg, resample=True, params=params), X, y_perm,
                            cv=make_cv(cfg), scoring=_scorer(cfg), n_jobs=cfg["cv"]["n_jobs"])
        rows.append({"model": name, "repeat": r + 1, "mean_roc_auc": float(np.mean(cv["test_roc_auc"]))})
    return pd.DataFrame(rows)


def run_training(X_train: pd.DataFrame, y_train: pd.Series, cfg: dict, train_ids_digest: str,
                 log=print) -> dict:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    n_splits = cfg["cv"]["n_splits"]
    summaries, folds, searches, best_params, timings = [], [], {}, {}, {}

    for name in MODEL_NAMES:
        log(f"  tuning {DISPLAY_NAMES[name]} ...")
        search, secs = tune_model(name, X_train, y_train, cfg)
        searches[name] = search
        best_params[name] = search.best_params_
        timings[name] = {"grid_search_seconds": round(secs, 2),
                         "n_candidates": len(search.cv_results_["params"]),
                         "n_fits": len(search.cv_results_["params"]) * n_splits,
                         "refit_seconds": round(float(search.refit_time_), 3)}
        _search_table(name, search).to_csv(TABLES_DIR / f"cv_search_{name}.csv", index=False)
        folds.append(_best_fold_scores(name, search, n_splits))
        i = search.best_index_
        row = {"model": name, "display_name": DISPLAY_NAMES[name],
               "best_params": {k.split("__")[-1]: v for k, v in search.best_params_.items()}}
        for m in METRICS:
            row[f"mean_{m}"] = float(search.cv_results_[f"mean_test_{m}"][i])
            row[f"std_{m}"] = float(search.cv_results_[f"std_test_{m}"][i])
        summaries.append(row)
        joblib.dump(search.best_estimator_, model_path(name))
        log(f"    best CV ROC-AUC {row['mean_roc_auc']:.4f} +/- {row['std_roc_auc']:.4f} "
            f"({secs:.1f}s, {timings[name]['n_fits']} fits)")

    summary = pd.DataFrame(summaries)
    summary.to_csv(TABLES_DIR / "cv_model_comparison.csv", index=False)
    pd.concat(folds).to_csv(TABLES_DIR / "cv_fold_scores.csv", index=False)

    winner, ranked, reason = select_model(summary, cfg)
    ranked.to_csv(TABLES_DIR / "cv_selection_ranking.csv", index=False)
    selection = {
        "selected_model": winner,
        "display_name": DISPLAY_NAMES[winner],
        "reason": reason,
        "rule": cfg["selection"],
        "ranking": ranked[["model", "selection_rank", "mean_roc_auc", "mean_recall", "mean_f1",
                           "within_auc_tolerance"]].to_dict(orient="records"),
        "test_set_used": False,
        "decided_at_utc": now_utc(),
    }
    write_json(selection, SELECTION_PATH)
    log(f"  selected: {DISPLAY_NAMES[winner]} -- {reason}")

    log("  resampling ablation (SMOTENC vs none, same folds and hyper-parameters) ...")
    resampling_ablation(best_params, X_train, y_train, cfg).to_csv(
        TABLES_DIR / "cv_resampling_ablation.csv", index=False)
    log("  missingness-only probe ...")
    missingness_probe(X_train, y_train, cfg).to_csv(TABLES_DIR / "cv_missingness_probe.csv", index=False)
    log("  out-of-fold CV predictions (for training-set ROC curves) ...")
    oof = pd.DataFrame({schema.ID_COLUMN: X_train.index, "true_label": y_train.values})
    for name, params in best_params.items():
        oof[f"{name}_p_ckd"] = cross_val_predict(build_pipeline(name, cfg, resample=True, params=params),
                                                 X_train, y_train, cv=make_cv(cfg), method="predict_proba",
                                                 n_jobs=cfg["cv"]["n_jobs"])[:, 1]
    oof.to_csv(TABLES_DIR / "cv_oof_predictions.csv", index=False)
    log("  label-permutation sanity check ...")
    permutation_check(winner, best_params[winner], X_train, y_train, cfg).to_csv(
        TABLES_DIR / "cv_permutation_check.csv", index=False)

    best = searches[winner].best_estimator_
    joblib.dump(best, SELECTED_MODEL_PATH)
    meta = {
        "model": winner,
        "display_name": DISPLAY_NAMES[winner],
        "hyperparameters": {k.split("__")[-1]: v for k, v in best_params[winner].items()},
        "pipeline_steps": [n for n, _ in best.steps],
        "classifier_class": type(best.named_steps["clf"]).__name__,
        "threshold": cfg["selection"]["threshold"],
        "positive_class": {"label": 1, "name": schema.POSITIVE_LABEL},
        "target_mapping": schema.TARGET_MAPPING,
        "raw_feature_order": schema.RAW_FEATURE_ORDER,
        "model_feature_order": schema.MODEL_FEATURES,
        "smotenc_categorical_columns": [schema.MODEL_FEATURES[i] for i in schema.SMOTENC_CATEGORICAL_IDX],
        "shap_output": SHAP_OUTPUT[winner],
        "cv_summary": {m: summary.set_index("model").loc[winner, f"mean_{m}"] for m in METRICS},
        "n_training_records": int(len(X_train)),
        "training_class_counts": {int(k): int(v) for k, v in y_train.value_counts().items()},
        "train_ids_sha256": train_ids_digest,
        "random_seed": cfg["project"]["random_seed"],
        "trained_at_utc": now_utc(),
        "model_file_sha256": sha256(SELECTED_MODEL_PATH),
        "environment": env_info(),
    }
    write_json(meta, SELECTED_META_PATH)
    write_json(timings, TABLES_DIR.parent / "logs" / "training_times.json")
    pd.DataFrame([{"model": DISPLAY_NAMES[m], **{k.split("__")[-1]: v for k, v in p.items()}}
                  for m, p in best_params.items()]).to_csv(TABLES_DIR / "selected_hyperparameters.csv",
                                                          index=False)
    return {"selection": selection, "summary": summary, "timings": timings}
