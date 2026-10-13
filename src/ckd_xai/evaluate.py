"""Final evaluation of the frozen pipelines on the untouched test partition.

Runs only after models/selection.json exists, so the test scores cannot have
influenced model choice, hyper-parameters, features or the threshold.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_curve)

from . import schema
from .config import (FIGURES_DIR, SELECTED_MODEL_PATH, SELECTION_PATH, SPLIT_MANIFEST_PATH, TABLES_DIR,
                     model_path)
from .eda import CAPTIONS_PATH, _captions
from .models import DISPLAY_NAMES, MODEL_NAMES
from .plotting import CLASS_COLORS, COL_W, INK, MODEL_COLORS, MODEL_STYLES, MUTED, apply_style, save
from .preprocessing import transform_features
from .utils import METRICS, ckd_probability, metrics_from_scores, now_utc, read_json, write_json


def load_models() -> dict:
    return {m: joblib.load(model_path(m)) for m in MODEL_NAMES}


def exact_ci(k: int, n: int, level: float) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    ci = binomtest(k, n).proportion_ci(confidence_level=level, method="exact")
    return float(ci.low), float(ci.high)


def _fast_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Vectorised version of utils.metrics_from_scores (same definitions) for the bootstrap."""
    from scipy.stats import rankdata

    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    ranks = rankdata(p)
    auc = (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)       # Mann-Whitney U / (n1 n0)
    yhat = p >= threshold
    tp, fp = np.sum(yhat & pos), np.sum(yhat & ~pos)
    fn, tn = n1 - tp, n0 - fp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / n1
    return {"roc_auc": auc, "accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "specificity": tn / n0}


def stratified_bootstrap(y, p, threshold, n_boot, level, seed) -> dict:
    """Percentile bootstrap, resampling within each class so every replicate has both classes."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    idx_pos, idx_neg = np.where(y == 1)[0], np.where(y == 0)[0]
    reps = {m: [] for m in METRICS}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(idx_pos, len(idx_pos)), rng.choice(idx_neg, len(idx_neg))])
        for m, v in _fast_metrics(y[idx], p[idx], threshold).items():
            reps[m].append(v)
    a = (1 - level) / 2
    return {m: (float(np.quantile(v, a)), float(np.quantile(v, 1 - a))) for m, v in reps.items()}


def evaluate(X_test: pd.DataFrame, y_test: pd.Series, cfg: dict, log=print) -> dict:
    if not SELECTION_PATH.exists():
        raise RuntimeError("Model selection must be completed (on training CV) before test evaluation.")
    selection = read_json(SELECTION_PATH)
    apply_style()
    caps = _captions()
    thr = cfg["selection"]["threshold"]
    level = cfg["evaluation"]["ci_level"]
    seed = cfg["project"]["random_seed"]
    models = load_models()
    y = y_test.values.astype(int)

    preds = pd.DataFrame({schema.ID_COLUMN: X_test.index, "true_label": y,
                          "true_class": [schema.CLASS_NAMES[v] for v in y]})
    rows, probs = [], {}
    for name, pipe in models.items():
        p = ckd_probability(pipe, X_test)
        probs[name] = p
        yhat = (p >= thr).astype(int)
        preds[f"{name}_p_ckd"] = p.round(6)
        preds[f"{name}_pred"] = yhat
        m = metrics_from_scores(y, p, thr)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        boot = stratified_bootstrap(y, p, thr, cfg["evaluation"]["bootstrap_resamples"], level, seed)
        row = {"model": name, "display_name": DISPLAY_NAMES[name],
               "selected": name == selection["selected_model"], "tp": tp, "fp": fp, "tn": tn, "fn": fn}
        for k in METRICS:
            row[k] = m[k]
            row[f"{k}_boot_lo"], row[f"{k}_boot_hi"] = boot[k]
        row["accuracy_exact_lo"], row["accuracy_exact_hi"] = exact_ci(tp + tn, len(y), level)
        row["recall_exact_lo"], row["recall_exact_hi"] = exact_ci(tp, tp + fn, level)
        row["specificity_exact_lo"], row["specificity_exact_hi"] = exact_ci(tn, tn + fp, level)
        row["precision_exact_lo"], row["precision_exact_hi"] = exact_ci(tp, tp + fp, level)
        row["average_precision"] = average_precision_score(y, p)
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLES_DIR / "test_metrics.csv", index=False)

    def err(row, name):
        t, pr = row["true_label"], row[f"{name}_pred"]
        return "TP" if t == pr == 1 else "TN" if t == pr == 0 else "FP" if pr == 1 else "FN"
    for name in models:
        preds[f"{name}_outcome"] = preds.apply(lambda r: err(r, name), axis=1)
    preds.to_csv(TABLES_DIR / "test_predictions.csv", index=False)

    errors = []
    for name in models:
        bad = preds[preds[f"{name}_outcome"].isin(["FP", "FN"])]
        for _, r in bad.iterrows():
            rec = X_test.loc[r[schema.ID_COLUMN]]
            errors.append({"model": name, schema.ID_COLUMN: int(r[schema.ID_COLUMN]),
                           "error_type": r[f"{name}_outcome"], "p_ckd": r[f"{name}_p_ckd"],
                           "n_missing_features": int(rec.isna().sum())})
    pd.DataFrame(errors, columns=["model", schema.ID_COLUMN, "error_type", "p_ckd",
                                  "n_missing_features"]).to_csv(TABLES_DIR / "test_errors.csv", index=False)

    _plot_confusion(metrics, caps)
    _plot_roc(y, probs, caps)
    _plot_pr(y, probs, caps)
    _plot_cv_vs_test(metrics, caps)
    _plot_cv_roc(caps)
    _plot_test_scores(y, probs, thr, caps)
    write_json(caps, CAPTIONS_PATH)
    audit = leakage_audit(X_test, models, cfg)
    log(metrics[["display_name", "roc_auc", "accuracy", "precision", "recall", "f1", "fp", "fn"]]
        .round(4).to_string(index=False))
    return {"metrics": metrics, "predictions": preds, "audit": audit}


# --------------------------------------------------------------------------- figures
def _plot_confusion(metrics, caps):
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.3))
    for ax, (_, r) in zip(axes.flat, metrics.iterrows()):
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()) * 1.25)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > cm.max() * 0.6 else INK, fontweight="bold")
        ax.set_xticks([0, 1], ["Non-CKD", "CKD"], fontsize=6.5)
        ax.set_yticks([0, 1], ["Non-CKD", "CKD"], fontsize=6.5)
        ax.set_xlabel("Predicted", fontsize=6.8, labelpad=1)
        ax.set_ylabel("True", fontsize=6.8, labelpad=1)
        ax.set_title(r.display_name + ("  (selected)" if r.selected else ""), fontsize=7.2)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    save(fig, FIGURES_DIR / "test_confusion_matrices.png", caps,
         "Confusion matrices of the four frozen pipelines on the held-out test set (n = 80: 50 CKD, "
         "30 non-CKD) at the default threshold of 0.5. Rows are true classes, columns predicted classes.")


def _plot_roc(y, probs, caps):
    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.7, linestyle=(0, (1, 2)))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = metrics_from_scores(y, p)["roc_auc"]
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], linestyle=MODEL_STYLES[name],
                label=f"{DISPLAY_NAMES[name]} (AUC {auc:.3f})", linewidth=1.5)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.01)
    ax.legend(loc="lower right")
    ax.set_title("ROC curves, held-out test set")
    save(fig, FIGURES_DIR / "test_roc_curves.png", caps,
         "Receiver operating characteristic curves of the four frozen pipelines on the held-out test set, "
         "with CKD as the positive class. Line styles distinguish the models; the dotted diagonal is chance.")


def _plot_pr(y, probs, caps):
    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    base = y.mean()
    ax.axhline(base, color=MUTED, linewidth=0.7, linestyle=(0, (1, 2)))
    ax.text(0.02, base - 0.04, f"CKD prevalence in test set = {base:.3f}", fontsize=6.5, color=MUTED)
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(rec, prec, color=MODEL_COLORS[name], linestyle=MODEL_STYLES[name],
                label=f"{DISPLAY_NAMES[name]} (AP {ap:.3f})", linewidth=1.5, drawstyle="steps-post")
    ax.set_xlabel("Recall (CKD)")
    ax.set_ylabel("Precision (CKD)")
    ax.set_ylim(base - 0.08, 1.02)
    ax.legend(loc="lower left")
    ax.set_title("Precision-recall curves, held-out test set")
    save(fig, FIGURES_DIR / "test_pr_curves.png", caps,
         "Precision-recall curves on the held-out test set with CKD as the positive class; AP is average "
         "precision. The dotted line is the precision of a classifier with no skill (test-set prevalence).")


def _plot_cv_vs_test(metrics, caps):
    cv = pd.read_csv(TABLES_DIR / "cv_model_comparison.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    names = list(metrics.model)
    x = np.arange(len(names))
    ax.errorbar(x - 0.12, cv.loc[names, "mean_roc_auc"], yerr=cv.loc[names, "std_roc_auc"], fmt="o",
                color="#2a78d6", markersize=4, capsize=2, linewidth=1, label="Training CV (mean +/- SD)")
    ax.errorbar(x + 0.12, metrics.roc_auc,
                yerr=[metrics.roc_auc - metrics.roc_auc_boot_lo, metrics.roc_auc_boot_hi - metrics.roc_auc],
                fmt="s", color="#eb6834", markersize=4, capsize=2, linewidth=1,
                label="Test (95% bootstrap CI)")
    ax.set_xticks(x, [DISPLAY_NAMES[n] for n in names], fontsize=7)
    ax.set_ylabel("ROC-AUC")
    lo = min((cv["mean_roc_auc"] - cv["std_roc_auc"]).min(), metrics.roc_auc_boot_lo.min())
    ax.set_ylim(max(0.0, lo - 0.01), 1.004)
    ax.legend(loc="lower left", fontsize=6.8)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("ROC-AUC: cross-validation vs held-out test")
    save(fig, FIGURES_DIR / "cv_vs_test_auc.png", caps,
         "ROC-AUC of each tuned model in 5-fold cross-validation on the training set (mean +/- standard "
         "deviation over folds) and on the held-out test set (point estimate with 95% stratified bootstrap "
         "interval).")


def _plot_cv_roc(caps):
    path = TABLES_DIR / "cv_oof_predictions.csv"
    if not path.exists():
        return
    oof = pd.read_csv(path)
    y = oof["true_label"].values
    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.7, linestyle=(0, (1, 2)))
    for name in MODEL_NAMES:
        p = oof[f"{name}_p_ckd"].values
        fpr, tpr, _ = roc_curve(y, p)
        auc = metrics_from_scores(y, p)["roc_auc"]
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], linestyle=MODEL_STYLES[name], linewidth=1.5,
                label=f"{DISPLAY_NAMES[name]} (pooled AUC {auc:.4f})")
    ax.set_xlim(-0.005, 0.25)
    ax.set_ylim(0.75, 1.005)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.legend(loc="lower right", fontsize=6.8)
    ax.set_title("Out-of-fold ROC curves, training-set CV (zoomed)")
    save(fig, FIGURES_DIR / "cv_roc_curves.png", caps,
         "ROC curves from out-of-fold predictions of the tuned pipelines in 5-fold cross-validation on the "
         "320 training records (axes zoomed to the top-left corner). Every record is scored by a model that "
         "did not see it during fitting; the pooled AUC can differ slightly from the mean of per-fold AUCs.")


def _plot_test_scores(y, probs, thr, caps):
    rng = np.random.default_rng(0)
    names = list(probs)
    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    for row, name in enumerate(names):
        p = probs[name]
        for k, marker in ((0, "o"), (1, "^")):
            sel = y == k
            jitter = rng.uniform(-0.18, 0.18, sel.sum())
            ax.scatter(p[sel], row + jitter + (0.2 if k else -0.2), s=10, marker=marker,
                       color=CLASS_COLORS[k], alpha=0.75, linewidths=0,
                       label=("CKD (true)" if k else "Non-CKD (true)") if row == 0 else None)
    ax.axvline(thr, color=INK, linewidth=0.7)
    ax.text(thr + 0.01, len(names) - 0.45, f"threshold {thr}", fontsize=6.3, color=INK)
    ax.set_yticks(range(len(names)), [DISPLAY_NAMES[n] for n in names], fontsize=7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, len(names) - 0.25)
    ax.invert_yaxis()
    ax.set_xlabel("Predicted P(CKD), model estimate")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.8)
    save(fig, FIGURES_DIR / "test_score_distributions.png", caps,
         "Predicted CKD probabilities of the four frozen pipelines for the 80 held-out test records, "
         "separated by true class (circles: non-CKD, triangles: CKD). Records near the 0.5 threshold are "
         "the least certain predictions. Probabilities are model estimates after SMOTENC balancing, not "
         "calibrated individual risks.")


# --------------------------------------------------------------------------- leakage audit
def leakage_audit(X_test: pd.DataFrame, models: dict, cfg: dict) -> pd.DataFrame:
    from .data import get_partition, load_clean, row_keys
    from .config import SPLIT_PATH
    import hashlib

    df = load_clean()
    split = pd.read_csv(SPLIT_PATH)
    X_train, _ = get_partition(df, split, "train")
    selection = read_json(SELECTION_PATH)
    manifest = read_json(SPLIT_MANIFEST_PATH)
    checks = []

    def add(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    overlap = set(X_train.index) & set(X_test.index)
    add("Train and test record IDs are disjoint", not overlap, f"{len(overlap)} shared IDs")
    tr_keys = set(row_keys(X_train))
    te_keys = row_keys(X_test)
    shared = int(te_keys.isin(tr_keys).sum())
    add("No identical feature vector appears in both partitions", shared == 0, f"{shared} test records")
    test_ids = sorted(X_test.index.tolist())
    digest = hashlib.sha256(",".join(map(str, test_ids)).encode()).hexdigest()
    add("Test partition matches the split locked before tuning", digest == manifest["test_ids_sha256"],
        f"sha256 {digest[:12]}...")
    add("Model chosen from training-set CV only (test unused)", selection.get("test_set_used") is False,
        f"selection recorded at {selection['decided_at_utc']}")

    sel = joblib.load(SELECTED_MODEL_PATH)
    prep = sel.named_steps["prep"]
    med_fit = prep.named_transformers_["continuous"].named_steps["impute"].statistics_
    med_train = X_train[schema.CONTINUOUS].median().values
    med_all = df[schema.CONTINUOUS].median().values
    add("Imputation medians equal training-only medians", np.allclose(med_fit, med_train),
        f"{int((~np.isclose(med_train, med_all)).sum())} of {len(med_all)} medians differ from all-data medians")
    sc = prep.named_transformers_["continuous"].named_steps["scale"]
    Xtr_imp = X_train[schema.CONTINUOUS].fillna(pd.Series(med_train, index=schema.CONTINUOUS))
    add("Scaling statistics equal training-only statistics", np.allclose(sc.mean_, Xtr_imp.mean().values),
        "StandardScaler.mean_ compared with imputed training means")
    add("Predictors exclude ID and target",
        list(prep.feature_names_in_) == schema.RAW_FEATURE_ORDER and schema.ID_COLUMN not in prep.feature_names_in_
        and schema.TARGET not in prep.feature_names_in_, f"{len(prep.feature_names_in_)} input columns")

    # imbalanced-learn skips samplers at predict time: pipeline output must equal prep -> clf directly.
    p_pipe = ckd_probability(sel, X_test)
    p_direct = sel.named_steps["clf"].predict_proba(transform_features(sel, X_test).values)[:, 1]
    add("Resampling is not applied at prediction time", np.allclose(p_pipe, p_direct) and len(p_pipe) == len(X_test),
        f"{len(p_pipe)} predictions for {len(X_test)} test records")
    has_sampler = type(sel.named_steps["resample"]).__name__
    add("Resampler sits inside the fitted pipeline (fit on training folds only)", has_sampler == "SMOTENC",
        f"resample step = {has_sampler}")

    perm = pd.read_csv(TABLES_DIR / "cv_permutation_check.csv")
    add("Shuffled-label training falls to chance level", abs(perm.mean_roc_auc.mean() - 0.5) < 0.1,
        f"mean CV ROC-AUC with permuted labels = {perm.mean_roc_auc.mean():.3f}")

    t0 = time.perf_counter()
    reloaded = joblib.load(SELECTED_MODEL_PATH)
    same = np.allclose(ckd_probability(reloaded, X_test), p_pipe)
    add("Reloaded pipeline reproduces predictions", same, f"reload took {time.perf_counter() - t0:.2f}s")
    audit = pd.DataFrame(checks)
    audit.to_csv(TABLES_DIR / "leakage_audit.csv", index=False)
    write_json({"generated_at_utc": now_utc(), "checks": checks}, TABLES_DIR.parent / "logs" / "leakage_audit.json")
    return audit
