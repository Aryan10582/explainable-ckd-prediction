"""SHAP explanations for the selected pipeline.

What is explained
-----------------
The classifier step of the fitted pipeline, applied to the *preprocessed*
features exactly as at prediction time (imputation + scaling + encoding; the
resampler is not used at prediction time). Each model column corresponds to
exactly one original feature, so attributions are reported per original
feature with no aggregation. Feature values in plots are mapped back to
clinical units (after imputation) and imputed values are flagged.

Output scale per model (recorded in the metadata):
* Logistic Regression, XGBoost -> log-odds of CKD (additive in the margin)
* Random Forest, SVM           -> probability of CKD

The background (reference) distribution is a fixed random sample of real
*training* records -- never test records and never SMOTENC synthetic records.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import shap

from . import schema
from .config import FIGURES_DIR, MODELS_DIR, SELECTED_META_PATH, SELECTED_MODEL_PATH, TABLES_DIR
from .eda import CAPTIONS_PATH, _captions
from .models import DISPLAY_NAMES, SHAP_OUTPUT
from .plotting import COL_W, INK, MUTED, apply_style, fit_to_width, save
from .preprocessing import imputed_mask, to_original_units, transform_features
from .utils import ckd_probability, read_json, write_json

BACKGROUND_PATH = MODELS_DIR / "shap_background.joblib"
POS_COLOR, NEG_COLOR = "#e34948", "#2a78d6"   # pushes toward CKD / toward non-CKD


@dataclass
class ShapBundle:
    model_name: str
    kind: str              # "linear" | "tree" | "kernel"
    output: str            # e.g. "log-odds of CKD"
    explainer: object
    base_value: float
    background_size: int
    note: str = ""


def _float32(a: np.ndarray) -> np.ndarray:
    """Round to single precision (kept as float64): tree models compare features in float32."""
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def forest_copy_for_shap(clf):
    """Copy of a scikit-learn forest whose split thresholds SHAP routes identically.

    scikit-learn casts inputs to float32 and sends a record left when
    float32(x) <= threshold (a float64). SHAP's interventional Tree SHAP does
    not reproduce that rule exactly, so a background record lying within
    float32 rounding of a threshold can be routed down the wrong branch.
    SMOTENC makes this likely, because interpolation creates synthetic values
    almost identical to real ones and split thresholds fall between them.
    Rounding each threshold *down* to the nearest float32 value leaves every
    scikit-learn decision unchanged (for a float32 x, x <= t  <=>  x <= floor32(t)),
    and with float32 inputs SHAP then follows exactly the same paths.
    """
    import copy

    patched = copy.deepcopy(clf)
    n_changed = n_total = 0
    for est in patched.estimators_:
        thr = est.tree_.threshold            # writable view into the fitted tree
        t32 = thr.astype(np.float32)
        up = t32.astype(np.float64) > thr
        t32[up] = np.nextafter(t32[up], np.float32(-np.inf))
        internal = est.tree_.feature >= 0
        n_changed += int((t32.astype(np.float64) != thr)[internal].sum())
        n_total += int(internal.sum())
        thr[:] = t32.astype(np.float64)
    return patched, n_changed, n_total


def model_output(model_name: str, pipeline, Xt: pd.DataFrame) -> np.ndarray:
    """The exact quantity whose SHAP decomposition is computed."""
    clf = pipeline.named_steps["clf"]
    if list(clf.classes_) != [0, 1]:
        raise ValueError("Positive class (CKD) must be label 1.")
    if model_name == "logistic_regression":
        return clf.decision_function(Xt.values)
    if model_name == "xgboost":
        return clf.predict(Xt.values, output_margin=True)
    return clf.predict_proba(Xt.values)[:, 1]


def output_to_probability(model_name: str, values: np.ndarray) -> np.ndarray:
    if "log-odds" in SHAP_OUTPUT[model_name]:
        return 1.0 / (1.0 + np.exp(-np.asarray(values)))
    return np.asarray(values)


def build_explainer(pipeline, model_name: str, X_background: pd.DataFrame, cfg: dict) -> ShapBundle:
    clf = pipeline.named_steps["clf"]
    Xb = transform_features(pipeline, X_background)
    if model_name == "logistic_regression":
        expl = shap.LinearExplainer(clf, shap.maskers.Independent(Xb.values, max_samples=len(Xb)))
        kind, base = "linear", float(np.ravel(expl.expected_value)[0])
    elif model_name in ("xgboost", "random_forest"):
        note, target = "", clf
        if model_name == "random_forest":
            target, n_changed, n_total = forest_copy_for_shap(clf)
            check = _float32(Xb.values)
            if not np.array_equal(target.predict_proba(check), clf.predict_proba(check)):
                raise AssertionError("Threshold-rounded forest copy changed predictions.")
            note = (f"{n_changed} of {n_total} split thresholds rounded down to float32 for SHAP; "
                    "predictions unchanged")
        expl = shap.TreeExplainer(target, data=_float32(Xb.values), feature_perturbation="interventional",
                                  model_output="raw")
        ev = np.ravel(expl.expected_value)
        kind, base = "tree", float(ev[1] if len(ev) == 2 else ev[0])
        return ShapBundle(model_name, kind, SHAP_OUTPUT[model_name], expl, base, len(X_background), note)
    elif model_name == "svm":
        n = min(cfg["shap"]["kernel_background_size"], len(Xb))
        bg = Xb.sample(n=n, random_state=cfg["project"]["random_seed"]).values
        expl = shap.KernelExplainer(lambda Z: clf.predict_proba(Z)[:, 1], bg)
        kind, base = "kernel", float(np.ravel(expl.expected_value)[0])
        X_background = X_background.loc[Xb.sample(n=n, random_state=cfg["project"]["random_seed"]).index]
    else:
        raise ValueError(model_name)
    return ShapBundle(model_name, kind, SHAP_OUTPUT[model_name], expl, base, len(X_background))


def shap_values(bundle: ShapBundle, pipeline, X_raw: pd.DataFrame, cfg: dict | None = None) -> np.ndarray:
    """SHAP values (n_records x 24) in MODEL_FEATURES order for the CKD output."""
    Xt = transform_features(pipeline, X_raw)
    if bundle.kind == "kernel":
        nsamples = (cfg or {}).get("shap", {}).get("kernel_nsamples", 1000)
        vals = bundle.explainer.shap_values(Xt.values, nsamples=nsamples, silent=True)
        vals = np.asarray(vals)
    elif bundle.kind == "tree":
        vals = np.asarray(bundle.explainer.shap_values(_float32(Xt.values)))
        if vals.ndim == 3:                     # (n, features, classes) -> CKD column
            vals = vals[:, :, 1]
    else:
        vals = np.asarray(bundle.explainer.shap_values(Xt.values))
        if vals.ndim == 3:
            vals = vals[:, :, 1]
    if vals.shape != (len(X_raw), len(schema.MODEL_FEATURES)):
        raise ValueError(f"Unexpected SHAP shape {vals.shape}")
    return vals


def additivity_tolerance(bundle: ShapBundle) -> float:
    """Allowed |baseline + sum(SHAP) - model output|.

    Kernel SHAP solves a regression (enforces the sum up to solver precision);
    XGBoost computes margins in float32, so sums of ~100 trees carry ~1e-6
    rounding; the scikit-learn trees and the linear model are exact to ~1e-8.
    """
    if bundle.kind == "kernel":
        return 1e-3
    if bundle.model_name == "xgboost":
        return 1e-4
    return 1e-6


def additivity_error(bundle: ShapBundle, pipeline, X_raw: pd.DataFrame, vals: np.ndarray) -> float:
    out = model_output(bundle.model_name, pipeline, transform_features(pipeline, X_raw))
    return float(np.max(np.abs(bundle.base_value + vals.sum(axis=1) - out)))


def display_values(pipeline, X_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(numeric original-unit values, human-readable strings, imputed mask) per model feature."""
    Xt = transform_features(pipeline, X_raw)
    num = to_original_units(pipeline, Xt)
    mask = imputed_mask(X_raw)
    txt = pd.DataFrame(index=num.index, columns=schema.MODEL_FEATURES, dtype=object)
    for c in schema.MODEL_FEATURES:
        f = schema.FEATURE_BY_NAME[c]
        if f.kind == "binary":
            s = num[c].map(lambda v: f.levels[int(round(v))])
        elif f.kind == "ordinal":
            s = num[c].map(lambda v: f"{v:g}")
        else:
            s = num[c].map(lambda v: f"{v:.3g}" + (f" {f.unit}" if f.unit else ""))
        txt[c] = s.where(~mask[c], s + " (imputed)")
    return num, txt, mask


def feature_labels(kind: str = "plain") -> list[str]:
    out = []
    for c in schema.MODEL_FEATURES:
        f = schema.FEATURE_BY_NAME[c]
        if kind == "coded" and f.kind == "binary":
            out.append(f"{f.label} ({f.levels[1]}=1)")
        elif kind == "coded" and f.unit:
            out.append(f"{f.label} ({f.unit})")
        else:
            out.append(f.label)
    return out


# --------------------------------------------------------------------------- figures
def waterfall(values: np.ndarray, base: float, txt_row: pd.Series, output: str, title: str,
              max_display: int = 9, width: float = COL_W):
    """Compact SHAP waterfall: baseline at the bottom, model output f(x) at the top.

    Same convention as shap.plots.waterfall (red = pushes towards CKD, blue =
    pushes away), redrawn so labels stay legible at single-column width.
    """
    order = np.argsort(-np.abs(values))
    top, rest = order[:max_display], order[max_display:]

    def short_value(text: str) -> str:            # long count units would crowd a column-width figure
        for unit in (" millions/cmm", " cells/cumm"):
            text = text.replace(unit, "")
        return text

    labels = [f"{schema.FEATURE_BY_NAME[schema.MODEL_FEATURES[i]].label} = {short_value(txt_row.iloc[i])}"
              for i in top]
    contribs = list(values[top])
    if len(rest):
        labels.append(f"{len(rest)} other features")
        contribs.append(float(values[rest].sum()))
    labels, contribs = labels[::-1], contribs[::-1]          # draw bottom -> top
    fx = base + values.sum()
    fig, ax = plt.subplots(figsize=(width, 0.85 + 0.235 * len(labels)))
    # x-range covering every bar end, then room on both sides for the value labels
    ends, cum = [base], base
    for v in contribs:
        cum += v
        ends.append(cum)
    lo, hi = min(ends), max(ends)
    rng_ = max(hi - lo, 1e-9)
    ax.set_xlim(lo - 0.32 * rng_, hi + 0.30 * rng_)
    cum = base
    for y, v in enumerate(contribs):
        ax.barh(y, v, left=cum, color=POS_COLOR if v >= 0 else NEG_COLOR, height=0.62)
        end = cum + v
        ax.text(end + (0.015 if v >= 0 else -0.015) * rng_, y,
                f"{v:+.2f}" if abs(v) >= 0.005 else f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=6.2, color=INK)
        if y < len(labels) - 1:
            ax.plot([end, end], [y + 0.31, y + 0.69], color=MUTED, linewidth=0.5)
        cum = end
    ax.set_yticks(range(len(labels)), labels, fontsize=6.4)
    ax.axvline(base, color=MUTED, linewidth=0.7, linestyle=(0, (2, 2)))
    ax.axvline(fx, color=INK, linewidth=0.7)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("Model output: " + ("P(CKD)" if output.startswith("probability") else "log-odds of CKD"),
                  fontsize=6.6)
    ax.tick_params(axis="x", labelsize=6.3)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # Explicit layout: the figure keeps exactly `width`; the left margin fits the longest row label.
    fig.suptitle(title, fontsize=7.0, y=0.995, va="top")
    fig.text(0.5, 0.008, f"baseline E[f(X)] = {base:.3f} (dashed)    f(x) = {fx:.3f} (solid)",
             ha="center", va="bottom", fontsize=6.2, color=INK)
    fig.set_dpi(plt.rcParams["savefig.dpi"])
    fig.canvas.draw()
    w_in, h_in = fig.get_size_inches()
    label_w = max(t.get_window_extent().width for t in ax.get_yticklabels()) / fig.dpi
    fig.subplots_adjust(left=(label_w + 0.08) / w_in, right=0.985, top=1 - 0.26 / h_in, bottom=0.52 / h_in)
    # Widen the x-range until every value label sits inside the axes (measured, not guessed).
    for _ in range(4):                       # widening changes the scale, so repeat until stable
        fig.canvas.draw()
        inv = ax.transData.inverted()
        xs = [inv.transform(t.get_window_extent())[:, 0] for t in ax.texts]
        x0, x1 = ax.get_xlim()
        margin = 0.02 * (x1 - x0)
        new = (min(x0, min(x[0] for x in xs) - margin), max(x1, max(x[1] for x in xs) + margin))
        if np.allclose(new, (x0, x1)):
            break
        ax.set_xlim(*new)
    return fig


def _beeswarm(vals, num, base, output, max_display, caps, n):
    expl = shap.Explanation(values=vals, base_values=np.full(len(vals), base), data=num.values,
                            feature_names=feature_labels("coded"))
    shap.plots.beeswarm(expl, max_display=max_display, show=False, plot_size=(COL_W, 3.45), s=7,
                        color_bar_label="Feature value (low to high)")
    fig = plt.gcf()
    fig.set_size_inches(COL_W, 3.45)
    for t in fig.findobj(mtext.Text):
        t.set_fontsize(min(t.get_fontsize(), 6.6))
    ax = fig.axes[0]
    ax.set_xlabel(f"SHAP value (effect on {output})", fontsize=6.6)
    ax.set_title(f"SHAP summary, test set (n = {n})", fontsize=7.2)
    save(fig, FIGURES_DIR / "shap_beeswarm.png", caps, fit_width=COL_W, caption=
         f"SHAP summary (beeswarm) plot for the selected model on the {n} held-out test records. Each dot "
         f"is one record; its horizontal position is that feature's contribution to the {output}, and its "
         "colour is the feature value (red = high, blue = low; for binary features red = the condition "
         "shown in brackets). Positive values push the prediction towards CKD.")


def _global_bar(imp: pd.DataFrame, output: str, max_display: int, caps, n):
    top = imp.head(max_display).iloc[::-1]
    fig, ax = plt.subplots(figsize=(COL_W, 0.55 + 0.19 * len(top)))
    ax.barh(top["label"], top["mean_abs_shap_test"], color="#2a78d6", height=0.68)
    for i, v in enumerate(top["mean_abs_shap_test"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=6.0, color=MUTED)
    ax.set_xlabel(f"Mean |SHAP value| ({output})")
    ax.set_xlim(0, top["mean_abs_shap_test"].max() * 1.18)
    ax.tick_params(axis="y", labelsize=6.6)
    ax.set_title(f"Global feature importance, test set (n={n})")
    save(fig, FIGURES_DIR / "shap_global_importance.png", caps,
         f"Global SHAP importance of the selected model: mean absolute SHAP value of each feature over the "
         f"{n} held-out test records, in units of {output}. Only the top {len(top)} features are shown.")


def _dependence(vals, num, mask, imp, output, caps, k=4):
    feats = imp["feature"].head(k).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.1))
    for ax, c in zip(axes.flat, feats):
        j = schema.MODEL_FEATURES.index(c)
        f = schema.FEATURE_BY_NAME[c]
        obs = ~mask[c].values
        ax.axhline(0, color=MUTED, linewidth=0.5)
        ax.scatter(num[c].values[obs], vals[obs, j], s=9, color="#2a78d6", alpha=0.8, linewidths=0,
                   label="observed")
        if (~obs).any():
            ax.scatter(num[c].values[~obs], vals[~obs, j], s=12, marker="x", color="#eb6834",
                       linewidths=0.8, label="imputed")
        if f.kind == "binary":
            ax.set_xticks([0, 1], list(f.levels), fontsize=6)
        xl = f.display if f.kind != "binary" else f.label
        ax.set_xlabel(xl, fontsize=6.6, labelpad=1)
        ax.set_ylabel("SHAP value", fontsize=6.6, labelpad=1)
        ax.tick_params(labelsize=6)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    for ax in axes.flat[1:]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.9, w_pad=0.9)
    save(fig, FIGURES_DIR / "shap_dependence.png", caps,
         f"SHAP dependence plots for the four most important features (test set). The vertical axis is the "
         f"contribution to the {output}; the horizontal axis is the feature value in clinical units after "
         "imputation. Crosses mark values that were missing and filled with the training median or mode.")


# --------------------------------------------------------------------------- local examples
def choose_examples(preds: pd.DataFrame, model: str, threshold: float) -> list[dict]:
    """Predeclared rule for report examples (no cherry-picking):
    * typical CKD / typical non-CKD: the correctly classified test record whose
      CKD probability is closest to the median probability of the correctly
      classified records of that class (lowest record id breaks ties);
    * borderline: the test record whose probability is closest to the threshold;
    * every misclassified test record.
    """
    p, out = f"{model}_p_ckd", f"{model}_outcome"
    ex = []
    for outcome, tag in (("TP", "typical_ckd"), ("TN", "typical_non_ckd")):
        g = preds[preds[out] == outcome]
        if len(g):
            d = (g[p] - g[p].median()).abs()
            pick = g.assign(d=d).sort_values(["d", schema.ID_COLUMN]).iloc[0]
            ex.append({"example": tag, schema.ID_COLUMN: int(pick[schema.ID_COLUMN])})
    d = (preds[p] - threshold).abs()
    pick = preds.assign(d=d).sort_values(["d", schema.ID_COLUMN]).iloc[0]
    ex.append({"example": "borderline", schema.ID_COLUMN: int(pick[schema.ID_COLUMN])})
    for _, r in preds[preds[out].isin(["FP", "FN"])].iterrows():
        ex.append({"example": f"misclassified_{r[out]}", schema.ID_COLUMN: int(r[schema.ID_COLUMN])})
    return ex


def run_explain(X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, cfg: dict, log=print) -> dict:
    apply_style()
    caps = _captions()
    meta = read_json(SELECTED_META_PATH)
    name = meta["model"]
    pipe = joblib.load(SELECTED_MODEL_PATH)
    seed = cfg["project"]["random_seed"]
    thr = cfg["selection"]["threshold"]

    bg = X_train.sample(n=min(cfg["shap"]["background_size"], len(X_train)), random_state=seed)
    t0 = time.perf_counter()
    bundle = build_explainer(pipe, name, bg, cfg)
    joblib.dump(bg, BACKGROUND_PATH)
    pd.DataFrame({schema.ID_COLUMN: bg.index}).to_csv(MODELS_DIR / "shap_background_ids.csv", index=False)

    vals = shap_values(bundle, pipe, X_test, cfg)
    t_test = time.perf_counter() - t0
    add_err = additivity_error(bundle, pipe, X_test, vals)
    unfixed_err = None
    if name == "random_forest":
        # For the record: the same explanation computed on the unmodified forest (see forest_copy_for_shap).
        clf = pipe.named_steps["clf"]
        plain = shap.TreeExplainer(clf, data=transform_features(pipe, bg).values,
                                   feature_perturbation="interventional", model_output="raw")
        Xt_test = transform_features(pipe, X_test).values
        v_plain = np.asarray(plain.shap_values(Xt_test, check_additivity=False))[:, :, 1]
        unfixed_err = float(np.max(np.abs(np.ravel(plain.expected_value)[1] + v_plain.sum(axis=1)
                                          - clf.predict_proba(Xt_test)[:, 1])))
    num, txt, mask = display_values(pipe, X_test)
    out_name = bundle.output

    # Stability of the global ranking: also explain the training records (a sample for Kernel SHAP).
    X_tr_expl = X_train if bundle.kind != "kernel" else X_train.sample(n=min(120, len(X_train)), random_state=seed)
    vals_tr = shap_values(bundle, pipe, X_tr_expl, cfg)
    imp = pd.DataFrame({
        "feature": schema.MODEL_FEATURES,
        "label": feature_labels("plain"),
        "mean_abs_shap_test": np.abs(vals).mean(axis=0),
        "mean_shap_test": vals.mean(axis=0),
        "mean_abs_shap_train": np.abs(vals_tr).mean(axis=0),
    }).sort_values("mean_abs_shap_test", ascending=False).reset_index(drop=True)
    imp["rank_test"] = np.arange(1, len(imp) + 1)
    imp["rank_train"] = imp["mean_abs_shap_train"].rank(ascending=False, method="min").astype(int)
    imp["share_of_total_pct"] = 100 * imp["mean_abs_shap_test"] / imp["mean_abs_shap_test"].sum()
    imp.to_csv(TABLES_DIR / "shap_global_importance.csv", index=False)
    spearman = float(imp[["mean_abs_shap_test", "mean_abs_shap_train"]].corr(method="spearman").iloc[0, 1])

    sv = pd.DataFrame(vals, columns=schema.MODEL_FEATURES, index=X_test.index)
    sv.insert(0, "base_value", bundle.base_value)
    sv["model_output"] = model_output(name, pipe, transform_features(pipe, X_test))
    sv["p_ckd"] = ckd_probability(pipe, X_test)
    sv.to_csv(TABLES_DIR / "shap_values_test.csv")

    md = cfg["shap"]["max_display"]
    _global_bar(imp, out_name, md, caps, len(X_test))
    _beeswarm(vals, num, bundle.base_value, out_name, md, caps, len(X_test))
    _dependence(vals, num, mask, imp, out_name, caps)

    preds = pd.read_csv(TABLES_DIR / "test_predictions.csv")
    examples = choose_examples(preds, name, thr)
    rows = []
    pos = {rid: i for i, rid in enumerate(X_test.index)}
    for ex in examples:
        rid = ex[schema.ID_COLUMN]
        i = pos[rid]
        p = float(sv["p_ckd"].iloc[i])
        true = schema.CLASS_NAMES[int(y_test.loc[rid])]
        pred = schema.CLASS_NAMES[int(p >= thr)]
        title = f"Record {rid} (true {true}): P(CKD) = {p:.3f}"
        fig = waterfall(vals[i], bundle.base_value, txt.iloc[i], out_name, title, max_display=6)
        fname = f"shap_waterfall_{ex['example']}_{rid}.png"
        kind_txt = {"typical_ckd": "a typical correctly classified CKD record",
                    "typical_non_ckd": "a typical correctly classified non-CKD record",
                    "borderline": "the test record whose CKD probability is closest to the 0.5 threshold"}
        desc = kind_txt.get(ex["example"], f"a misclassified test record ({ex['example'][-2:]})")
        save(fig, FIGURES_DIR / fname, caps, layout=False, caption=
             f"SHAP waterfall plot for {desc} (record {rid}). Starting from the baseline (mean "
             f"{out_name} over the training background), red bars push the output towards CKD and blue bars "
             "towards non-CKD; the bars sum exactly to the model output f(x). Values marked 'imputed' were "
             "missing and filled by the pipeline.")
        order = np.argsort(-np.abs(vals[i]))[:5]
        rows.append({"example": ex["example"], schema.ID_COLUMN: rid, "true_class": true,
                     "predicted_class": pred, "p_ckd": p, "model_output": float(sv["model_output"].iloc[i]),
                     "n_imputed": int(mask.iloc[i].sum()), "figure": fname,
                     "top_contributions": "; ".join(
                         f"{schema.FEATURE_BY_NAME[schema.MODEL_FEATURES[j]].label} = {txt.iloc[i, j]}: "
                         f"{vals[i, j]:+.3f}" for j in order)})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "shap_local_examples.csv", index=False)
    write_json(caps, CAPTIONS_PATH)

    checks = {
        "model": name,
        "explainer": type(bundle.explainer).__name__,
        "explainer_mode": "interventional (background of real training records)"
                          if bundle.kind != "kernel" else "Kernel SHAP (background of real training records)",
        "explainer_note": bundle.note,
        "explained_output": out_name,
        "positive_class": "CKD (label 1); classifier classes_ = " + str(list(pipe.named_steps['clf'].classes_)),
        "feature_order": schema.MODEL_FEATURES,
        "background": f"{bundle.background_size} randomly sampled training records (seed {seed})",
        "base_value": bundle.base_value,
        "base_value_as_probability": float(output_to_probability(name, bundle.base_value)),
        "n_explained_test": int(len(X_test)),
        "n_explained_train_for_stability": int(len(X_tr_expl)),
        "shap_shape": list(vals.shape),
        "max_additivity_error": add_err,
        "max_additivity_error_without_threshold_fix": unfixed_err,
        "additivity_tolerance": additivity_tolerance(bundle),
        "additivity_ok": add_err < additivity_tolerance(bundle),
        "spearman_rank_corr_test_vs_train_importance": spearman,
        "seconds_build_and_explain_test": round(t_test, 2),
        "misclassified_test_records": [e[schema.ID_COLUMN] for e in examples if e["example"].startswith("mis")],
    }
    write_json(checks, TABLES_DIR.parent / "logs" / "shap_checks.json")
    if not checks["additivity_ok"]:
        raise AssertionError(f"SHAP additivity check failed: max error {add_err}")
    log(f"  SHAP ({checks['explainer']}, {out_name}): additivity error {add_err:.2e}; "
        f"top features: {', '.join(imp.label.head(5))}")
    return checks
