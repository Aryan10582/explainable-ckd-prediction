"""Exploratory data analysis.

Two kinds of output, kept clearly apart:
* whole-dataset descriptive summaries (no class information): class counts,
  missingness, feature distributions;
* class-conditioned exploration that could influence modelling choices
  (class differences, missingness by class, correlations): training records only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import schema
from .config import FIGURES_DIR, TABLES_DIR
from .plotting import CLASS_COLORS, COL_W, FULL_W, GRID, INK, MUTED, apply_style, save
from .utils import read_json, write_json

CAPTIONS_PATH = FIGURES_DIR / "figure_captions.json"
CLASS_LABEL = {1: "CKD", 0: "Non-CKD"}


def _captions() -> dict:
    return read_json(CAPTIONS_PATH) if CAPTIONS_PATH.exists() else {}


def _label(c: str) -> str:
    return schema.FEATURE_BY_NAME[c].display


def class_distribution(df, caps):
    counts = df["target"].value_counts().reindex([1, 0])
    fig, ax = plt.subplots(figsize=(COL_W, 1.55))
    bars = ax.barh([CLASS_LABEL[k] for k in counts.index], counts.values,
                   color=[CLASS_COLORS[k] for k in counts.index], height=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(v + 4, b.get_y() + b.get_height() / 2, f"{v}  ({100 * v / counts.sum():.1f}%)",
                va="center", fontsize=7.5, color=INK)
    ax.set_xlim(0, counts.max() * 1.35)
    ax.set_xlabel("Number of records")
    ax.invert_yaxis()
    ax.set_title(f"Class distribution, all {len(df)} records")
    save(fig, FIGURES_DIR / "eda_class_distribution.png", caps,
         f"Class distribution of the complete UCI CKD dataset (n = {len(df)}; whole-dataset descriptive "
         "summary). CKD is the majority and the positive class.")


def missingness(df, train_df, caps):
    miss = 100 * df[schema.RAW_FEATURE_ORDER].isna().mean()
    miss = miss.sort_values()
    fig, ax = plt.subplots(figsize=(COL_W, 3.6))
    ax.barh([_label(c) for c in miss.index], miss.values, color="#2a78d6", height=0.7)
    for i, v in enumerate(miss.values):
        if v > 0:
            ax.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=6.5, color=MUTED)
    ax.set_xlabel("Missing values (% of 400 records)")
    ax.set_xlim(0, miss.max() * 1.15)
    ax.tick_params(axis="y", labelsize=6.8)
    ax.set_title("Missing values by feature, all records")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    top = miss.sort_values(ascending=False).head(3)
    top_txt = ", ".join(f"{schema.FEATURE_BY_NAME[c].label.lower()} ({v:.1f}%)" for c, v in top.items())
    save(fig, FIGURES_DIR / "eda_missingness.png", caps,
         "Percentage of missing values per feature in the complete dataset (whole-dataset descriptive "
         f"summary). The most frequently missing features are {top_txt}.")

    # Training records only: number of missing values per record, by class.
    n_miss = train_df[schema.RAW_FEATURE_ORDER].isna().sum(axis=1)
    fig, ax = plt.subplots(figsize=(COL_W, 1.65))
    bins = np.arange(0, n_miss.max() + 2) - 0.5
    for k in (0, 1):
        vals = n_miss[train_df["target"] == k]
        ax.hist(vals, bins=bins, color=CLASS_COLORS[k], alpha=0.75 if k else 0.85,
                label=f"{CLASS_LABEL[k]} (n={len(vals)}, median {np.median(vals):.0f})",
                rwidth=0.85, density=True)
    ax.set_xlabel("Missing values per record")
    ax.set_ylabel("Share of class")
    ax.legend(loc="upper right")
    ax.set_title("Missing values per record by class, training set")
    save(fig, FIGURES_DIR / "eda_missing_per_record_train.png", caps,
         "Number of missing values per record in the training set, by class. Non-CKD records are "
         "mostly complete while CKD records often have several missing values, so missingness itself "
         "carries class information (training records only).")
    tab = pd.DataFrame({
        "feature": schema.RAW_FEATURE_ORDER,
        "missing_pct_train_ckd": [100 * train_df.loc[train_df.target == 1, c].isna().mean()
                                  for c in schema.RAW_FEATURE_ORDER],
        "missing_pct_train_nonckd": [100 * train_df.loc[train_df.target == 0, c].isna().mean()
                                     for c in schema.RAW_FEATURE_ORDER],
    }).round(1)
    tab.to_csv(TABLES_DIR / "missingness_by_class_train.csv", index=False)


def numeric_distributions(df, caps):
    cols = schema.CONTINUOUS
    fig, axes = plt.subplots(3, 4, figsize=(FULL_W, 4.4))
    for ax, c in zip(axes.flat, cols):
        vals = df[c].dropna()
        ax.hist(vals, bins=25, color="#2a78d6", rwidth=0.9)
        ax.set_title(f"{_label(c)}\n(n={len(vals)})", fontsize=7)
        ax.tick_params(labelsize=6.5)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    fig.supylabel("Number of records", fontsize=7.5)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, FIGURES_DIR / "eda_numeric_distributions.png", caps,
         "Distributions of the eleven continuous features over all records with a recorded value "
         "(whole-dataset descriptive summary; no class information).")


def categorical_distributions(df, caps):
    cols = schema.ORDINAL + schema.BINARY
    fig, axes = plt.subplots(4, 4, figsize=(FULL_W, 5.0))
    for ax, c in zip(axes.flat, cols):
        f = schema.FEATURE_BY_NAME[c]
        levels = list(f.levels)
        counts = [int((df[c] == lv).sum()) if f.kind == "binary" else int(np.isclose(df[c], lv).sum())
                  for lv in levels]
        names = [str(lv) if f.kind == "binary" else f"{lv:g}" for lv in levels] + ["missing"]
        counts.append(int(df[c].isna().sum()))
        ax.bar(names, counts, color=["#2a78d6"] * len(levels) + ["#b9b8b3"], width=0.7)
        ax.set_title(f.label, fontsize=7)
        ax.tick_params(axis="x", labelsize=6, rotation=30)
        ax.tick_params(axis="y", labelsize=6)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    fig.supylabel("Number of records", fontsize=7.5)
    fig.tight_layout(h_pad=0.6, w_pad=0.5)
    save(fig, FIGURES_DIR / "eda_categorical_distributions.png", caps,
         "Counts of each recorded level of the ordinal and binary features, with missing values shown "
         "in grey (whole-dataset descriptive summary; no class information).")


def class_differences(train_df, caps):
    cols = schema.CONTINUOUS
    fig, axes = plt.subplots(3, 4, figsize=(FULL_W, 4.6))
    for ax, c in zip(axes.flat, cols):
        data = [train_df.loc[train_df.target == k, c].dropna() for k in (0, 1)]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", markersize=2, alpha=0.5, markeredgewidth=0),
                        medianprops=dict(color=INK, linewidth=1.0))
        for patch, k in zip(bp["boxes"], (0, 1)):
            patch.set_facecolor(CLASS_COLORS[k])
            patch.set_alpha(0.8)
            patch.set_edgecolor(MUTED)
        for fl, k in zip(bp["fliers"], (0, 1)):
            fl.set_markerfacecolor(CLASS_COLORS[k])
        ax.set_xticks([1, 2], ["Non-CKD", "CKD"], fontsize=6.5)
        ax.set_title(_label(c), fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
    axes.flat[-1].axis("off")
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, FIGURES_DIR / "eda_class_differences_train.png", caps,
         "Continuous features by class in the training set (n = 320; observed values only). Boxes show "
         "the median and inter-quartile range. Differences describe association in this sample, not "
         "causation.")

    rows = []
    for c in schema.RAW_FEATURE_ORDER:
        f = schema.FEATURE_BY_NAME[c]
        row = {"feature": c, "name": f.display, "type": f.kind}
        for k in (1, 0):
            s = train_df.loc[train_df.target == k, c].dropna()
            tag = CLASS_LABEL[k]
            if f.kind == "binary":
                row[tag] = f"{100 * (s == f.levels[1]).mean():.1f}% {f.levels[1]} (n={len(s)})"
            else:
                q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
                row[tag] = f"{med:g} [{q1:g}-{q3:g}] (n={len(s)})"
        rows.append(row)
    pd.DataFrame(rows).to_csv(TABLES_DIR / "class_comparison_train.csv", index=False)


def correlations(train_df, caps):
    cols = schema.CONTINUOUS + schema.ORDINAL
    corr = train_df[cols].corr(method="spearman", min_periods=30)
    corr.to_csv(TABLES_DIR / "spearman_correlation_train.csv")
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    fig, ax = plt.subplots(figsize=(COL_W, 3.3))
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    labels = [schema.FEATURE_BY_NAME[c].label for c in cols]
    ax.set_xticks(range(len(cols)), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(cols)), labels, fontsize=6)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            if i != j and abs(v) >= 0.5:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=4.8,
                        color="white" if abs(v) > 0.75 else INK)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Spearman correlation", fontsize=6.5)
    ax.set_title("Feature correlations, training set")
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    strongest = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(2)
    pair_txt = "; ".join(f"{schema.FEATURE_BY_NAME[a].label.lower()} and "
                         f"{schema.FEATURE_BY_NAME[b].label.lower()} (r = {v:.2f})"
                         for (a, b), v in strongest.items())
    save(fig, FIGURES_DIR / "eda_correlation_train.png", caps,
         "Spearman rank correlations between numeric and ordinal features in the training set "
         "(pairwise complete observations); values with |r| >= 0.5 are printed. The strongest pairs are "
         f"{pair_txt}.")


def feature_dictionary():
    pd.DataFrame(schema.feature_dictionary_rows()).to_csv(TABLES_DIR / "feature_dictionary.csv", index=False)


def summary_tables(df):
    desc = df[schema.CONTINUOUS + schema.ORDINAL].describe().T
    desc.insert(0, "name", [_label(c) for c in desc.index])
    desc.round(3).to_csv(TABLES_DIR / "descriptive_numeric_all_records.csv")


def run_eda(df: pd.DataFrame, train_df: pd.DataFrame) -> dict:
    apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    caps = _captions()
    feature_dictionary()
    summary_tables(df)
    class_distribution(df, caps)
    missingness(df, train_df, caps)
    numeric_distributions(df, caps)
    categorical_distributions(df, caps)
    class_differences(train_df, caps)
    correlations(train_df, caps)
    write_json(caps, CAPTIONS_PATH)
    return caps
