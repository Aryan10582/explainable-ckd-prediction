"""System workflow diagram (single IEEE column), drawn from the saved run metadata."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import FIGURES_DIR, SELECTION_PATH, SPLIT_MANIFEST_PATH
from .eda import CAPTIONS_PATH, _captions
from .plotting import COL_W, INK, MUTED, apply_style, save
from .utils import read_json, write_json

STYLES = {
    "data": ("#f3f2ef", "#8c8b86"),
    "train": ("#e7f0fb", "#2a78d6"),
    "test": ("#fdece4", "#eb6834"),
    "explain": ("#e6f6ef", "#1baf7a"),
}
ARROW = "→"


def _box(ax, x, y, w, h, title, body, kind):
    face, edge = STYLES[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor=face, edgecolor=edge, linewidth=0.9))
    ax.text(x + w / 2, y + h - 0.012, title, ha="center", va="top", fontsize=6.6, fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.034, body, ha="center", va="top", fontsize=5.6, color=INK,
                linespacing=1.25)


def _arrow(ax, p, q, color=MUTED):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, linewidth=0.8, color=color,
                                 shrinkA=0, shrinkB=0))


def make_workflow_diagram() -> None:
    apply_style()
    sel = read_json(SELECTION_PATH)
    man = read_json(SPLIT_MANIFEST_PATH)
    tr, te = man["class_counts"]["train"], man["class_counts"]["test"]
    a = ARROW
    fig = plt.figure(figsize=(COL_W, 5.6))
    ax = fig.add_axes((0.005, 0.005, 0.99, 0.99))   # the diagram uses the full canvas
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    L, W = 0.02, 0.96              # full-width boxes
    lw, rx, rw = 0.58, 0.64, 0.34  # left (training) column and right (test) column
    lc, rc = L + lw / 2, rx + rw / 2

    _box(ax, L, 0.915, W, 0.075, "1. UCI Chronic Kidney Disease dataset",
         "400 records, 24 features (official archive, SHA-256 recorded)", "data")
    _box(ax, L, 0.805, W, 0.085, "2. Cleaning and schema validation",
         "whitespace and '?' fixed, 3 malformed lines repaired,\n"
         "3 impossible values set to missing; no record removed", "data")
    _box(ax, L, 0.672, lw, 0.088, f"3a. Training set ({man['n_train']})",
         f"{tr['ckd']} CKD / {tr['notckd']} non-CKD", "train")
    _box(ax, rx, 0.672, rw, 0.088, f"3b. Test set ({man['n_test']})",
         f"{te['ckd']} CKD / {te['notckd']} non-CKD\nlocked before tuning", "test")
    _box(ax, L, 0.470, lw, 0.177, "4. 5-fold CV + grid search",
         f"stratified folds; inside each training fold:\nmedian / mode imputation\n{a} standard scaling\n"
         f"{a} SMOTE-NC (fit only)\n{a} LR | SVM | RF | XGBoost", "train")
    _box(ax, L, 0.350, lw, 0.095, "5. Predeclared selection",
         f"mean CV ROC-AUC (tie within 0.005)\n{a} recall {a} F1  ⇒  {sel['display_name']}", "train")
    _box(ax, L, 0.240, lw, 0.085, "6. Refit and freeze",
         "best pipeline refitted on all\ntraining records and saved", "train")
    _box(ax, rx, 0.240, rw, 0.205, "7. Final test\nevaluation",
         "\n\nmetrics, exact and\nbootstrap 95% CIs,\nconfusion matrices,\nleakage audit", "test")
    _box(ax, L, 0.120, W, 0.095, "8. SHAP explanations",
         "interventional TreeExplainer, background of 100 real\ntraining records; global and local explanations",
         "explain")
    _box(ax, L, 0.010, W, 0.085, "9. Streamlit demonstration",
         f"validate {a} impute {a} predict {a} explain, using saved artifacts only", "explain")

    _arrow(ax, (0.5, 0.915), (0.5, 0.890))
    _arrow(ax, (lc, 0.805), (lc, 0.760))
    _arrow(ax, (rc, 0.805), (rc, 0.760))
    ax.text(0.5, 0.783, "stratified 80/20 split, seed 42", ha="center", va="center", fontsize=5.5, color=MUTED,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    _arrow(ax, (lc, 0.672), (lc, 0.647))
    _arrow(ax, (lc, 0.470), (lc, 0.445))
    _arrow(ax, (lc, 0.350), (lc, 0.325))
    _arrow(ax, (rc, 0.672), (rc, 0.445), color="#eb6834")
    _arrow(ax, (L + lw, 0.2825), (rx, 0.2825))
    _arrow(ax, (lc, 0.240), (lc, 0.215))
    _arrow(ax, (0.5, 0.120), (0.5, 0.095))
    caps = _captions()
    save(fig, FIGURES_DIR / "workflow_diagram.png", caps, layout=False, caption=
         "Workflow of the proposed system. Blue steps use training records only; the test partition (orange) "
         "is locked after the split and used once, for the final evaluation of the frozen pipelines. All "
         "preprocessing and SMOTE-NC resampling are refitted inside each cross-validation training fold.")
    write_json(caps, CAPTIONS_PATH)
