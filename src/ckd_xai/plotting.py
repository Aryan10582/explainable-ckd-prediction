"""Shared figure style.

Figures are drawn at IEEE single-column width (3.5 in) with 8-pt text so they
stay readable when placed in the two-column paper, and saved at 300 dpi so
they also scale up cleanly on slides.

Colours come from a palette checked for colour-vision-deficiency separation
(all pairs). Model identity is never shown by colour alone: curves also differ
in line style and every figure has a legend.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL_W = 3.25     # column width of the prescribed IEEE template, inches
FULL_W = 7.16    # IEEE text width, inches

MODEL_COLORS = {"logistic_regression": "#2a78d6", "svm": "#eb6834",
                "random_forest": "#1baf7a", "xgboost": "#4a3aa7"}
MODEL_STYLES = {"logistic_regression": "-", "svm": "--", "random_forest": "-.", "xgboost": ":"}
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}   # 0 = Non-CKD (blue), 1 = CKD (orange)
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#e4e4e1"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.linewidth": 0.6, "lines.linewidth": 1.6,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 300,
        "savefig.bbox": None,
    })


def fit_to_width(fig, width: float = COL_W, pad: float = 0.03, iterations: int = 4) -> None:
    """Resize the canvas so the tight bounding box (all labels included) is exactly `width` inches.

    Used for figures whose labels sit outside the axes (SHAP plots): the saved image then has the
    column width without any downscaling, so its fonts keep their designed size in the paper.
    """
    fig.set_dpi(plt.rcParams["savefig.dpi"])   # measure text at the resolution it will be saved at
    for _ in range(iterations):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = fig.get_size_inches()
        excess = (bb.width + 2 * pad) - width
        if abs(excess) < 0.01:
            break
        fig.set_size_inches(max(1.0, w - excess), h)


def save(fig, path: Path, captions: dict | None = None, caption: str | None = None, layout: bool = True,
         fit_width: float | None = None) -> Path:
    """Save at the designed size, so a figure drawn at column width keeps its font sizes in the paper.

    layout=True applies tight_layout; fit_width resizes the canvas so the tight bounding box
    (labels included) is exactly that wide, then saves with a tight box."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if layout and fit_width is None:
            fig.tight_layout(pad=0.25)
        if fit_width is not None:
            fit_to_width(fig, fit_width)
            fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.03)
        else:
            fig.savefig(path, facecolor="white")
    plt.close(fig)
    if captions is not None and caption:
        captions[path.name] = caption
    return path
