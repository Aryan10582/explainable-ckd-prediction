"""Cached loaders and shared text for the Streamlit demo.

The app never trains anything: it only loads artifacts written by
`python scripts/run_pipeline.py`.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ckd_xai import schema  # noqa: E402
from ckd_xai.config import (CLEAN_DATA_PATH, FIGURES_DIR, PROCESSED_DIR, SELECTED_META_PATH,  # noqa: E402
                            SELECTED_MODEL_PATH, SELECTION_PATH, SPLIT_PATH, TABLES_DIR)

DISCLAIMER = (
    "Educational research prototype, not a diagnostic tool. The probability shown is an estimate from "
    "a model trained on 320 records from a single hospital dataset; it is not a clinically validated "
    "individual risk and must not be used for diagnosis or treatment decisions."
)
NOT_RECORDED = "Not recorded"
REQUIRED = [SELECTED_MODEL_PATH, SELECTED_META_PATH, SELECTION_PATH, CLEAN_DATA_PATH, SPLIT_PATH,
            TABLES_DIR / "test_metrics.csv", TABLES_DIR / "shap_global_importance.csv"]


def disclaimer() -> None:
    st.warning(DISCLAIMER, icon=":material/info:")


def require_artifacts() -> None:
    missing = [p.relative_to(ROOT).as_posix() for p in REQUIRED if not p.exists()]
    if missing:
        st.error("Saved experiment outputs are missing: " + ", ".join(missing))
        st.code("python scripts/run_pipeline.py", language="bash")
        st.stop()


@st.cache_resource(show_spinner="Loading the saved model and SHAP explainer...")
def get_predictor():
    from ckd_xai.inference import CKDPredictor

    return CKDPredictor()


@st.cache_data(max_entries=40)
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / name)


@st.cache_data(max_entries=10)
def json_file(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def figure(name: str) -> Path:
    return FIGURES_DIR / name


def captions() -> dict:
    return json_file(str(FIGURES_DIR / "figure_captions.json"))


def selection() -> dict:
    return json_file(str(SELECTION_PATH))


def cleaning_summary() -> dict:
    return json_file(str(PROCESSED_DIR / "cleaning_summary.json"))


@st.cache_data
def sample_records() -> dict[str, dict]:
    """Real records from the held-out test partition, labelled as such.

    The model is frozen, so showing them does not change any reported result.
    """
    df = pd.read_csv(CLEAN_DATA_PATH)
    split = pd.read_csv(SPLIT_PATH)
    test = df[df[schema.ID_COLUMN].isin(split.loc[split.partition == "test", schema.ID_COLUMN])]
    ex = table("shap_local_examples.csv")
    picks = [(int(r[schema.ID_COLUMN]), r["example"]) for _, r in ex.iterrows()]
    n_missing = test[schema.RAW_FEATURE_ORDER].isna().sum(axis=1)
    most_missing = int(test.loc[n_missing.idxmax(), schema.ID_COLUMN])
    if most_missing not in [p[0] for p in picks]:
        picks.append((most_missing, "most_missing"))
    names = {"typical_ckd": "typical CKD prediction", "typical_non_ckd": "typical non-CKD prediction",
             "borderline": "least certain prediction", "most_missing": "most missing values"}
    out = {}
    for rid, tag in picks:
        row = test[test[schema.ID_COLUMN] == rid].iloc[0]
        rec = {}
        for c in schema.RAW_FEATURE_ORDER:
            v = row[c]
            rec[c] = None if (isinstance(v, float) and math.isnan(v)) or pd.isna(v) else v
        label = (f"Dataset record {rid} - held-out test set, true class {schema.CLASS_NAMES[int(row.target)]} "
                 f"({names.get(tag, tag.replace('_', ' '))}, {int(n_missing[row.name])} missing)")
        out[label] = rec
    return out
