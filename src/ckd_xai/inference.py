"""Single-record prediction and explanation from the saved artifacts.

Used by the Streamlit app, the notebook and the tests. Nothing here trains a
model: it only loads `models/selected_model.joblib`, its metadata and the
saved SHAP background of training records.
"""
from __future__ import annotations

import math

import joblib
import numpy as np
import pandas as pd

from . import schema
from .config import MODELS_DIR, SELECTED_META_PATH, SELECTED_MODEL_PATH, load_config
from .explain import BACKGROUND_PATH, build_explainer, display_values, model_output, output_to_probability, shap_values
from .preprocessing import transform_features
from .utils import ckd_probability, read_json

RANGES_PATH = MODELS_DIR / "training_feature_ranges.json"


def save_training_ranges(X_train: pd.DataFrame) -> dict:
    """Observed range of each continuous feature in the training data (for out-of-range warnings)."""
    from .utils import write_json

    ranges = {}
    for c in schema.CONTINUOUS + schema.ORDINAL:
        s = X_train[c].dropna()
        ranges[c] = {"min": float(s.min()), "max": float(s.max()),
                     "p01": float(s.quantile(0.01)), "p99": float(s.quantile(0.99)),
                     "median": float(s.median())}
    write_json(ranges, RANGES_PATH)
    return ranges


def record_frame(record: dict) -> pd.DataFrame:
    row = {c: record.get(c, math.nan) for c in schema.RAW_FEATURE_ORDER}
    X = pd.DataFrame([row], columns=schema.RAW_FEATURE_ORDER)
    for c in schema.BINARY:
        X[c] = X[c].astype(object)
    for c in schema.CONTINUOUS + schema.ORDINAL:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X


class CKDPredictor:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.pipeline = joblib.load(SELECTED_MODEL_PATH)
        self.meta = read_json(SELECTED_META_PATH)
        self.model_name = self.meta["model"]
        self.threshold = float(self.meta["threshold"])
        self.ranges = read_json(RANGES_PATH) if RANGES_PATH.exists() else {}
        background = joblib.load(BACKGROUND_PATH)
        self.bundle = build_explainer(self.pipeline, self.model_name, background, self.cfg)

    # ------------------------------------------------------------------ prediction
    def range_warnings(self, record: dict) -> list[str]:
        out = []
        for c, r in self.ranges.items():
            v = record.get(c)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            if v < r["min"] or v > r["max"]:
                f = schema.FEATURE_BY_NAME[c]
                out.append(f"{f.label} = {v:g} is outside the range seen in training "
                           f"({r['min']:g}-{r['max']:g}); the model is extrapolating.")
        return out

    def predict(self, raw_record: dict, strict: bool = True) -> dict:
        v = schema.validate_record(raw_record, strict_categories=strict)
        if not v.ok:
            return {"ok": False, "errors": v.errors, "warnings": v.warnings}
        X = record_frame(v.record)
        p = float(ckd_probability(self.pipeline, X)[0])
        label = int(p >= self.threshold)
        _, txt, mask = display_values(self.pipeline, X)
        imputed = {schema.FEATURE_BY_NAME[c].label: txt.iloc[0][c].replace(" (imputed)", "")
                   for c in schema.MODEL_FEATURES if mask.iloc[0][c]}
        return {"ok": True, "p_ckd": p, "label": label, "class_name": schema.CLASS_NAMES[label],
                "threshold": self.threshold, "missing": v.missing, "imputed_values": imputed,
                "warnings": v.warnings + self.range_warnings(v.record), "errors": [],
                "record": v.record}

    # ------------------------------------------------------------------ explanation
    def explain(self, clean_record: dict) -> dict:
        X = record_frame(clean_record)
        vals = shap_values(self.bundle, self.pipeline, X, self.cfg)[0]
        out = float(model_output(self.model_name, self.pipeline, transform_features(self.pipeline, X))[0])
        recon = self.bundle.base_value + float(vals.sum())
        p_from_shap = float(output_to_probability(self.model_name, recon))
        p_model = float(ckd_probability(self.pipeline, X)[0])
        num, txt, mask = display_values(self.pipeline, X)
        table = pd.DataFrame({
            "feature": schema.MODEL_FEATURES,
            "label": [schema.FEATURE_BY_NAME[c].label for c in schema.MODEL_FEATURES],
            "value": txt.iloc[0].values,
            "imputed": mask.iloc[0].values,
            "shap": vals,
        })
        table = table.reindex(table["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        return {"values": vals, "base_value": self.bundle.base_value, "model_output": out,
                "reconstructed_output": recon, "p_from_shap": p_from_shap, "p_model": p_model,
                "consistent": abs(p_from_shap - p_model) < 1e-3 and abs(recon - out) < 1e-3,
                "output_name": self.bundle.output, "table": table, "text": txt.iloc[0]}
