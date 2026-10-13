"""Explicit schema for the UCI Chronic Kidney Disease dataset.

The schema is the single source of truth for feature names, types, allowed
categories, documented units, plausibility limits and the target mapping. The
cleaning code, the model pipelines, the Streamlit form and the tests all read
from here, so they cannot drift apart.

Units are copied from the UCI metadata / dataset info file. Where the source
does not state a unit, `unit` is None and `unit_note` says so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

TARGET = "class"
ID_COLUMN = "record_id"
POSITIVE_LABEL = "ckd"
NEGATIVE_LABEL = "notckd"
# CKD is the positive class (1) everywhere: metrics, SHAP outputs and the app.
TARGET_MAPPING = {POSITIVE_LABEL: 1, NEGATIVE_LABEL: 0}
CLASS_NAMES = {1: "CKD", 0: "Non-CKD"}


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    kind: str                      # "continuous" | "ordinal" | "binary"
    unit: str | None = None
    unit_note: str = ""
    description: str = ""
    levels: tuple = ()             # ordinal: allowed numeric levels; binary: (code 0, code 1)
    plausible: tuple | None = None  # continuous: (low, high) physiological limits
    step: float = 1.0              # input-form step for continuous features

    @property
    def display(self) -> str:
        return f"{self.label} ({self.unit})" if self.unit else self.label


# Order below = column order in the original ARFF file.
FEATURES: list[Feature] = [
    Feature("age", "Age", "continuous", "years", description="Age of the patient",
            plausible=(0, 120), step=1.0),
    Feature("bp", "Blood pressure", "continuous", "mm Hg",
            unit_note="Documented as 'mm/Hg'; the source does not state systolic or diastolic.",
            description="Blood pressure", plausible=(30, 250), step=10.0),
    Feature("sg", "Specific gravity", "ordinal", None,
            unit_note="Dimensionless; recorded at five discrete levels.",
            description="Specific gravity (urine, by clinical convention; the source says only 'specific gravity')",
            levels=(1.005, 1.010, 1.015, 1.020, 1.025)),
    Feature("al", "Albumin grade", "ordinal", None,
            unit_note="Ordinal grade 0-5; no unit documented.",
            description="Albumin level, recorded as a grade from 0 to 5", levels=(0, 1, 2, 3, 4, 5)),
    Feature("su", "Sugar grade", "ordinal", None,
            unit_note="Ordinal grade 0-5; no unit documented.",
            description="Sugar level, recorded as a grade from 0 to 5", levels=(0, 1, 2, 3, 4, 5)),
    Feature("rbc", "Red blood cells", "binary", description="Red blood cells (normal / abnormal)",
            levels=("normal", "abnormal")),
    Feature("pc", "Pus cells", "binary", description="Pus cells (normal / abnormal)",
            levels=("normal", "abnormal")),
    Feature("pcc", "Pus cell clumps", "binary", description="Pus cell clumps (present / not present)",
            levels=("notpresent", "present")),
    Feature("ba", "Bacteria", "binary", description="Bacteria (present / not present)",
            levels=("notpresent", "present")),
    Feature("bgr", "Blood glucose (random)", "continuous", "mg/dL",
            unit_note="Documented as 'mgs/dl'.", description="Random blood glucose",
            plausible=(10, 1500), step=1.0),
    Feature("bu", "Blood urea", "continuous", "mg/dL", unit_note="Documented as 'mgs/dl'.",
            description="Blood urea", plausible=(1, 500), step=1.0),
    Feature("sc", "Serum creatinine", "continuous", "mg/dL", unit_note="Documented as 'mgs/dl'.",
            description="Serum creatinine", plausible=(0.1, 100), step=0.1),
    Feature("sod", "Sodium", "continuous", "mEq/L", description="Serum sodium",
            plausible=(90, 200), step=1.0),
    Feature("pot", "Potassium", "continuous", "mEq/L", description="Serum potassium",
            plausible=(1.0, 15.0), step=0.1),
    Feature("hemo", "Haemoglobin", "continuous", "g/dL",
            unit_note="Documented only as 'gms'; g/dL is the conventional unit and matches the value range.",
            description="Haemoglobin", plausible=(1.0, 25.0), step=0.1),
    Feature("pcv", "Packed cell volume", "continuous", None,
            unit_note="No unit documented by the source (conventionally %).",
            description="Packed cell volume (haematocrit)", plausible=(5, 80), step=1.0),
    Feature("wbcc", "White blood cell count", "continuous", "cells/cumm",
            unit_note="Named 'wc' in the info file; 'wbcc' in the data file.",
            description="White blood cell count", plausible=(500, 100000), step=100.0),
    Feature("rbcc", "Red blood cell count", "continuous", "millions/cmm",
            unit_note="Named 'rc' in the info file; 'rbcc' in the data file.",
            description="Red blood cell count", plausible=(0.5, 10.0), step=0.1),
    Feature("htn", "Hypertension", "binary", description="Hypertension (yes / no)", levels=("no", "yes")),
    Feature("dm", "Diabetes mellitus", "binary", description="Diabetes mellitus (yes / no)",
            levels=("no", "yes")),
    Feature("cad", "Coronary artery disease", "binary", description="Coronary artery disease (yes / no)",
            levels=("no", "yes")),
    Feature("appet", "Appetite", "binary", description="Appetite (good / poor)", levels=("good", "poor")),
    Feature("pe", "Pedal oedema", "binary", description="Pedal oedema (yes / no)", levels=("no", "yes")),
    Feature("ane", "Anaemia", "binary", description="Anaemia (yes / no)", levels=("no", "yes")),
]

FEATURE_BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}
RAW_FEATURE_ORDER = [f.name for f in FEATURES]
CONTINUOUS = [f.name for f in FEATURES if f.kind == "continuous"]
ORDINAL = [f.name for f in FEATURES if f.kind == "ordinal"]
BINARY = [f.name for f in FEATURES if f.kind == "binary"]
# Column order produced by the preprocessing step (and therefore seen by every
# classifier and by SHAP): continuous, then ordinal, then binary.
MODEL_FEATURES = CONTINUOUS + ORDINAL + BINARY
# Indices (in MODEL_FEATURES) that SMOTENC must treat as categorical.
SMOTENC_CATEGORICAL_IDX = list(range(len(CONTINUOUS), len(MODEL_FEATURES)))
ARFF_COLUMNS = RAW_FEATURE_ORDER + [TARGET]


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip() in {"", "?", "nan", "NaN"}


@dataclass
class ValidationResult:
    record: dict
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_record(raw: dict, strict_categories: bool = True) -> ValidationResult:
    """Validate one patient record for inference.

    Returns a cleaned dict with every schema feature present (NaN where
    missing). Unknown categories or impossible values are errors when
    `strict_categories` is True (the app's behaviour); otherwise they are
    converted to missing with a warning so the pipeline can impute them.
    Unrecognised extra keys are reported and ignored.
    """
    out: dict = {}
    res = ValidationResult(record=out)
    extra = set(raw) - set(RAW_FEATURE_ORDER) - {ID_COLUMN, TARGET}
    if extra:
        res.warnings.append(f"Ignored unknown fields: {sorted(extra)}")

    def reject(msg: str, name: str) -> None:
        if strict_categories:
            res.errors.append(msg)
        else:
            res.warnings.append(msg + " Treated as missing.")
        out[name] = math.nan
        res.missing.append(name)

    for f in FEATURES:
        value = raw.get(f.name)
        if is_missing(value):
            out[f.name] = math.nan
            res.missing.append(f.name)
            continue
        if f.kind == "binary":
            v = str(value).strip().lower()
            if v not in f.levels:
                reject(f"{f.label}: '{value}' is not one of {list(f.levels)}.", f.name)
            else:
                out[f.name] = v
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            reject(f"{f.label}: '{value}' is not a number.", f.name)
            continue
        if f.kind == "ordinal":
            match = [lv for lv in f.levels if abs(lv - v) < 1e-9]
            if not match:
                reject(f"{f.label}: {value} is not one of the recorded levels {list(f.levels)}.", f.name)
            else:
                out[f.name] = float(match[0])
            continue
        lo, hi = f.plausible
        if not lo <= v <= hi:
            reject(f"{f.label}: {v:g} is outside the plausible range {lo:g}-{hi:g}.", f.name)
        else:
            out[f.name] = v
    return res


def feature_dictionary_rows() -> list[dict]:
    rows = []
    for f in FEATURES:
        if f.kind == "binary":
            values = f"{f.levels[0]} (0) / {f.levels[1]} (1)"
        elif f.kind == "ordinal":
            values = ", ".join(f"{lv:g}" for lv in f.levels)
        else:
            values = f"plausibility limits {f.plausible[0]:g}-{f.plausible[1]:g}"
        rows.append({
            "code": f.name, "name": f.label, "type": f.kind,
            "unit": f.unit or "-", "values / limits": values,
            "description": f.description, "unit note": f.unit_note,
        })
    return rows
