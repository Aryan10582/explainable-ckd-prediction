"""Leak-free preprocessing + SMOTENC + classifier pipelines.

Pipeline order (imbalanced-learn Pipeline, so resampling happens in fit only):

    prep      ColumnTransformer, fitted on the training fold only
              * continuous -> median imputation -> standardisation
              * ordinal    -> most-frequent imputation -> standardisation
              * binary     -> 0/1 encoding (unknown -> missing) -> most-frequent imputation
    resample  SMOTENC (or 'passthrough' for the no-resampling baseline)
    clf       classifier

Why this order:
* SMOTENC needs complete data, so imputation comes first.
* SMOTENC finds nearest neighbours with Euclidean distance on the continuous
  columns, so they must be on a common scale before resampling.
* Ordinal levels (specific gravity, albumin, sugar) and binary columns are
  passed to SMOTENC as *categorical*: synthetic records copy the most frequent
  neighbour value instead of interpolating, so no synthetic patient gets an
  impossible level such as albumin 2.4 or "0.6 hypertension".
* At predict time imbalanced-learn skips the resampling step, so validation,
  test and new records are never resampled.

Every predictor maps to exactly one model column, so SHAP values need no
aggregation over one-hot columns.
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler

from . import schema
from .models import make_classifier


def as_object_array(X) -> np.ndarray:
    """Categorical inputs as a plain object array with np.nan for missing.

    pandas 3 may hand over its new string dtype (or pd.NA); scikit-learn's
    OrdinalEncoder only handles unknown categories robustly for object input.
    """
    import pandas as pd

    df = pd.DataFrame(X)
    return df.astype(object).where(df.notna(), np.nan).to_numpy()


def build_preprocessor() -> ColumnTransformer:
    continuous = SkPipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    ordinal = SkPipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("scale", StandardScaler()),
    ])
    binary = SkPipeline([
        ("as_object", FunctionTransformer(as_object_array, feature_names_out="one-to-one")),
        ("encode", OrdinalEncoder(
            categories=[list(schema.FEATURE_BY_NAME[c].levels) for c in schema.BINARY],
            handle_unknown="use_encoded_value", unknown_value=np.nan,
            encoded_missing_value=np.nan, dtype=np.float64)),
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    ct = ColumnTransformer(
        [("continuous", continuous, schema.CONTINUOUS),
         ("ordinal", ordinal, schema.ORDINAL),
         ("binary", binary, schema.BINARY)],
        remainder="drop", verbose_feature_names_out=False)
    return ct


def build_resampler(cfg: dict, seed: int) -> SMOTENC:
    return SMOTENC(categorical_features=schema.SMOTENC_CATEGORICAL_IDX,
                   k_neighbors=cfg["resampling"]["k_neighbors"], random_state=seed)


def build_pipeline(model_name: str, cfg: dict, resample: bool = True, params: dict | None = None) -> Pipeline:
    seed = cfg["project"]["random_seed"]
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("resample", build_resampler(cfg, seed) if resample else "passthrough"),
        ("clf", make_classifier(model_name, seed)),
    ])
    if params:
        pipe.set_params(**params)
    return pipe


def transform_features(pipeline: Pipeline, X) -> "np.ndarray":
    """Apply only the fitted preprocessing (what the classifier sees at predict time)."""
    import pandas as pd

    Xt = pipeline.named_steps["prep"].transform(X[schema.RAW_FEATURE_ORDER])
    return pd.DataFrame(Xt, columns=schema.MODEL_FEATURES, index=getattr(X, "index", None))


def to_original_units(pipeline: Pipeline, Xt) -> "pd.DataFrame":
    """Map preprocessed values back to clinical units (after imputation).

    Standardisation is an invertible per-feature affine map, so each model
    column corresponds to exactly one original feature value. Binary columns
    are returned as their 0/1 code (1 = abnormal / present / yes / poor).
    """
    import pandas as pd

    prep = pipeline.named_steps["prep"]
    Xt = pd.DataFrame(np.asarray(Xt, dtype=float), columns=schema.MODEL_FEATURES,
                      index=getattr(Xt, "index", None))
    out = Xt.copy()
    nc, no = len(schema.CONTINUOUS), len(schema.ORDINAL)
    sc_c = prep.named_transformers_["continuous"].named_steps["scale"]
    sc_o = prep.named_transformers_["ordinal"].named_steps["scale"]
    out[schema.CONTINUOUS] = sc_c.inverse_transform(Xt[schema.CONTINUOUS].values)
    out[schema.ORDINAL] = sc_o.inverse_transform(Xt[schema.ORDINAL].values)
    # Remove floating-point noise from the inverse transform on discrete levels.
    for c in schema.ORDINAL:
        levels = np.array(schema.FEATURE_BY_NAME[c].levels, dtype=float)
        out[c] = levels[np.abs(out[c].values[:, None] - levels[None, :]).argmin(axis=1)]
    assert nc + no + len(schema.BINARY) == Xt.shape[1]
    return out


def imputed_mask(X) -> "pd.DataFrame":
    """True where a raw value is missing or not an allowed category (so it gets imputed)."""
    import pandas as pd

    mask = pd.DataFrame(False, index=X.index, columns=schema.MODEL_FEATURES)
    for c in schema.MODEL_FEATURES:
        f = schema.FEATURE_BY_NAME[c]
        col = X[c]
        miss = col.isna()
        if f.kind == "binary":
            miss = miss | ~col.isin(f.levels)
        mask[c] = miss.values
    return mask
