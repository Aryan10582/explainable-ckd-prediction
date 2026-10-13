"""Leakage-related behaviour of the preprocessing + SMOTENC pipelines."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from ckd_xai import schema
from ckd_xai.models import MODEL_NAMES
from ckd_xai.preprocessing import build_pipeline, to_original_units, transform_features


class SpyClassifier(ClassifierMixin, BaseEstimator):
    """Records how many rows it sees in fit and predict."""

    def fit(self, X, y):
        self.n_fit_, self.fit_counts_ = len(X), np.bincount(y)
        self.classes_ = np.array([0, 1])
        self.fit_X_ = np.asarray(X)
        return self

    def predict_proba(self, X):
        self.n_predict_ = len(X)
        return np.tile([0.5, 0.5], (len(X), 1))


def spy_pipeline(cfg):
    pipe = build_pipeline("logistic_regression", cfg, resample=True)
    pipe.set_params(clf=SpyClassifier())
    return pipe


def test_smotenc_resamples_only_during_fit(partitions, cfg):
    (Xtr, ytr), (Xte, _) = partitions["train"], partitions["test"]
    pipe = spy_pipeline(cfg).fit(Xtr, ytr)
    spy = pipe.named_steps["clf"]
    assert spy.n_fit_ > len(Xtr)                         # synthetic minority records added in fit
    assert spy.fit_counts_[0] == spy.fit_counts_[1] == 200
    pipe.predict_proba(Xte)
    assert spy.n_predict_ == len(Xte)                    # never resampled at prediction time


def test_synthetic_records_have_valid_categorical_levels(partitions, cfg):
    (Xtr, ytr) = partitions["train"]
    pipe = spy_pipeline(cfg).fit(Xtr, ytr)
    real = transform_features(pipe, Xtr).values
    synth = pipe.named_steps["clf"].fit_X_
    for i in schema.SMOTENC_CATEGORICAL_IDX:
        assert set(np.unique(synth[:, i])) <= set(np.unique(real[:, i]))


def test_imputation_and_scaling_use_only_the_fitting_data(partitions, cfg):
    (Xtr, ytr) = partitions["train"]
    idx = ytr.groupby(ytr).sample(frac=0.5, random_state=0).index      # stratified half
    half = Xtr.loc[idx]
    pipe = build_pipeline("logistic_regression", cfg).fit(half, ytr.loc[idx])
    prep = pipe.named_steps["prep"]
    med = prep.named_transformers_["continuous"].named_steps["impute"].statistics_
    assert np.allclose(med, half[schema.CONTINUOUS].median().values)
    assert not np.allclose(med, Xtr[schema.CONTINUOUS].median().values)


def test_preprocessed_columns_follow_model_feature_order_and_invert_exactly(partitions, cfg):
    (Xtr, ytr) = partitions["train"]
    pipe = build_pipeline("random_forest", cfg).fit(Xtr, ytr)
    Xt = transform_features(pipe, Xtr)
    assert list(Xt.columns) == schema.MODEL_FEATURES
    back = to_original_units(pipe, Xt)
    observed = Xtr[schema.CONTINUOUS].notna()
    assert np.allclose(back[schema.CONTINUOUS].values[observed.values], Xtr[schema.CONTINUOUS].values[observed.values])


def test_unknown_category_at_inference_is_treated_as_missing(partitions, cfg):
    (Xtr, ytr), (Xte, _) = partitions["train"], partitions["test"]
    pipe = build_pipeline("logistic_regression", cfg).fit(Xtr, ytr)
    rec = Xte.iloc[[0]].copy()
    unknown, missing = rec.copy(), rec.copy()
    unknown["htn"] = "maybe"
    missing["htn"] = np.nan
    assert np.allclose(pipe.predict_proba(unknown), pipe.predict_proba(missing))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_pipeline_fits_and_outputs_ckd_probability(partitions, cfg, name):
    (Xtr, ytr), (Xte, _) = partitions["train"], partitions["test"]
    pipe = build_pipeline(name, cfg).fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)
    assert list(pipe.classes_) == [0, 1] and p.shape == (len(Xte), 2)
    assert np.allclose(p.sum(axis=1), 1) and ((p >= 0) & (p <= 1)).all()


def test_predictors_cannot_include_identifier_or_target(partitions, cfg):
    (Xtr, ytr) = partitions["train"]
    pipe = build_pipeline("xgboost", cfg).fit(Xtr, ytr)
    assert list(pipe.named_steps["prep"].feature_names_in_) == schema.RAW_FEATURE_ORDER
    with_extra = Xtr.assign(record_id=Xtr.index, target=ytr.values)
    a = pipe.predict_proba(Xtr.iloc[:5])
    b = pipe.predict_proba(with_extra.iloc[:5])          # extra columns are dropped, not used
    assert np.allclose(a, b)
