"""SHAP explanations: class alignment, feature order, and additivity."""
import json

import joblib
import numpy as np
import pytest

from ckd_xai import schema
from ckd_xai.config import LOGS_DIR, SELECTED_MODEL_PATH, TABLES_DIR, load_config
from ckd_xai.explain import (additivity_error, additivity_tolerance, build_explainer, forest_copy_for_shap,
                             model_output, output_to_probability, shap_values)
from ckd_xai.models import MODEL_NAMES
from ckd_xai.preprocessing import build_pipeline, transform_features
from ckd_xai.utils import ckd_probability


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_shap_reconstructs_the_explained_output_for_every_model_type(partitions, name):
    cfg = load_config()
    cfg["shap"]["kernel_background_size"] = 20
    cfg["shap"]["kernel_nsamples"] = 300
    (Xtr, ytr), (Xte, _) = partitions["train"], partitions["test"]
    pipe = build_pipeline(name, cfg).fit(Xtr, ytr)
    bundle = build_explainer(pipe, name, Xtr.sample(40, random_state=0), cfg)
    X = Xte.iloc[:6]
    vals = shap_values(bundle, pipe, X, cfg)
    assert vals.shape == (6, len(schema.MODEL_FEATURES))
    assert additivity_error(bundle, pipe, X, vals) < additivity_tolerance(bundle)
    # the explained output maps back to the probability the pipeline reports for CKD
    p = output_to_probability(name, bundle.base_value + vals.sum(axis=1))
    assert np.allclose(p, ckd_probability(pipe, X), atol=1e-3)


def test_forest_threshold_rounding_preserves_predictions(partitions, cfg):
    (Xtr, ytr), (Xte, _) = partitions["train"], partitions["test"]
    pipe = build_pipeline("random_forest", cfg).fit(Xtr, ytr)
    clf = pipe.named_steps["clf"]
    patched, changed, total = forest_copy_for_shap(clf)
    X = transform_features(pipe, Xte).values
    assert total > 0 and np.array_equal(patched.predict_proba(X), clf.predict_proba(X))


def test_saved_explanations_pass_their_checks(artifacts_ready):
    checks = json.loads((LOGS_DIR / "shap_checks.json").read_text(encoding="utf-8"))
    assert checks["additivity_ok"]
    assert checks["feature_order"] == schema.MODEL_FEATURES
    assert checks["shap_shape"] == [80, 24]
    assert "CKD (label 1)" in checks["positive_class"]


def test_saved_shap_table_matches_model_output(artifacts_ready, partitions):
    import pandas as pd

    Xte, _ = partitions["test"]
    sv = pd.read_csv(TABLES_DIR / "shap_values_test.csv", index_col=0).loc[Xte.index]
    pipe = joblib.load(SELECTED_MODEL_PATH)
    meta_name = json.loads((LOGS_DIR / "shap_checks.json").read_text(encoding="utf-8"))["model"]
    out = model_output(meta_name, pipe, transform_features(pipe, Xte))
    recon = sv["base_value"] + sv[schema.MODEL_FEATURES].sum(axis=1)
    assert np.allclose(recon.values, out, atol=1e-6)
    assert np.allclose(sv["p_ckd"].values, ckd_probability(pipe, Xte), atol=1e-6)


def test_app_explanation_is_consistent_with_displayed_prediction(artifacts_ready, partitions):
    import math

    from ckd_xai.inference import CKDPredictor

    Xte, _ = partitions["test"]
    predictor = CKDPredictor()
    for i in (0, 7, 33):
        raw = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in Xte.iloc[i].items()}
        res = predictor.predict(raw)
        ex = predictor.explain(res["record"])
        assert ex["consistent"]
        assert ex["p_model"] == pytest.approx(res["p_ckd"], abs=1e-12)
        assert ex["p_from_shap"] == pytest.approx(res["p_ckd"], abs=1e-6)
