"""Saved artifacts: reload, reproduce, and serve single records consistently."""
import json
import math

import joblib
import numpy as np
import pandas as pd
import pytest

from ckd_xai import schema
from ckd_xai.config import SELECTED_META_PATH, SELECTED_MODEL_PATH, SELECTION_PATH, TABLES_DIR, model_path
from ckd_xai.data import sha256
from ckd_xai.models import MODEL_NAMES
from ckd_xai.utils import ckd_probability, metrics_from_scores


@pytest.fixture(scope="module")
def saved(artifacts_ready):
    return {
        "pipe": joblib.load(SELECTED_MODEL_PATH),
        "meta": json.loads(SELECTED_META_PATH.read_text(encoding="utf-8")),
        "selection": json.loads(SELECTION_PATH.read_text(encoding="utf-8")),
        "preds": pd.read_csv(TABLES_DIR / "test_predictions.csv"),
        "metrics": pd.read_csv(TABLES_DIR / "test_metrics.csv"),
    }


def test_metadata_matches_selection_and_file(saved):
    meta, sel = saved["meta"], saved["selection"]
    assert meta["model"] == sel["selected_model"]
    assert sel["test_set_used"] is False
    assert meta["positive_class"] == {"label": 1, "name": "ckd"}
    assert meta["model_feature_order"] == schema.MODEL_FEATURES
    assert meta["model_file_sha256"] == sha256(SELECTED_MODEL_PATH)


def test_reloaded_models_reproduce_saved_test_predictions(saved, partitions):
    Xte, yte = partitions["test"]
    preds = saved["preds"].set_index(schema.ID_COLUMN).loc[Xte.index]
    assert (preds["true_label"].values == yte.values).all()
    for name in MODEL_NAMES:
        p = ckd_probability(joblib.load(model_path(name)), Xte)
        assert np.allclose(p, preds[f"{name}_p_ckd"].values, atol=1e-6), name


def test_reported_metrics_recompute_from_saved_predictions(saved):
    preds, metrics = saved["preds"], saved["metrics"].set_index("model")
    for name in MODEL_NAMES:
        m = metrics_from_scores(preds["true_label"], preds[f"{name}_p_ckd"])
        for k in ("roc_auc", "accuracy", "precision", "recall", "f1"):
            assert m[k] == pytest.approx(metrics.loc[name, k], abs=1e-6), (name, k)


def test_selected_model_was_chosen_by_the_predeclared_rule(saved, cfg):
    from ckd_xai.train import select_model

    cv = pd.read_csv(TABLES_DIR / "cv_model_comparison.csv")
    winner, _, _ = select_model(cv, cfg)
    assert winner == saved["meta"]["model"]


def test_single_record_inference_matches_batch(saved, partitions):
    from ckd_xai.inference import CKDPredictor

    Xte, _ = partitions["test"]
    predictor = CKDPredictor()
    batch = ckd_probability(saved["pipe"], Xte)
    for i in range(5):
        raw = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in Xte.iloc[i].items()}
        res = predictor.predict(raw)
        assert res["ok"]
        assert res["p_ckd"] == pytest.approx(batch[i], abs=1e-9)
        assert res["label"] == int(batch[i] >= 0.5)


def test_inference_rejects_invalid_input_and_imputes_missing(artifacts_ready):
    from ckd_xai.inference import CKDPredictor

    predictor = CKDPredictor()
    assert not predictor.predict({"htn": "maybe"})["ok"]
    res = predictor.predict({})                                   # everything missing
    assert res["ok"] and len(res["missing"]) == 24 and len(res["imputed_values"]) == 24
    res = predictor.predict({"sc": 90.0})                         # plausible but outside training range
    assert res["ok"] and any("outside the range seen in training" in w for w in res["warnings"])
