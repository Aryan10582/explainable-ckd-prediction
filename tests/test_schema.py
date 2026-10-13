"""Schema definitions and inference-time record validation."""
import math

from ckd_xai import schema


def test_feature_lists_are_consistent():
    assert len(schema.FEATURES) == 24
    assert sorted(schema.MODEL_FEATURES) == sorted(schema.RAW_FEATURE_ORDER)
    assert (len(schema.CONTINUOUS), len(schema.ORDINAL), len(schema.BINARY)) == (11, 3, 10)
    cat = [schema.MODEL_FEATURES[i] for i in schema.SMOTENC_CATEGORICAL_IDX]
    assert cat == schema.ORDINAL + schema.BINARY
    assert schema.TARGET_MAPPING == {"ckd": 1, "notckd": 0}


def test_missing_values_are_reported_not_invented():
    res = schema.validate_record({"age": 50, "sc": None, "htn": "", "sg": "?"})
    assert res.ok
    assert math.isnan(res.record["sc"]) and math.isnan(res.record["htn"]) and math.isnan(res.record["sg"])
    assert {"sc", "htn", "sg", "bp"} <= set(res.missing)
    assert len(res.missing) == 23


def test_unknown_category_is_rejected_in_strict_mode():
    res = schema.validate_record({"htn": "maybe"})
    assert not res.ok and "Hypertension" in res.errors[0]


def test_unknown_category_becomes_missing_in_lenient_mode():
    res = schema.validate_record({"htn": "maybe"}, strict_categories=False)
    assert res.ok and math.isnan(res.record["htn"]) and res.warnings


def test_out_of_range_and_invalid_levels():
    assert not schema.validate_record({"pot": 47}).ok             # impossible potassium
    assert not schema.validate_record({"sg": 1.012}).ok           # not a recorded level
    assert not schema.validate_record({"age": "old"}).ok
    ok = schema.validate_record({"sg": "1.010", "al": 3, "htn": " YES "})
    assert ok.ok and ok.record["sg"] == 1.010 and ok.record["htn"] == "yes"


def test_extra_fields_are_ignored_with_warning():
    res = schema.validate_record({"age": 40, "favourite_colour": "blue"})
    assert res.ok and any("favourite_colour" in w for w in res.warnings)
