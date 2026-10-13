"""Acquisition, parsing, cleaning and split tests."""
import json

import numpy as np
import pytest

from ckd_xai import data, schema
from ckd_xai.config import PROCESSED_DIR, RAW_DIR, SPLIT_MANIFEST_PATH

HEADER = "@relation x\n" + "".join(f"@attribute '{c}' numeric\n" for c in schema.ARFF_COLUMNS) + "@data\n"
GOOD = "48,80,1.020,1,0,?,normal,notpresent,notpresent,121,36,1.2,?,?,15.4,44,7800,5.2,yes,yes,no,good,no,no,ckd"


def _parse(lines):
    return data.parse_arff_text(HEADER + "\n".join(lines))


def test_parser_repairs_known_formatting_problems():
    trailing = GOOD + ","
    doubled = GOOD.replace("yes,yes,no", "yes,,yes,no")
    tabs = GOOD.replace(",yes,yes,", ",\tyes, yes,").replace(",ckd", ",ckd\t")
    df, fixes = _parse([GOOD, trailing, doubled, tabs])
    assert df.shape == (4, 25)
    assert [f["issue"] for f in fixes] == ["trailing comma", "doubled comma after field 19"]
    clean, summary = data.clean(df)
    assert clean[schema.TARGET].tolist() == ["ckd"] * 4          # 'ckd\t' normalised
    assert clean["dm"].tolist() == ["yes"] * 4                    # ' yes' / '\tyes' normalised
    assert clean["rbc"].isna().all()                             # '?' -> missing (categorical)
    assert clean["sod"].isna().all()                             # '?' -> missing (numeric)
    assert clean["sc"].dtype == float                             # numeric strings -> numbers
    assert summary["whitespace_fixed_cells"]["class"] == 1


def test_parser_rejects_unrepairable_line():
    with pytest.raises(ValueError):
        _parse([GOOD + ",extra,fields"])


def test_unknown_label_raises():
    df, _ = _parse([GOOD.replace(",ckd", ",maybe")])
    with pytest.raises(ValueError, match="Unknown class labels"):
        data.clean(df)


def test_implausible_value_becomes_missing_but_record_is_kept():
    df, _ = _parse([GOOD.replace(",121,36,1.2,?,?,", ",121,36,1.2,4.5,47,")])
    clean, summary = data.clean(df)
    assert len(clean) == 1
    assert np.isnan(clean.loc[0, "sod"]) and np.isnan(clean.loc[0, "pot"])
    assert {d["feature"] for d in summary["implausible_values_set_missing"]} == {"sod", "pot"}


def test_real_dataset_matches_uci_documentation(clean_df, cfg):
    assert len(clean_df) == 400
    assert len([c for c in clean_df.columns if c in schema.RAW_FEATURE_ORDER]) == 24
    assert clean_df[schema.TARGET].value_counts().to_dict() == {"ckd": 250, "notckd": 150}
    assert set(clean_df["target"]) == {0, 1}
    assert (clean_df.loc[clean_df[schema.TARGET] == "ckd", "target"] == 1).all()   # CKD is positive
    assert data.validate_clean(clean_df, cfg) == []


def test_raw_file_checksum_and_csv_cross_check(cfg):
    arff = RAW_DIR / data.ARFF_NAME
    if not arff.exists():
        pytest.skip("raw ARFF not cached")
    assert data.sha256(arff) == cfg["data"]["expected_sha256"][data.ARFF_NAME]
    summary = json.loads((PROCESSED_DIR / "cleaning_summary.json").read_text(encoding="utf-8"))
    assert summary["cross_check"]["n_cells_different"] == 0
    assert summary["n_records_clean"] == summary["n_records_raw"] == 400


def test_no_duplicate_patient_vectors(clean_df):
    rep = data.duplicate_report(clean_df)
    assert rep["n_conflicting_groups"] == 0
    assert data.row_keys(clean_df).nunique() == len(clean_df) - rep["n_records_in_duplicate_groups"] + rep["n_groups"]


def test_split_is_stratified_disjoint_and_locked(partitions, clean_df, cfg):
    (Xtr, ytr), (Xte, yte) = partitions["train"], partitions["test"]
    assert len(Xtr) == 320 and len(Xte) == 80
    assert not set(Xtr.index) & set(Xte.index)
    assert ytr.sum() == 200 and yte.sum() == 50
    assert schema.ID_COLUMN not in Xtr.columns and schema.TARGET not in Xtr.columns and "target" not in Xtr.columns
    manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    again = data._compute_split(clean_df, cfg)                    # recomputing gives the same test ids
    assert sorted(again.loc[again.partition == "test", schema.ID_COLUMN]) == sorted(Xte.index)
    assert manifest["n_test"] == 80
