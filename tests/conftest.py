import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckd_xai import data  # noqa: E402
from ckd_xai.config import CLEAN_DATA_PATH, SELECTED_MODEL_PATH, load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def clean_df():
    if not CLEAN_DATA_PATH.exists():
        pytest.skip("Run `python scripts/run_pipeline.py --stages data` first.")
    return data.load_clean()


@pytest.fixture(scope="session")
def partitions(clean_df, cfg):
    split = data.make_or_verify_split(clean_df, cfg)
    return {name: data.get_partition(clean_df, split, name) for name in ("train", "test")}


@pytest.fixture(scope="session")
def artifacts_ready():
    if not SELECTED_MODEL_PATH.exists():
        pytest.skip("Run `python scripts/run_pipeline.py` first to create saved models.")
    return True
