"""Project paths and configuration loading."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

CLEAN_DATA_PATH = PROCESSED_DIR / "ckd_clean.csv"
SPLIT_PATH = PROCESSED_DIR / "split.csv"
SPLIT_MANIFEST_PATH = PROCESSED_DIR / "split_manifest.json"
SELECTED_MODEL_PATH = MODELS_DIR / "selected_model.joblib"
SELECTED_META_PATH = MODELS_DIR / "selected_model_metadata.json"
SELECTION_PATH = MODELS_DIR / "selection.json"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background_ids.csv"


def load_config(path: Path | str | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def model_path(name: str) -> Path:
    return MODELS_DIR / f"{name}_pipeline.joblib"
