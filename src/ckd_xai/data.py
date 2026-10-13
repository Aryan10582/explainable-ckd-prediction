"""Dataset acquisition, parsing, cleaning, validation and the locked train/test split.

Source: UCI Machine Learning Repository, dataset 336 (Chronic Kidney Disease),
doi:10.24432/C5G020. The official archive contains a RAR file holding the
original ARFF files; UCI also serves a converted `data.csv`. The ARFF is used
as the primary raw file and the CSV as a fallback and an independent
cross-check.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import schema
from .config import (CLEAN_DATA_PATH, PROCESSED_DIR, RAW_DIR, SPLIT_MANIFEST_PATH, SPLIT_PATH,
                     TABLES_DIR, ensure_dirs)

ARFF_NAME = "chronic_kidney_disease_full.arff"
CSV_NAME = "uci_data.csv"
ZIP_NAME = "uci_336_chronic_kidney_disease.zip"
META_NAME = "uci_metadata.json"
SOURCE_NAME = "SOURCE.json"


# --------------------------------------------------------------------------- utilities
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, timeout: int = 60) -> None:
    import requests

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def _extract_rar(rar_path: Path, out_dir: Path) -> bool:
    """Extract a RAR archive with whichever tool is available (bsdtar/tar, unrar, 7z)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        ["bsdtar", "-xf", str(rar_path), "-C", str(out_dir)],
        ["tar", "-xf", str(rar_path), "-C", str(out_dir)],      # Windows 10+/macOS tar is bsdtar
        ["unrar", "x", "-o+", str(rar_path), str(out_dir)],
        ["7z", "x", "-y", f"-o{out_dir}", str(rar_path)],
    ]
    for cmd in candidates:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            continue
        if list(out_dir.rglob(ARFF_NAME)):
            return True
    return False


def _extract_zip_payload(zip_path: Path) -> Path | None:
    """Unpack the official zip; return the path of the full ARFF if it could be extracted."""
    extract_dir = RAW_DIR / "extracted"
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (extract_dir / member).resolve()
            if extract_dir.resolve() not in target.parents:
                raise ValueError(f"Unsafe path in archive: {member}")
        zf.extractall(extract_dir)
    found = list(extract_dir.rglob(ARFF_NAME))
    if not found:
        for rar in extract_dir.rglob("*.rar"):
            if _extract_rar(rar, extract_dir):
                found = list(extract_dir.rglob(ARFF_NAME))
                break
    if not found:
        return None
    for f in found[0].parent.iterdir():
        if f.is_file():
            shutil.copy2(f, RAW_DIR / f.name)
    return RAW_DIR / ARFF_NAME


# --------------------------------------------------------------------------- acquisition
def acquire(cfg: dict, local_file: str | Path | None = None, force: bool = False) -> dict:
    """Make sure a genuine raw copy of the dataset is cached in data/raw.

    Order: user-supplied local file -> existing cache -> official UCI zip ->
    official UCI CSV. Writes data/raw/SOURCE.json with URLs, time and SHA-256.
    """
    ensure_dirs()
    dcfg = cfg["data"]
    log: list[str] = []
    arff_path, csv_path = RAW_DIR / ARFF_NAME, RAW_DIR / CSV_NAME
    method = None

    if local_file is not None:
        src = Path(local_file)
        if not src.exists():
            raise FileNotFoundError(src)
        suffix = src.suffix.lower()
        if suffix == ".zip":
            shutil.copy2(src, RAW_DIR / ZIP_NAME)
            if _extract_zip_payload(RAW_DIR / ZIP_NAME) is None:
                raise RuntimeError("The supplied zip did not contain an extractable ARFF file.")
        elif suffix == ".arff":
            shutil.copy2(src, arff_path)
        elif suffix == ".csv":
            shutil.copy2(src, csv_path)
        else:
            raise ValueError("Supply the official .zip, the .arff or the UCI data.csv file.")
        method = f"local file: {src.name}"
        log.append(f"Copied user-supplied file {src} into data/raw.")
    elif (arff_path.exists() or csv_path.exists()) and not force:
        method = "cached copy in data/raw"
        log.append("Using cached raw files (no download).")
    else:
        try:
            _download(dcfg["zip_url"], RAW_DIR / ZIP_NAME)
            log.append(f"Downloaded {dcfg['zip_url']}")
            if _extract_zip_payload(RAW_DIR / ZIP_NAME):
                method = "official UCI archive (ARFF)"
            else:
                log.append("Could not extract the RAR inside the official zip; will use the UCI CSV.")
        except Exception as exc:  # network or archive problems
            log.append(f"Zip download/extraction failed: {exc!r}")
        try:
            _download(dcfg["csv_url"], csv_path)
            log.append(f"Downloaded {dcfg['csv_url']}")
            method = method or "official UCI data.csv"
        except Exception as exc:
            log.append(f"CSV download failed: {exc!r}")
        try:
            _download(dcfg["metadata_url"], RAW_DIR / META_NAME)
            log.append("Downloaded UCI metadata JSON.")
        except Exception as exc:
            log.append(f"Metadata download failed: {exc!r}")
        if method is None:
            raise RuntimeError(
                "Could not obtain the dataset. Download the zip manually from "
                f"{dcfg['repository_url']} and run: python scripts/download_data.py --local-file <path>")

    checks = {}
    for name in sorted(p.name for p in RAW_DIR.iterdir() if p.is_file() and p.name != SOURCE_NAME):
        digest = sha256(RAW_DIR / name)
        expected = dcfg.get("expected_sha256", {}).get(name)
        checks[name] = {"sha256": digest, "matches_expected": None if expected is None else digest == expected}
        if expected is not None and digest != expected:
            warnings.warn(f"{name}: SHA-256 differs from the recorded value; the upstream file may have changed.")
    source = {
        "dataset": "Chronic Kidney Disease (UCI ML Repository id 336)",
        "doi": dcfg["dataset_doi"],
        "repository_url": dcfg["repository_url"],
        "zip_url": dcfg["zip_url"],
        "csv_url": dcfg["csv_url"],
        "metadata_url": dcfg["metadata_url"],
        "method": method,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": checks,
        "log": log,
    }
    previous = RAW_DIR / SOURCE_NAME
    if previous.exists() and method == "cached copy in data/raw":
        old = json.loads(previous.read_text(encoding="utf-8"))
        source["first_retrieved"] = old.get("first_retrieved", old.get("recorded_at_utc"))
        source["original_method"] = old.get("original_method", old.get("method"))
    else:
        source["first_retrieved"] = source["recorded_at_utc"]
        source["original_method"] = method
    previous.write_text(json.dumps(source, indent=2), encoding="utf-8")
    return source


# --------------------------------------------------------------------------- parsing
def parse_arff_text(text: str) -> tuple[pd.DataFrame, list[dict]]:
    """Parse the data section of the CKD ARFF file into raw strings.

    The file has a few malformed lines; each fix is recorded:
    * a trailing comma (26 fields, last one empty)  -> drop the empty last field
    * a doubled comma (26 fields, one interior field empty) -> drop the empty field
    Anything else that does not give 25 fields raises an error.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip().lower() == "@data") + 1
    except StopIteration as exc:
        raise ValueError("No @data section found") from exc
    rows, fixes = [], []
    n_cols = len(schema.ARFF_COLUMNS)
    for line in lines[start:]:
        if not line.strip() or line.strip().startswith("%"):
            continue
        fields = line.split(",")
        row_idx = len(rows)
        if len(fields) == n_cols + 1 and fields[-1].strip() == "":
            fields = fields[:-1]
            fixes.append({"record_id": row_idx, "issue": "trailing comma", "action": "dropped empty last field"})
        elif len(fields) == n_cols + 1:
            empty = [i for i, x in enumerate(fields) if x.strip() == ""]
            if len(empty) != 1:
                raise ValueError(f"Cannot repair line {row_idx}: {line!r}")
            fields.pop(empty[0])
            fixes.append({"record_id": row_idx, "issue": f"doubled comma after field {empty[0]}",
                          "action": "dropped the empty field"})
        if len(fields) != n_cols:
            raise ValueError(f"Line {row_idx} has {len(fields)} fields: {line!r}")
        rows.append(fields)
    return pd.DataFrame(rows, columns=schema.ARFF_COLUMNS, dtype=object), fixes


def parse_csv(path: Path) -> tuple[pd.DataFrame, list[dict]]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False).astype(object)
    if list(df.columns) != schema.ARFF_COLUMNS:
        raise ValueError(f"Unexpected CSV columns: {list(df.columns)}")
    return df, []


def load_raw() -> tuple[pd.DataFrame, list[dict], str]:
    arff_path, csv_path = RAW_DIR / ARFF_NAME, RAW_DIR / CSV_NAME
    if arff_path.exists():
        df, fixes = parse_arff_text(arff_path.read_bytes().decode("latin-1"))
        return df, fixes, ARFF_NAME
    if csv_path.exists():
        df, fixes = parse_csv(csv_path)
        return df, fixes, CSV_NAME
    raise FileNotFoundError("No raw data found. Run: python scripts/download_data.py")


# --------------------------------------------------------------------------- cleaning
MISSING_TOKENS = {"?", "", "nan", "NaN"}


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Turn the raw string table into a typed, validated table.

    No record is dropped. Returns the clean frame (with `record_id`, the 24
    features, `class` label and binary `target`) and a summary of every change.
    """
    df = raw.copy()
    summary: dict = {"n_records_raw": int(len(df))}
    ws = {}
    for col in df.columns:
        s = df[col].astype(object)
        stripped = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        changed = int((s != stripped).sum())
        if changed:
            ws[col] = changed
        df[col] = stripped
    summary["whitespace_fixed_cells"] = ws

    missing_tokens = {}
    for col in df.columns:
        mask = df[col].map(lambda v: v is None or (isinstance(v, str) and v in MISSING_TOKENS))
        missing_tokens[col] = int(mask.sum())
        df.loc[mask, col] = np.nan
    summary["missing_tokens_converted"] = {k: v for k, v in missing_tokens.items() if v}

    labels = df[schema.TARGET].str.lower()
    bad = ~labels.isin(schema.TARGET_MAPPING)
    if bad.any():
        raise ValueError(f"Unknown class labels: {labels[bad].unique().tolist()}")
    df[schema.TARGET] = labels

    implausible, invalid = [], []
    for f in schema.FEATURES:
        col = df[f.name]
        if f.kind == "binary":
            lowered = col.map(lambda v: v.lower() if isinstance(v, str) else v)
            bad_mask = lowered.notna() & ~lowered.isin(f.levels)
            for rid in df.index[bad_mask]:
                invalid.append({"record_id": int(rid), "feature": f.name, "value": col[rid]})
            df[f.name] = lowered.where(~bad_mask, np.nan).astype(object)
            continue
        numeric = pd.to_numeric(col, errors="coerce")
        not_numeric = col.notna() & numeric.isna()
        if not_numeric.any():
            raise ValueError(f"{f.name}: non-numeric values {col[not_numeric].unique().tolist()}")
        if f.kind == "ordinal":
            ok = numeric.isna() | numeric.map(lambda v: any(abs(v - lv) < 1e-9 for lv in f.levels))
            for rid in df.index[~ok]:
                invalid.append({"record_id": int(rid), "feature": f.name, "value": float(numeric[rid])})
            numeric = numeric.where(ok, np.nan)
        else:
            lo, hi = f.plausible
            out = numeric.notna() & ((numeric < lo) | (numeric > hi))
            for rid in df.index[out]:
                implausible.append({"record_id": int(rid), "feature": f.name, "value": float(numeric[rid]),
                                    "plausible_range": [lo, hi], "class": df.loc[rid, schema.TARGET],
                                    "action": "set to missing (record kept)"})
            numeric = numeric.where(~out, np.nan)
        df[f.name] = numeric.astype(float)
    summary["implausible_values_set_missing"] = implausible
    summary["invalid_categories_set_missing"] = invalid

    df.insert(0, schema.ID_COLUMN, np.arange(len(df)))
    df["target"] = df[schema.TARGET].map(schema.TARGET_MAPPING).astype(int)
    df = df[[schema.ID_COLUMN] + schema.RAW_FEATURE_ORDER + [schema.TARGET, "target"]]

    summary["n_records_clean"] = int(len(df))
    summary["class_counts"] = df[schema.TARGET].value_counts().to_dict()
    summary["target_mapping"] = schema.TARGET_MAPPING
    summary["duplicates"] = duplicate_report(df)
    return df, summary


def row_keys(X: pd.DataFrame) -> pd.Series:
    """One string per record over the 24 features; missing values compare equal."""
    cols = [X[c].map(lambda v: "nan" if pd.isna(v) else repr(v)) for c in schema.RAW_FEATURE_ORDER]
    return pd.concat(cols, axis=1).agg("|".join, axis=1)


def duplicate_report(df: pd.DataFrame) -> dict:
    feats = df[schema.RAW_FEATURE_ORDER].map(lambda v: "nan" if pd.isna(v) else repr(v))
    dup_mask = feats.duplicated(keep=False)
    groups = []
    if dup_mask.any():
        for _, g in df[dup_mask].groupby(list(feats[dup_mask].columns), dropna=False):
            groups.append({"record_ids": g[schema.ID_COLUMN].tolist(),
                           "labels": sorted(g[schema.TARGET].unique().tolist())})
    return {
        "n_records_in_duplicate_groups": int(dup_mask.sum()),
        "n_groups": len(groups),
        "n_conflicting_groups": sum(len(g["labels"]) > 1 for g in groups),
        "groups": groups,
    }


def validate_clean(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Return a list of schema violations (empty list = valid)."""
    problems = []
    dcfg = cfg["data"]
    if len(df) != dcfg["expected_records"]:
        problems.append(f"expected {dcfg['expected_records']} records, found {len(df)}")
    feats = [c for c in df.columns if c in schema.RAW_FEATURE_ORDER]
    if len(feats) != dcfg["expected_features"]:
        problems.append(f"expected {dcfg['expected_features']} features, found {len(feats)}")
    counts = df[schema.TARGET].value_counts().to_dict()
    if counts != dcfg["expected_class_counts"]:
        problems.append(f"class counts {counts} differ from {dcfg['expected_class_counts']}")
    if df[schema.ID_COLUMN].duplicated().any():
        problems.append("duplicate record ids")
    for f in schema.FEATURES:
        col = df[f.name].dropna()
        if f.kind == "binary" and not col.isin(f.levels).all():
            problems.append(f"{f.name}: unexpected categories")
        if f.kind == "ordinal" and not col.map(lambda v: any(abs(v - lv) < 1e-9 for lv in f.levels)).all():
            problems.append(f"{f.name}: unexpected levels")
        if f.kind == "continuous" and ((col < f.plausible[0]) | (col > f.plausible[1])).any():
            problems.append(f"{f.name}: values outside plausibility limits")
    return problems


def cross_check_with_csv(clean_df: pd.DataFrame) -> dict | None:
    """Compare the cleaned ARFF with UCI's own CSV conversion (if both exist)."""
    csv_path = RAW_DIR / CSV_NAME
    if not csv_path.exists() or not (RAW_DIR / ARFF_NAME).exists():
        return None
    other, _ = clean(parse_csv(csv_path)[0])
    diffs = {}
    for c in schema.RAW_FEATURE_ORDER + [schema.TARGET]:
        a, b = clean_df[c], other[c]
        same = (a == b) | (a.isna() & b.isna())
        if not same.all():
            diffs[c] = int((~same).sum())
    return {"compared_with": CSV_NAME, "n_cells_different": sum(diffs.values()), "by_column": diffs}


def build_clean_dataset(cfg: dict) -> tuple[pd.DataFrame, dict]:
    raw, fixes, source_file = load_raw()
    df, summary = clean(raw)
    summary["source_file"] = source_file
    summary["line_repairs"] = fixes
    summary["cross_check"] = cross_check_with_csv(df) if source_file == ARFF_NAME else None
    problems = validate_clean(df, cfg)
    summary["validation_problems"] = problems
    if problems:
        raise ValueError("Dataset validation failed: " + "; ".join(problems))

    miss = df[schema.RAW_FEATURE_ORDER].isna()
    summary["missing_cells_total"] = int(miss.values.sum())
    summary["records_with_any_missing"] = int(miss.any(axis=1).sum())
    summary["complete_records_by_class"] = (
        df[~miss.any(axis=1)][schema.TARGET].value_counts().to_dict())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    (PROCESSED_DIR / "cleaning_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                                        encoding="utf-8")
    mt = pd.DataFrame({
        "feature": schema.RAW_FEATURE_ORDER,
        "name": [schema.FEATURE_BY_NAME[c].label for c in schema.RAW_FEATURE_ORDER],
        "missing_n": miss.sum(axis=0).values,
        "missing_pct": (100 * miss.mean(axis=0)).round(1).values,
    }).sort_values("missing_n", ascending=False)
    mt.to_csv(TABLES_DIR / "missingness_by_feature_all_records.csv", index=False)
    return df, summary


def load_clean() -> pd.DataFrame:
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError("Clean data not found. Run: python scripts/run_pipeline.py --stages data")
    df = pd.read_csv(CLEAN_DATA_PATH)
    for c in schema.BINARY:
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df


# --------------------------------------------------------------------------- split
def _compute_split(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    seed = cfg["project"]["random_seed"]
    test_size = cfg["split"]["test_size"]
    dups = duplicate_report(df)
    # Identical records are split as one unit so no patient vector can appear on both sides.
    group = df[schema.ID_COLUMN].copy()
    for g in dups["groups"]:
        group[df[schema.ID_COLUMN].isin(g["record_ids"])] = min(g["record_ids"])
    reps = df.assign(group=group).groupby("group").agg(target=("target", "max")).reset_index()
    tr_g, te_g = train_test_split(reps["group"], test_size=test_size, random_state=seed,
                                  stratify=reps["target"] if cfg["split"]["stratify"] else None)
    part = np.where(group.isin(set(te_g)), "test", "train")
    return pd.DataFrame({schema.ID_COLUMN: df[schema.ID_COLUMN].values, "partition": part})


def make_or_verify_split(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create the hold-out split once; on later runs verify it is unchanged (locked)."""
    split = _compute_split(df, cfg)
    test_ids = sorted(split.loc[split.partition == "test", schema.ID_COLUMN].tolist())
    digest = hashlib.sha256(",".join(map(str, test_ids)).encode()).hexdigest()
    if SPLIT_PATH.exists() and SPLIT_MANIFEST_PATH.exists():
        manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest["test_ids_sha256"] != digest:
            raise RuntimeError("Regenerated split differs from the locked split; refusing to continue.")
        return pd.read_csv(SPLIT_PATH)
    merged = df.merge(split, on=schema.ID_COLUMN)
    manifest = {
        "random_seed": cfg["project"]["random_seed"],
        "test_size": cfg["split"]["test_size"],
        "stratified": cfg["split"]["stratify"],
        "n_train": int((split.partition == "train").sum()),
        "n_test": int((split.partition == "test").sum()),
        "class_counts": {p: merged[merged.partition == p][schema.TARGET].value_counts().to_dict()
                         for p in ("train", "test")},
        "test_ids_sha256": digest,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Test partition locked before any tuning; it is used only for final evaluation.",
    }
    split.to_csv(SPLIT_PATH, index=False)
    SPLIT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return split


def get_partition(df: pd.DataFrame, split: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for one partition with predictors only (no id, no target)."""
    ids = split.loc[split.partition == name, schema.ID_COLUMN]
    part = df[df[schema.ID_COLUMN].isin(ids)].sort_values(schema.ID_COLUMN)
    X = part.set_index(schema.ID_COLUMN)[schema.RAW_FEATURE_ORDER]
    y = part.set_index(schema.ID_COLUMN)["target"]
    return X, y
