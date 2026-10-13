"""Download (or register a local copy of) the UCI Chronic Kidney Disease dataset.

Examples
    python scripts/download_data.py                       # use cache, download if missing
    python scripts/download_data.py --force               # re-download from UCI
    python scripts/download_data.py --local-file path/to/chronic+kidney+disease.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckd_xai.config import load_config  # noqa: E402
from ckd_xai.data import acquire, build_clean_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-file", help="official .zip, .arff or UCI data.csv to use instead of downloading")
    ap.add_argument("--force", action="store_true", help="download again even if a cached copy exists")
    args = ap.parse_args()
    cfg = load_config()
    src = acquire(cfg, local_file=args.local_file, force=args.force)
    print(f"Source: {src['method']}")
    for name, info in src["files"].items():
        flag = {True: "matches recorded checksum", False: "CHECKSUM DIFFERS", None: ""}[info["matches_expected"]]
        print(f"  {name:40s} {info['sha256'][:16]}...  {flag}")
    df, summary = build_clean_dataset(cfg)
    print(f"Clean dataset: {len(df)} records, classes {summary['class_counts']}, "
          f"{summary['missing_cells_total']} missing cells; written to data/processed/ckd_clean.csv")


if __name__ == "__main__":
    main()
