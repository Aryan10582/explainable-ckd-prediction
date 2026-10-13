"""Run the full experiment, or selected stages of it.

Stages (in order):
    data      acquire/validate/clean the dataset and lock the train/test split
    eda       exploratory figures and tables
    train     cross-validated tuning of the four models + predeclared selection (training data only)
    evaluate  final evaluation of the frozen models on the untouched test set + leakage audit
    explain   SHAP explanations of the selected model
    latency   single-record inference / explanation timing

Examples
    python scripts/run_pipeline.py                          # everything
    python scripts/run_pipeline.py --stages eda evaluate explain latency   # regenerate outputs, no retraining
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckd_xai import data  # noqa: E402
from ckd_xai.config import LOGS_DIR, SPLIT_MANIFEST_PATH, ensure_dirs, load_config  # noqa: E402
from ckd_xai.utils import env_info, now_utc, read_json, write_json  # noqa: E402

STAGES = ["data", "eda", "train", "evaluate", "explain", "latency"]


class Logger:
    def __init__(self, path: Path):
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()


def stage_latency(cfg, X_test, log):
    import numpy as np
    from ckd_xai.inference import CKDPredictor

    t0 = time.perf_counter()
    pred = CKDPredictor(cfg)
    load_s = time.perf_counter() - t0
    records = [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()}
               for row in X_test.head(cfg["latency"]["repeats"]).to_dict(orient="records")]
    t_pred, t_expl = [], []
    for r in records:
        t = time.perf_counter()
        res = pred.predict(r)
        t_pred.append(time.perf_counter() - t)
        t = time.perf_counter()
        pred.explain(res["record"])
        t_expl.append(time.perf_counter() - t)
    out = {"model": pred.model_name, "explainer": pred.bundle.kind,
           "artifact_load_and_explainer_build_s": round(load_s, 3),
           "predict_ms_median": round(1000 * float(np.median(t_pred)), 2),
           "predict_ms_p95": round(1000 * float(np.percentile(t_pred, 95)), 2),
           "explain_ms_median": round(1000 * float(np.median(t_expl)), 2),
           "explain_ms_p95": round(1000 * float(np.percentile(t_expl, 95)), 2),
           "n_records_timed": len(records),
           "note": "Single-record calls including validation and preprocessing, on the machine in run_manifest.json."}
    write_json(out, LOGS_DIR / "latency.json")
    log(f"  predict {out['predict_ms_median']} ms, explain {out['explain_ms_median']} ms (median)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    ap.add_argument("--local-file", help="use this raw data file instead of downloading")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    cfg = load_config()
    ensure_dirs()
    log = Logger(LOGS_DIR / "pipeline_log.txt")
    log(f"\n=== run started {now_utc()} | stages: {' '.join(args.stages)} ===")
    timings = {}

    def timed(name, fn):
        log(f"[{name}]")
        t = time.perf_counter()
        result = fn()
        timings[name] = round(time.perf_counter() - t, 2)
        log(f"  done in {timings[name]} s")
        return result

    if "data" in args.stages:
        def _data():
            src = data.acquire(cfg, local_file=args.local_file)
            df_, summary = data.build_clean_dataset(cfg)
            log(f"  source: {src['method']}; {len(df_)} records; classes {summary['class_counts']}; "
                f"{len(summary['line_repairs'])} line repairs; "
                f"{len(summary['implausible_values_set_missing'])} implausible values set missing")
            return df_
        timed("data", _data)

    df = data.load_clean()
    split = data.make_or_verify_split(df, cfg)
    X_train, y_train = data.get_partition(df, split, "train")
    X_test, y_test = data.get_partition(df, split, "test")
    manifest = read_json(SPLIT_MANIFEST_PATH)
    log(f"  split locked: train {len(X_train)} / test {len(X_test)} (test ids sha256 "
        f"{manifest['test_ids_sha256'][:12]}...)")

    if "eda" in args.stages:
        from ckd_xai.eda import run_eda
        timed("eda", lambda: run_eda(df, df[df.record_id.isin(X_train.index)]))

    if "train" in args.stages:
        from ckd_xai.inference import save_training_ranges
        from ckd_xai.train import run_training
        save_training_ranges(X_train)
        timed("train", lambda: run_training(X_train, y_train, cfg, manifest["test_ids_sha256"], log))

    if "evaluate" in args.stages:
        from ckd_xai.evaluate import evaluate
        res = timed("evaluate", lambda: evaluate(X_test, y_test, cfg, log))
        failed = res["audit"][~res["audit"].passed]
        log(f"  leakage audit: {int(res['audit'].passed.sum())}/{len(res['audit'])} checks passed")
        if len(failed):
            log(failed.to_string(index=False))

    if "explain" in args.stages:
        from ckd_xai.diagram import make_workflow_diagram
        from ckd_xai.explain import run_explain
        timed("explain", lambda: run_explain(X_train, X_test, y_test, cfg, log))
        make_workflow_diagram()

    if "latency" in args.stages:
        timed("latency", lambda: stage_latency(cfg, X_test, log))

    manifest_path = LOGS_DIR / "run_manifest.json"
    previous = read_json(manifest_path) if manifest_path.exists() else {}
    stage_times = previous.get("stage_seconds", {})
    stage_times.update(timings)
    write_json({"last_run_utc": now_utc(), "stages_run": args.stages, "stage_seconds": stage_times,
                "random_seed": cfg["project"]["random_seed"], "config": cfg, "environment": env_info(),
                "split": manifest}, manifest_path)
    log(f"=== run finished {now_utc()} ===")


if __name__ == "__main__":
    main()
