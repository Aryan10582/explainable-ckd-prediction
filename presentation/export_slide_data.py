"""Export the numbers the slides need (from the saved results) to presentation/slide_data.json.

    python presentation/export_slide_data.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckd_xai.models import DISPLAY_NAMES, MODEL_NAMES  # noqa: E402
from ckd_xai.report_data import load_results, median_iqr  # noqa: E402


def main():
    R = load_results()
    cv, test, W = R["cv"], R["test"], R["winner"]
    tw = test.loc[W]
    loc = R["local"]
    imp = R["shap_imp"]
    shapc = R["shap_checks"]

    def params(m):
        p = eval(cv.loc[m, "best_params"]) if isinstance(cv.loc[m, "best_params"], str) else cv.loc[m, "best_params"]
        if m == "logistic_regression":
            return f"{'L1' if p['l1_ratio'] == 1 else 'L2'} penalty, C = {p['C']:g}"
        if m == "svm":
            return f"{p['kernel'].upper()} kernel, C = {p['C']:g}, gamma = {p['gamma']}"
        if m == "random_forest":
            return f"{p['n_estimators']} trees, full depth, {p['max_features']} features/split"
        return f"{p['n_estimators']} trees, depth {p['max_depth']}, learning rate {p['learning_rate']:g}"

    def contribs(example, n=4):
        out = []
        for part in [s.strip() for s in loc.loc[example, "top_contributions"].split(";")][:n]:
            lv, shap = part.rsplit(": ", 1)
            out.append({"label": lv, "shap": float(shap)})
        return out

    data = {
        "student": {"name": "Vedank Singh", "reg": "240962043", "program": "B.Tech. CSE (AI & ML), V Semester, Section B",
                    "guides": "Dr. Narendra V G (Professor) and Ms Vibha (Assistant Professor - Senior Scale)",
                    "school": "School of Computer Engineering, Manipal Institute of Technology",
                    "year": "Academic Year 2026–27"},
        "gbd": {"adults_millions": 788, "prevalence_pct": 14.2},
        "data": {"records": R["clean"]["n_records_clean"], "ckd": R["clean"]["class_counts"]["ckd"],
                 "nonckd": R["clean"]["class_counts"]["notckd"], "missing_pct": round(R["missing_pct_cells"], 1),
                 "records_missing": R["clean"]["records_with_any_missing"],
                 "line_repairs": len(R["clean"]["line_repairs"]), "whitespace": R["whitespace_cells"],
                 "implausible": len(R["clean"]["implausible_values_set_missing"]),
                 "train": R["split"]["n_train"], "test": R["split"]["n_test"],
                 "test_ckd": R["split"]["class_counts"]["test"]["ckd"],
                 "test_non": R["split"]["class_counts"]["test"]["notckd"],
                 "sc_ckd": median_iqr(R, "sc", "CKD").split(" ")[0], "sc_non": median_iqr(R, "sc", "Non-CKD").split(" ")[0],
                 "hb_ckd": median_iqr(R, "hemo", "CKD").split(" ")[0], "hb_non": median_iqr(R, "hemo", "Non-CKD").split(" ")[0]},
        "models": [{"key": m, "name": DISPLAY_NAMES[m], "selected": m == W, "params": params(m),
                    "cv_auc": cv.loc[m, "mean_roc_auc"], "cv_auc_sd": cv.loc[m, "std_roc_auc"],
                    "cv_recall": cv.loc[m, "mean_recall"], "cv_recall_sd": cv.loc[m, "std_recall"],
                    "cv_f1": cv.loc[m, "mean_f1"], "cv_f1_sd": cv.loc[m, "std_f1"],
                    "test_acc": test.loc[m, "accuracy"], "test_recall": test.loc[m, "recall"],
                    "test_spec": test.loc[m, "specificity"], "test_auc": test.loc[m, "roc_auc"],
                    "fp": int(test.loc[m, "fp"]), "fn": int(test.loc[m, "fn"]),
                    "abl_recall": R["ablation"][m]["recall"], "abl_f1": R["ablation"][m]["f1"],
                    "abl_auc": R["ablation"][m]["roc_auc"]} for m in MODEL_NAMES],
        "winner": R["winner_name"],
        "selection_reason": R["selection"]["reason"],
        "rf_test": {"acc_lo": tw.accuracy_exact_lo, "acc_hi": tw.accuracy_exact_hi,
                    "rec_lo": tw.recall_exact_lo, "spec_lo": tw.specificity_exact_lo},
        "near_miss": {"id": R["near_miss_id"], "rf": R["near_miss_p"]["random_forest"], "xgb": R["near_miss_p"]["xgboost"]},
        "checks": {"probe_auc": R["probe_auc"], "perm_auc": R["perm_auc"], "audit_passed": R["audit_passed"],
                   "audit_total": R["audit_total"]},
        "shap": {"top": [{"label": r.label, "mean_abs": r.mean_abs_shap_test, "share": r.share_of_total_pct}
                         for r in imp.head(6).itertuples()],
                 "top5_share": R["shap_top5_share"], "base": shapc["base_value"],
                 "additivity": shapc["max_additivity_error"], "spearman": shapc["spearman_rank_corr_test_vs_train_importance"],
                 "local_ckd": {"id": int(loc.loc["typical_ckd", "record_id"]), "p": float(loc.loc["typical_ckd", "p_ckd"]),
                               "top": contribs("typical_ckd")},
                 "local_border": {"id": int(loc.loc["borderline", "record_id"]), "p": float(loc.loc["borderline", "p_ckd"]),
                                  "top": contribs("borderline")}},
        "runtime": {"pipeline_s": round(R["pipeline_seconds"]), "predict_ms": round(R["latency"]["predict_ms_median"]),
                    "explain_ms": round(R["latency"]["explain_ms_median"])},
        "fig": {k: str(ROOT / "results" / "figures" / v) for k, v in {
            "workflow": "workflow_diagram.png", "missing": "eda_missing_per_record_train.png",
            "scores": "test_score_distributions.png", "beeswarm": "shap_beeswarm.png",
            "wf_ckd": f"shap_waterfall_typical_ckd_{int(loc.loc['typical_ckd', 'record_id'])}.png",
            "wf_border": f"shap_waterfall_borderline_{int(loc.loc['borderline', 'record_id'])}.png"}.items()},
    }
    data["fig"].update({k: str(ROOT / "presentation" / "assets" / f"{k}.jpg")
                        for k in ("app_overview", "app_prediction", "app_prediction_top", "app_explanation")})
    out = ROOT / "presentation" / "slide_data.json"
    out.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
