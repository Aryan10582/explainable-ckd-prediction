import pandas as pd
import streamlit as st

from app_common import captions, disclaimer, figure, require_artifacts, selection, table

require_artifacts()
st.title("Model comparison")
disclaimer()

sel = selection()
caps = captions()
cv = table("cv_model_comparison.csv")
test = table("test_metrics.csv")
metrics = ["roc_auc", "accuracy", "precision", "recall", "f1"]
labels = {"roc_auc": "ROC-AUC", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
          "f1": "F1-score"}

st.subheader("1. Training-set cross-validation (used for selection)")
cv_view = pd.DataFrame({"Model": cv["display_name"]})
for m in metrics:
    cv_view[labels[m]] = [f"{a:.4f} ± {s:.4f}" for a, s in zip(cv[f"mean_{m}"], cv[f"std_{m}"])]
st.dataframe(cv_view, hide_index=True, alt="Cross-validation results, mean plus or minus standard deviation")
st.success(f"**Selected: {sel['display_name']}.** {sel['reason']}", icon=":material/check_circle:")
st.caption("Selection rule (fixed before the experiment): highest mean CV ROC-AUC; models within "
           f"{sel['rule']['auc_tie_tolerance']} are tied and ranked by mean CV recall, then F1. "
           "The test set played no part in this choice.")

st.subheader("2. Frozen models on the untouched test set (n = 80)")
tv = pd.DataFrame({"Model": test["display_name"]})
for m in metrics:
    tv[labels[m]] = test[m].round(4)
tv["Accuracy 95% CI (exact)"] = [f"{a:.3f}-{b:.3f}" for a, b in zip(test.accuracy_exact_lo, test.accuracy_exact_hi)]
tv["FP"] = test["fp"]
tv["FN"] = test["fn"]
st.dataframe(tv, hide_index=True, alt="Test-set metrics of the four frozen models")
st.caption("With only 80 test records, a perfect score is compatible with a true accuracy as low as the lower "
           "confidence bound shown. A false negative means a CKD case labelled non-CKD (a missed case); a false "
           "positive means a non-CKD case flagged as CKD.")

c1, c2 = st.columns(2)
with c1:
    st.image(str(figure("test_confusion_matrices.png")), caption=caps.get("test_confusion_matrices.png"),
             alt="Confusion matrices of the four models on the test set")
with c2:
    st.image(str(figure("test_score_distributions.png")), caption=caps.get("test_score_distributions.png"),
             alt="Predicted CKD probabilities on the test set by true class")
c3, c4 = st.columns(2)
with c3:
    st.image(str(figure("cv_roc_curves.png")), caption=caps.get("cv_roc_curves.png"),
             alt="Out-of-fold ROC curves from training-set cross-validation")
with c4:
    st.image(str(figure("test_pr_curves.png")), caption=caps.get("test_pr_curves.png"),
             alt="Precision-recall curves on the test set")

st.subheader("3. Checks")
abl = table("cv_resampling_ablation.csv")
abl_view = abl.assign(Model=abl["model"].map(dict(zip(cv["model"], cv["display_name"]))))[
    ["Model", "resampling", "mean_roc_auc", "mean_recall", "mean_f1", "mean_specificity"]].round(4)
st.markdown("**Does SMOTE-NC help?** Same folds and hyper-parameters, with and without resampling:")
st.dataframe(abl_view, hide_index=True, alt="SMOTE-NC versus no resampling in cross-validation")
probe = table("cv_missingness_probe.csv").iloc[0]
st.markdown(f"**Missingness shortcut probe:** a logistic regression that sees only *which* values are missing "
            f"reaches CV ROC-AUC {probe['mean_roc_auc']:.3f}, so missingness itself carries class information.")
audit = table("leakage_audit.csv")
st.markdown(f"**Leakage audit:** {int(audit.passed.sum())} of {len(audit)} checks passed.")
st.dataframe(audit, hide_index=True, alt="Leakage audit checks")
