import streamlit as st

from app_common import captions, disclaimer, figure, json_file, require_artifacts, selection, table
from ckd_xai.config import LOGS_DIR

require_artifacts()
st.title("Explainable machine learning for chronic kidney disease prediction")
st.caption("Undergraduate mini-project - SHAP-based feature attribution on the UCI Chronic Kidney Disease dataset")
disclaimer()

sel = selection()
test = table("test_metrics.csv").set_index("model")
cv = table("cv_model_comparison.csv").set_index("model")
chosen = sel["selected_model"]
row = test.loc[chosen]

st.markdown(
    "This app demonstrates the project pipeline: four classifiers (Logistic Regression, SVM, Random Forest, "
    "XGBoost) were tuned with 5-fold cross-validation on 320 training records, one model was chosen by a "
    "rule fixed before the experiment, and the frozen model was evaluated once on 80 untouched test records. "
    "SHAP then explains which inputs moved each prediction."
)

with st.container(horizontal=True):
    st.metric("Records (CKD / non-CKD)", "400 (250 / 150)", border=True)
    st.metric("Selected model", sel["display_name"], border=True)
    st.metric("Mean CV ROC-AUC (training)", f"{cv.loc[chosen, 'mean_roc_auc']:.4f}", border=True)
    st.metric("Test accuracy (n = 80)", f"{row['accuracy']:.3f}", border=True,
              help=f"95% exact CI {row['accuracy_exact_lo']:.3f}-{row['accuracy_exact_hi']:.3f}")

st.caption(f"Why {sel['display_name']}? {sel['reason']}")

with st.container(border=True):
    st.subheader("How a record flows through the system")
    wf = figure("workflow_diagram.png")
    if wf.exists():
        st.image(str(wf), width=520, alt="Workflow diagram from dataset to SHAP explanation and demo",
                 caption=captions().get("workflow_diagram.png"))
    st.markdown(
        "1. **Validate** the 24 inputs against the schema (allowed categories, plausible ranges).\n"
        "2. **Impute** missing values with the training median (numeric) or mode (categorical).\n"
        "3. **Scale** numeric and ordinal values with training means and standard deviations.\n"
        "4. **Predict** with the saved Random Forest (SMOTE-NC was used only while fitting).\n"
        "5. **Explain** with SHAP against a background of 100 real training records."
    )

latency = LOGS_DIR / "latency.json"
if latency.exists():
    lat = json_file(str(latency))
    st.caption(f"Measured on the development laptop: prediction {lat['predict_ms_median']:.0f} ms and explanation "
               f"{lat['explain_ms_median']:.0f} ms per record (median).")
