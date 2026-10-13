import pandas as pd
import streamlit as st

from app_common import captions, cleaning_summary, disclaimer, figure, require_artifacts, table

require_artifacts()
st.title("Data and limitations")
disclaimer()

cs = cleaning_summary()
st.subheader("Dataset")
st.markdown(
    "UCI Machine Learning Repository, *Chronic Kidney Disease* (id 336, doi:10.24432/C5G020), donated in 2015 "
    "by L. Rubini, P. Soundarapandian and P. Eswaran. According to the source, the records were collected over "
    "about two months at a hospital in Tamil Nadu, India. 400 records, 24 inputs (11 numeric, 3 ordinal, "
    "10 binary), target *ckd* / *notckd*."
)
with st.container(horizontal=True):
    st.metric("Records", cs["n_records_clean"], border=True)
    st.metric("CKD / non-CKD", f"{cs['class_counts']['ckd']} / {cs['class_counts']['notckd']}", border=True)
    st.metric("Records with a missing value", cs["records_with_any_missing"], border=True)
    st.metric("Missing cells", cs["missing_cells_total"], border=True)

st.markdown(
    f"**Cleaning (no record removed):** {len(cs['line_repairs'])} malformed lines repaired, stray tabs/spaces "
    f"removed in {sum(cs['whitespace_fixed_cells'].values())} cells, '?' converted to missing, and "
    f"{len(cs['implausible_values_set_missing'])} physiologically impossible values set to missing "
    "(sodium 4.5 mEq/L; potassium 39 and 47 mEq/L). The cleaned ARFF agrees cell-for-cell with UCI's own CSV."
)
st.image(str(figure("eda_missingness.png")), width=560, caption=captions().get("eda_missingness.png"),
         alt="Percentage of missing values per feature")

st.subheader("Feature dictionary")
fd = table("feature_dictionary.csv")
st.dataframe(fd, hide_index=True, alt="Feature names, types, units and allowed values")

st.subheader("Limitations")
st.markdown(
    "- **Small, single-source data.** 400 records from one hospital over about two months; results may not "
    "transfer to other populations, laboratories or time periods.\n"
    "- **Small test set.** 80 records (50 CKD, 30 non-CKD); a perfect score still leaves wide confidence intervals.\n"
    "- **Very separable classes.** Several features (e.g. serum creatinine, haemoglobin, specific gravity) separate "
    "the classes almost completely, so this dataset says little about hard, early-stage cases.\n"
    "- **Informative missingness.** CKD records have far more missing values than non-CKD records; the model may "
    "partly learn *which tests were ordered* rather than physiology.\n"
    "- **No eGFR, no time dimension.** CKD is defined clinically by kidney function and damage persisting for more "
    "than three months; this dataset has neither eGFR nor repeated measurements.\n"
    "- **Uncalibrated probabilities.** Training on SMOTE-NC-balanced data shifts probabilities; they are not "
    "individual risks.\n"
    "- **SHAP is not causal.** Attributions explain the model, not the disease, and correlated features can share "
    "or swap credit.\n"
    "- **No external validation or clinical evaluation** has been performed. This is an educational prototype."
)
