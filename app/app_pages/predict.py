import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from app_common import NOT_RECORDED, disclaimer, get_predictor, require_artifacts, sample_records  # noqa: E402
from ckd_xai import schema  # noqa: E402
from ckd_xai.explain import waterfall  # noqa: E402
from ckd_xai.plotting import apply_style  # noqa: E402

require_artifacts()
st.title("Predict and explain")
disclaimer()

predictor = get_predictor()
ranges = predictor.ranges
BLANK = "Blank form (every field not recorded)"


def key(name: str) -> str:
    return f"in_{name}"


def reset_value(f: schema.Feature):
    return None if f.kind == "continuous" else NOT_RECORDED


for f in schema.FEATURES:                       # initialise widget state once
    st.session_state.setdefault(key(f.name), reset_value(f))


def load_sample() -> None:
    choice = st.session_state["sample_choice"]
    rec = {} if choice == BLANK else sample_records()[choice]
    for f in schema.FEATURES:
        v = rec.get(f.name)
        if v is None:
            st.session_state[key(f.name)] = reset_value(f)
        elif f.kind == "continuous":
            st.session_state[key(f.name)] = float(v)
        elif f.kind == "ordinal":
            st.session_state[key(f.name)] = float(v)
        else:
            st.session_state[key(f.name)] = str(v)
    st.session_state.pop("result", None)


st.selectbox(
    "Load an example",
    [BLANK] + list(sample_records()),
    key="sample_choice",
    on_change=load_sample,
    help="Examples are real records from the held-out test partition of the UCI dataset. The model is "
         "frozen, so viewing them here does not affect any reported result.",
)


def help_text(f: schema.Feature) -> str:
    parts = [f.description + "."]
    if f.unit_note:
        parts.append(f.unit_note)
    r = ranges.get(f.name)
    if r:
        parts.append(f"Training range: {r['min']:g} to {r['max']:g}.")
    parts.append("Leave empty if not recorded; the model then uses the training median/mode.")
    return " ".join(parts)


def widget(f: schema.Feature) -> None:
    if f.kind == "continuous":
        lo, hi = f.plausible
        st.number_input(f.display, min_value=float(lo), max_value=float(hi), step=float(f.step),
                        key=key(f.name), placeholder=NOT_RECORDED, help=help_text(f),
                        format="%.1f" if f.step < 1 else "%.0f")
    elif f.kind == "ordinal":
        st.selectbox(f.display, [NOT_RECORDED] + [float(v) for v in f.levels], key=key(f.name),
                     format_func=lambda v: v if isinstance(v, str) else f"{v:g}", help=help_text(f))
    else:
        st.selectbox(f.label, [NOT_RECORDED] + list(f.levels), key=key(f.name), help=help_text(f))


GROUPS = {
    "History and examination": ["age", "bp", "htn", "dm", "cad", "appet", "pe", "ane"],
    "Urine tests": ["sg", "al", "su", "rbc", "pc", "pcc", "ba"],
    "Blood tests": ["bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"],
}

with st.form("patient_form"):
    for title, names in GROUPS.items():
        st.subheader(title)
        cols = st.columns(3)
        for i, n in enumerate(names):
            with cols[i % 3]:
                widget(schema.FEATURE_BY_NAME[n])
    submitted = st.form_submit_button("Predict and explain", type="primary", icon=":material/analytics:")

if submitted:
    record = {}
    for f in schema.FEATURES:
        v = st.session_state[key(f.name)]
        record[f.name] = None if v is None or v == NOT_RECORDED else v
    st.session_state["result"] = predictor.predict(record)

result = st.session_state.get("result")
if result is None:
    st.info("Fill in the form (or load an example) and press **Predict and explain**.", icon=":material/edit:")
    st.stop()

if not result["ok"]:
    for e in result["errors"]:
        st.error(e, icon=":material/error:")
    st.stop()

expl = predictor.explain(result["record"])

with st.container(border=True):
    st.subheader("Model output")
    with st.container(horizontal=True):
        st.metric("Model estimate of P(CKD)", f"{result['p_ckd']:.3f}", border=True,
                  help="Random Forest probability for the CKD class. A model estimate, not a clinical risk.")
        st.metric(f"Predicted class (threshold {result['threshold']:g})", result["class_name"], border=True)
        st.metric("Fields not recorded", f"{len(result['missing'])} of {len(schema.FEATURES)}", border=True)
    st.caption("Probabilities from a model trained on SMOTE-NC-balanced data are not calibrated to any real "
               "population's CKD prevalence. Treat them as a ranking score, not as a risk.")

if result["imputed_values"]:
    items = ", ".join(f"{k} = {v}" for k, v in result["imputed_values"].items())
    st.info(f"**Not recorded, so filled with training values:** {items}. Imputed values are guesses, not "
            "measurements; the explanation below marks them.", icon=":material/edit_note:")
for w in result["warnings"]:
    st.warning(w, icon=":material/warning:")

st.subheader("Why this prediction? (local SHAP explanation)")
apply_style()
fig = waterfall(expl["values"], expl["base_value"], expl["text"], expl["output_name"],
                f"P(CKD) = {expl['p_model']:.3f}, predicted {result['class_name']}", width=4.6)
st.pyplot(fig, alt="SHAP waterfall chart of feature contributions for this record", width=820)
plt.close(fig)

check = ":material/check_circle:" if expl["consistent"] else ":material/error:"
st.markdown(
    f"{check} Baseline **{expl['base_value']:.3f}** (average model output over 100 training records) "
    f"+ sum of contributions **{expl['values'].sum():+.3f}** = **{expl['reconstructed_output']:.3f}**, "
    f"which matches the displayed probability **{expl['p_model']:.3f}**."
)
st.caption("Red bars push this record's output towards CKD, blue bars towards non-CKD. SHAP describes how "
           "this model used the inputs; it is not evidence that a feature causes kidney disease.")

with st.expander("All 24 contributions", icon=":material/table_chart:"):
    tbl = expl["table"][["label", "value", "shap"]].rename(
        columns={"label": "Feature", "value": "Value used by the model", "shap": "SHAP contribution"})
    st.dataframe(tbl, hide_index=True, alt="Table of SHAP contributions for all 24 features",
                 column_config={"SHAP contribution": st.column_config.NumberColumn(format="%+.4f")})
