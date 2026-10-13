import altair as alt
import streamlit as st

from app_common import captions, disclaimer, figure, json_file, require_artifacts, table
from ckd_xai.config import LOGS_DIR

require_artifacts()
st.title("Global feature importance")
disclaimer()

checks = json_file(str(LOGS_DIR / "shap_checks.json"))
imp = table("shap_global_importance.csv")
caps = captions()

st.markdown(
    f"Mean absolute SHAP value of each feature over the {checks['n_explained_test']} held-out test records. "
    f"Units: **{checks['explained_output']}**. The ranking computed on the 320 training records agrees closely "
    f"(Spearman rank correlation {checks['spearman_rank_corr_test_vs_train_importance']:.3f})."
)

top = imp.head(15)
chart = (
    alt.Chart(top)
    .mark_bar(color="#2a78d6", cornerRadiusEnd=3)
    .encode(
        x=alt.X("mean_abs_shap_test:Q", title=f"Mean |SHAP value| ({checks['explained_output']})"),
        y=alt.Y("label:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("label:N", title="Feature"),
                 alt.Tooltip("mean_abs_shap_test:Q", title="Mean |SHAP|", format=".4f"),
                 alt.Tooltip("share_of_total_pct:Q", title="Share of total (%)", format=".1f"),
                 alt.Tooltip("rank_train:Q", title="Rank on training set")],
    )
    .properties(height=420)
)
st.altair_chart(chart, alt="Bar chart of mean absolute SHAP value for the 15 most important features")

left, right = st.columns(2)
with left:
    st.image(str(figure("shap_beeswarm.png")), caption=caps.get("shap_beeswarm.png"),
             alt="SHAP beeswarm summary plot")
with right:
    st.image(str(figure("shap_dependence.png")), caption=caps.get("shap_dependence.png"),
             alt="SHAP dependence plots for the four most important features")

st.info(
    "These rankings describe how the trained model uses the inputs. They show association within this "
    "dataset, not clinical causation, and they would change with a different model or population.",
    icon=":material/info:",
)
with st.expander("Explanation settings", icon=":material/settings:"):
    st.json({k: checks[k] for k in ("explainer", "explainer_mode", "explained_output", "background",
                                    "base_value", "max_additivity_error", "explainer_note")})
