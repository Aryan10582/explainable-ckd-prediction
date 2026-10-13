"""Streamlit demo: explainable CKD prediction (educational prototype).

Run from the project root:
    streamlit run app/streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="Explainable CKD prediction", page_icon=":material/nephrology:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/home:", default=True),
        st.Page("app_pages/predict.py", title="Predict and explain", icon=":material/stethoscope:"),
        st.Page("app_pages/global_importance.py", title="Global importance", icon=":material/insights:"),
        st.Page("app_pages/model_comparison.py", title="Model comparison", icon=":material/leaderboard:"),
        st.Page("app_pages/data_limitations.py", title="Data and limitations", icon=":material/dataset:"),
    ],
    position="top",
)
page.run()
