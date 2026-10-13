"""Headless tests of the Streamlit demo (streamlit.testing.v1.AppTest)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
PAGES = ["app_pages/overview.py", "app_pages/predict.py", "app_pages/global_importance.py",
         "app_pages/model_comparison.py", "app_pages/data_limitations.py"]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_error(artifacts_ready, page):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.switch_page(page).run()
    assert not at.exception
    assert any("not a diagnostic tool" in w.value for w in at.warning)   # disclaimer on every page


def test_prediction_and_explanation_workflow(artifacts_ready):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.switch_page("app_pages/predict.py").run()
    sample = [o for o in at.selectbox(key="sample_choice").options if "true class CKD" in o][0]
    at.selectbox(key="sample_choice").select(sample).run()
    at.button[0].click().run()                     # the form's submit button
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Model estimate of P(CKD)" in labels
    p = float(at.metric[labels.index("Model estimate of P(CKD)")].value)
    assert 0.5 <= p <= 1.0
    assert any("matches the displayed probability" in m.value for m in at.markdown)


def test_blank_form_imputes_and_warns(artifacts_ready):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.switch_page("app_pages/predict.py").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Not recorded, so filled with training values" in i.value for i in at.info)
