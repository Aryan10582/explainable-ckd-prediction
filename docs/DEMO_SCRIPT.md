# Live demo script (about 5 minutes)

All expected values below were produced by the saved model (`models/selected_model.joblib`). The example records are
**real records from the held-out test partition** of the UCI dataset; the app labels them as such. The "what-if" edit
in step 6 is a synthetic modification for illustration only, not evaluation evidence.

## Before the demo

1. Open a terminal in the project folder and start the app:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
   ```

   (Linux/WSL: `python -m streamlit run app/streamlit_app.py` with the environment activated.)
2. Open http://localhost:8501 and visit **Predict and explain** once, so the model and explainer are loaded and cached.
3. If the browser shows a stale page after code changes, press **R** (rerun) or refresh.

## Steps and what to say

| # | Action | What appears | What to say |
|---|---|---|---|
| 1 | **Overview** page | Disclaimer, 400 records, Random Forest, CV ROC-AUC 0.9994, test accuracy 1.000, workflow diagram | "The app only loads the saved pipeline. It never retrains." Point to the disclaimer. |
| 2 | **Predict and explain** → *Load an example* → "Dataset record 26 – held-out test set, true class CKD" | Form fills (creatinine 2.4 mg/dL, haemoglobin 11.6 g/dL, red blood cells not recorded) | "This is a real patient record from the untouched test set." |
| 3 | Press **Predict and explain** | P(CKD) = **0.993**, predicted **CKD**, 1 of 24 fields not recorded; blue box: red blood cells imputed as *normal* | "The missing field is filled with the training mode and shown, not hidden." |
| 4 | Scroll to the waterfall | Baseline 0.667; serum creatinine +0.07, specific gravity +0.06, haemoglobin +0.05, hypertension +0.04 … ; check line "0.667 + 0.326 = 0.993 … matches the displayed probability" | "Read bottom to top: the average prediction, plus each feature's push, gives exactly this patient's probability." |
| 5 | Load "Dataset record 324 – true class Non-CKD (least certain prediction)" and predict | P(CKD) = **0.397**, Non-CKD, 4 fields not recorded | "Correct, but close to the threshold. Its haemoglobin was missing; the imputed 12.6 g/dL pushes it towards CKD." |
| 6 | *What-if (synthetic):* load record 26 again, change serum creatinine to **0.8**, predict | P(CKD) = **0.940**, still CKD | "One normal value does not flip the prediction: creatinine now pushes down (−0.03), but specific gravity, haemoglobin, packed cell volume, hypertension and diabetes still point to CKD. This is a model sensitivity check, not evidence." |
| 7 | Clear the form (*Blank form*) and predict | P(CKD) = **0.803**, CKD, 24 of 24 imputed | "With nothing entered, the model sees the training medians, which come mostly from CKD records. This is why missing inputs must be shown, and why this is not a screening tool." |
| 8 | Enter potassium **47** | Red error: outside plausible range 1–15 | "Physiologically impossible values are rejected, as in the cleaning step." |
| 9 | **Global importance** page | Bar chart and beeswarm; creatinine, specific gravity, haemoglobin, albumin, packed cell volume | "These agree with clinical markers, but SHAP explains the model, not the disease." |
| 10 | **Model comparison** page | CV and test tables, exact CIs, SMOTE-NC ablation, missingness probe 0.864, leakage audit 11/11 | "Selection used training CV only. The checks show what the numbers can and cannot support." |

## If something goes wrong

* *"Saved experiment outputs are missing"* → run `scripts\run_pipeline.py` once.
* Port 8501 busy → add `--server.port 8502`.
* No internet is needed for the demo.
