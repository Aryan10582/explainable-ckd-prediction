# Explainable Machine Learning for Chronic Kidney Disease Prediction Using SHAP-Based Feature Attribution

Undergraduate mini-project (Machine Learning, CSE 3125) by **Vedank Singh** (Reg. No. 240962043), B.Tech. CSE (AI & ML),
School of Computer Engineering, Manipal Institute of Technology. Guides: Dr. Narendra V G and Ms Vibha. Academic Year 2026–27.

A reproducible pipeline that compares Logistic Regression, SVM, Random Forest and XGBoost for predicting chronic
kidney disease (CKD) on the UCI CKD dataset, selects one model with a rule fixed before the experiment, evaluates it once
on an untouched test set, and explains its predictions with SHAP. A local Streamlit app demonstrates the saved model.

> **Educational research prototype, not a diagnostic tool.** Probabilities are model estimates from 320 training
> records of a single-centre dataset, not clinically validated individual risks. No treatment advice is given.

## Results at a glance

All numbers come from `results/` (regenerate with `scripts/run_pipeline.py`; a re-run reproduces them byte for byte).

| Model | Mean 5-fold CV ROC-AUC (train) | CV recall | Test accuracy (n = 80) | Test FN / FP |
|---|---|---|---|---|
| Logistic Regression | 0.9994 | 0.960 | 0.975 | 2 / 0 |
| SVM | 1.0000 | 0.985 | 1.000 | 0 / 0 |
| **Random Forest (selected)** | 0.9994 | **1.000** | 1.000 (exact 95% CI 0.955–1.000) | 0 / 0 |
| XGBoost | 0.9988 | 0.985 | 1.000 | 0 / 0 |

* Selection (predeclared): all four were within 0.005 of the best CV ROC-AUC, so the tie was broken by CV recall → Random Forest.
* SHAP (test set): serum creatinine, specific gravity, haemoglobin, albumin grade and packed cell volume carry 63.7% of the attribution.
* Honest checks: SMOTE-NC gave no consistent benefit; missing-value indicators alone reach CV ROC-AUC 0.864; shuffled labels give 0.505; leakage audit 11/11 passed.
* The classes are almost separable (no non-CKD training record has any abnormal finding), so the perfect scores do **not** show clinical readiness.

## Deliverables

| What | Where |
|---|---|
| IEEE paper (prescribed ProComm template): DOCX, PDF, Markdown | `reports/IEEE_paper_CKD_XAI.docx`, `.pdf`, `.md` |
| College front page (separate from the paper) | `reports/Front_Page_CKD_XAI.docx`, `.pdf` |
| Front page + paper in one PDF | `reports/Submission_FrontPage_and_IEEE_Paper.pdf` |
| Presentation (11 slides, speaker notes) | `presentation/CKD_XAI_presentation.pptx`, `.pdf` |
| Demo script / project walkthrough / viva Q&A | `docs/DEMO_SCRIPT.md`, `docs/PROJECT_WALKTHROUGH.md`, `docs/VIVA_QA.md` |
| Reference verification | `docs/REFERENCE_VERIFICATION.md` |
| Educational notebook (executed) | `notebooks/CKD_XAI_walkthrough.ipynb` |
| Tables, figures, logs | `results/tables/`, `results/figures/`, `results/logs/` |
| Saved models and metadata | `models/` |
| Progress log | `PROJECT_STATUS.md` |

## Project structure

```
config.yaml                 all settings, grids and the predeclared selection rule
src/ckd_xai/                the one implementation used by scripts, notebook, app and tests
  schema.py                 feature types, allowed values, units, plausibility limits, target mapping
  data.py                   download + checksums, ARFF parsing, cleaning, validation, locked split
  preprocessing.py          imputation/scaling/encoding + SMOTE-NC + classifier pipelines
  models.py                 the four classifiers
  train.py                  grid search CV, predeclared selection, SMOTE-NC ablation, probes
  evaluate.py               test metrics, exact/bootstrap CIs, plots, leakage audit
  explain.py                SHAP explainer, additivity checks, global/local plots
  inference.py              single-record validation, prediction and explanation (used by the app)
  eda.py, diagram.py, plotting.py, report_data.py, utils.py, config.py
scripts/                    download_data.py, run_pipeline.py, build_notebook.py, verify_references.py
app/                        Streamlit app (streamlit_app.py + app_pages/)
tests/                      45 pytest tests (data, schema, leakage, saved models, SHAP, app)
notebooks/                  CKD_XAI_walkthrough.ipynb
data/raw/                   cached official files + SOURCE.json (SHA-256)
data/processed/             ckd_clean.csv, cleaning_summary.json, split.csv, split_manifest.json
models/                     fitted pipelines, selected_model.joblib + metadata, SHAP background
results/                    tables/, figures/, logs/
reports/                    paper + front-page builders, templates/, outputs
presentation/               slide builder (Node.js + pptxgenjs), assets/, outputs
docs/                       demo script, walkthrough, viva Q&A, reference checks
```

## Setup

Python 3.13 was used (3.11+ should work). Use paths relative to the project folder.

**Windows PowerShell**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

**Linux / WSL**

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

(The commands below use the Windows form `.\.venv\Scripts\python.exe`; on Linux/WSL write `python` after activating
the environment and use `/` in paths.)

## Data

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
```

Downloads the official UCI archive (dataset 336, doi:10.24432/C5G020), extracts the original ARFF, also fetches UCI's
`data.csv` for an independent cross-check, caches everything in `data/raw/` with SHA-256 checksums, and writes the
cleaned table. Later runs work offline from the cache.

* The archive holds a RAR file. Windows 10+/macOS `tar` can read it; on Linux install `libarchive-tools` (bsdtar),
  `unrar` or `7z`. Without any of them the script falls back to UCI's `data.csv` automatically (identical after cleaning).
* No internet: download the zip from the UCI page by hand and run
  `scripts\download_data.py --local-file path\to\chronic+kidney+disease.zip` (an `.arff` or the UCI `data.csv` also works).

## Run the experiment

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

Stages: `data`, `eda`, `train`, `evaluate`, `explain`, `latency` (33–45 s in total on the development laptop,
AMD Ryzen 7 7840HS; timings vary with load, results do not).
To regenerate figures and explanations from the saved models **without retraining**:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --stages eda evaluate explain latency
```

## Demo app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Open http://localhost:8501. The app only loads saved artifacts; run the pipeline first. See `docs/DEMO_SCRIPT.md`.

## Tests and notebook

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\build_notebook.py
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks\CKD_XAI_walkthrough.ipynb
```

## Rebuild the paper, front page and slides (optional)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-docs.txt
.\.venv\Scripts\python.exe reports\build_paper.py          # DOCX + Markdown; PDF via Microsoft Word
.\.venv\Scripts\python.exe reports\build_front_page.py     # needs Microsoft Word for the PDF
.\.venv\Scripts\python.exe presentation\export_slide_data.py
cd presentation; npm install; node build_slides.js; cd ..
.\.venv\Scripts\python.exe presentation\export_pdf.py      # needs Microsoft PowerPoint
```

Without Word, use `reports\build_paper.py --no-pdf` and export the DOCX to PDF by hand. `scripts\verify_references.py`
re-checks every DOI online.

## Main methodological decisions

* **Primary class and threshold:** CKD = 1 (positive) everywhere; threshold 0.5, not tuned.
* **Split:** 80/20 stratified, seed 42, created once and locked with a checksum before any tuning.
* **Cleaning:** explicit rules for the known ARFF defects; three physiologically impossible values set to missing by
  limits fixed in `schema.py`; no record removed; no duplicate records exist.
* **Preprocessing (inside each training fold):** median imputation + standardisation (numeric), mode imputation +
  standardisation (ordinal grades), 0/1 encoding with unknown → missing → mode (binary). One model column per feature.
* **Class imbalance:** SMOTE-NC (the mixed-data variant of SMOTE) inside the imbalanced-learn pipeline, so it runs only
  during fitting. Ordinal and binary columns are treated as categorical so synthetic records keep valid levels.
* **Selection:** predeclared rule in `config.yaml` (mean CV ROC-AUC; ties within 0.005 broken by recall, then F1).
  The synopsis suggested choosing by test score; this was changed so the test estimate stays unbiased.
* **SVM probabilities:** sigmoid calibration fitted inside the training data (`CalibratedClassifierCV`, scikit-learn 1.9).
* **SHAP:** interventional Tree SHAP for the Random Forest, background of 100 real training records, output =
  probability of CKD. Additivity is checked for every record (max error 1.8e-8). A float32 threshold-routing mismatch
  between SHAP and scikit-learn was found and fixed with a prediction-preserving threshold rounding (see `explain.py`).
* **Uncertainty:** exact Clopper-Pearson CIs for proportions; stratified bootstrap (2000) for ROC-AUC/F1.

## Limitations

400 single-centre records with an apparently healthy control group; 80 test records; informative missingness; no eGFR
or repeated measurements; probabilities not calibrated to any population; SHAP is descriptive, not causal; no external
validation. The project demonstrates a careful method, not a clinically usable tool.
