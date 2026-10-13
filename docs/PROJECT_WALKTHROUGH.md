# Project walkthrough

How the project fits together, module by module, and what happens to one patient record from the form to the
explanation. File references are relative to the project folder.

## 1. One implementation, many users

All logic lives in the package `src/ckd_xai/`. The pipeline script, the notebook, the Streamlit app and the tests import
the same functions, so there is only one version of the cleaning, preprocessing, modelling and explanation code.
`config.yaml` holds every setting that can change a result (seed, split size, folds, grids, selection rule, SHAP sizes).

## 2. The modules

| Module | Responsibility | Key points to remember |
|---|---|---|
| `schema.py` | Single source of truth for the 24 features | Types (11 continuous, 3 ordinal, 10 binary), allowed categories, documented units, plausibility limits, target mapping CKD = 1. `validate_record()` checks one patient record (missing, unknown category, impossible value). |
| `data.py` | Getting and cleaning the data | `acquire()` downloads the official zip, extracts the ARFF from the inner RAR, fetches UCI's CSV and metadata, records SHA-256 hashes in `data/raw/SOURCE.json`, and supports a local file. `parse_arff_text()` repairs the 3 malformed lines and logs them. `clean()` strips tabs/spaces, turns "?" into missing, converts numbers, checks categories and sets 3 impossible values to missing; no record is dropped. `make_or_verify_split()` creates the stratified 80/20 split once and refuses to continue if a later run would produce a different test set. |
| `preprocessing.py` | The leak-free pipeline | `build_pipeline()` = `prep` (ColumnTransformer) → `resample` (SMOTE-NC or passthrough) → `clf`. Built with imbalanced-learn's `Pipeline`, which applies the resampler only in `fit`. `to_original_units()` maps standardised values back to clinical units for plots. |
| `models.py` | The four classifiers | LR (liblinear, L1/L2), SVM inside `CalibratedClassifierCV` (sigmoid, 5-fold, training data only), Random Forest, XGBoost (`binary:logistic`, `hist`, CPU). |
| `train.py` | Training-set experiments | Grid search with stratified 5-fold CV for each model; all metrics from the CKD probability at 0.5; `select_model()` applies the predeclared rule; also the SMOTE-NC ablation, the missingness probe, out-of-fold predictions and the shuffled-label check. Writes `models/selection.json` **before** any test record is read. |
| `evaluate.py` | Final test evaluation | Loads the four frozen pipelines, computes metrics, exact Clopper–Pearson and bootstrap intervals, confusion matrices, ROC/PR curves, per-record predictions, and runs the 11-point leakage audit. |
| `explain.py` | SHAP | Builds the right explainer for the selected model (here interventional Tree SHAP with 100 training records as background), checks that baseline + contributions = model output for every record, and draws the global and local plots. Contains the float32 threshold fix for scikit-learn forests. |
| `inference.py` | One record at a time | `CKDPredictor` loads the saved model and SHAP background, validates a record, predicts, lists imputed fields, warns about values outside the training range and returns an explanation that reconstructs the displayed probability. |
| `eda.py`, `diagram.py`, `plotting.py` | Figures | Whole-dataset summaries vs training-only class comparisons are kept apart. |
| `report_data.py` | Numbers for the paper and slides | Reads the saved outputs so written documents always match `results/`. |

## 3. The experiment, in order (`scripts/run_pipeline.py`)

1. **data** – acquire, clean, validate (400 records, 250/150, 24 features), cross-check with UCI's CSV (0 differences).
2. Split – 320 training / 80 test records, stratified, seed 42, checksum of the test IDs saved.
3. **eda** – figures and tables; class-conditioned ones use the training set only.
4. **train** – for each model: `GridSearchCV(pipeline, grid, cv=StratifiedKFold(5))`. Inside every fold the pipeline
   fits imputation and scaling on the 256 fold-training records, SMOTE-NC adds synthetic non-CKD records to them, the
   classifier is trained, and the 64 validation records are only transformed and predicted. Then the selection rule is
   applied and the best pipeline of each model is refitted on all 320 records and saved.
5. **evaluate** – the frozen pipelines predict the 80 test records once.
6. **explain** – SHAP for the selected Random Forest.
7. **latency** – timing of single-record prediction and explanation.

## 4. What happens to a new patient record (the app)

1. The form collects up to 24 values; blanks are allowed. Selectboxes only offer allowed categories; number boxes stop
   at the plausibility limits.
2. `schema.validate_record()` converts blanks to missing, rejects impossible values and unknown categories.
3. The record becomes a one-row table in the original column order.
4. The saved pipeline's `prep` step applies the **training** medians/modes, scaling statistics and 0/1 encoding.
   (SMOTE-NC is not applied at prediction time.)
5. The Random Forest returns P(CKD); the label is CKD if P ≥ 0.5.
6. SHAP computes one contribution per feature against the 100-record training background; the app shows the
   waterfall and checks that 0.667 + sum of contributions equals the displayed probability.
7. Imputed fields and out-of-training-range values are reported to the user.

## 5. Where to look when asked "how do you know…?"

| Claim | Evidence |
|---|---|
| No test data influenced the model | `models/selection.json` (`test_set_used: false`, timestamp), `results/tables/leakage_audit.csv` |
| Resampling only in fit | `tests/test_pipeline.py::test_smotenc_resamples_only_during_fit` |
| Imputation used training data only | audit check 5; `tests/test_pipeline.py::test_imputation_and_scaling_use_only_the_fitting_data` |
| Saved model reproduces results | `tests/test_inference.py`, notebook section 5 |
| SHAP adds up | `results/logs/shap_checks.json`, `tests/test_explain.py` |
| Data are genuine and unchanged | `data/raw/SOURCE.json`, `data/processed/cleaning_summary.json` |
| Results are reproducible | rerun gives identical `cv_model_comparison.csv` and `test_predictions.csv` |
