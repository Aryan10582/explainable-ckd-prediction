# Project status

Short log so work can resume if a session is interrupted. Last updated 14 October 2026. **Status: complete.**

## Inputs found
- Synopsis: `D:\3rd year\New folder (2)\Mini_Project_Synopsis_CKD.docx` (primary spec).
- College guidelines: `D:\3rd year\New folder\Mini-project-guidelines.pdf` -> report must use the
  IEEE ProComm 2024 `Full-Paper-template.docx` (downloaded from the official URL in the guidelines;
  identical SHA-256 to the local copy; ≤ 8 pages). Front page: `D:\3rd year\New folder\ML_Front_Page.docx`.

## Decisions fixed before running experiments
- Python 3.13 venv in `.venv` (3.14 skipped for package compatibility).
- Data: official UCI zip (ARFF inside a RAR) is primary; official UCI `data.csv` is fallback and cross-check
  (0 cells differ); `--local-file` for offline use.
- Test split 20 % stratified, seed 42, locked before tuning (checksum in `data/processed/split_manifest.json`).
- Selection rule predeclared in `config.yaml` (mean CV ROC-AUC, tie tolerance 0.005, then recall, F1, simplicity).
- SMOTE-NC inside training folds only; ordinal and binary columns treated as categorical during resampling.
- Impossible values (sodium 4.5, potassium 39 and 47) set to missing by a-priori plausibility limits.

## Results (final run; reproduced byte-for-byte, also from a fresh venv built from requirements.txt)
- CV ROC-AUC: LR 0.9994, SVM 1.0000, RF 0.9994, XGB 0.9988 -> all tied -> **Random Forest** (CV recall 1.000).
- Test (n = 80): SVM/RF/XGB 80/80 correct (RF exact 95% CI 0.955–1.000); LR 2 false negatives.
- SMOTE-NC ablation: no consistent benefit. Missingness-only probe CV AUC 0.864. Shuffled labels 0.505.
- Leakage audit 11/11. SHAP additivity 1.8e-8 (after a float32 threshold fix; 0.0065 without it).
- Top SHAP features: serum creatinine, specific gravity, haemoglobin, albumin grade, packed cell volume (63.7%).

## Progress
- [x] Environment created and library compatibility smoke-tested
- [x] Dataset downloaded, anomalies profiled, ARFF and CSV cross-checked
- [x] Core package (`src/ckd_xai`)
- [x] Full experiment run (`python scripts/run_pipeline.py`, 33–45 s)
- [x] Tests: 45 passed (`results/logs/pytest_report.txt`), also in a clean venv
- [x] Streamlit app verified live (all five pages, prediction + explanation) and with AppTest
- [x] Notebook executed from a clean kernel (14/14 cells, no errors)
- [x] References verified (`docs/REFERENCE_VERIFICATION.md`)
- [x] IEEE paper (DOCX/PDF/MD, 8 pages, prescribed template) + separate front page + combined PDF
- [x] Slides (PPTX/PDF, 11 slides, validated, real app screenshots)
- [x] Demo script, walkthrough, viva notes, README

## Open items for the student
- Roll number is not in the synopsis: fill it on the front page (`reports/Front_Page_CKD_XAI.docx`).
- The IEEE template asks for an author e-mail under the affiliation; none is in the synopsis, so it was left out.
- Run the college's Turnitin check (guideline: similarity ≤ 15 %); the text was written fresh, but this was not tested.
- Team size: guidelines say three per team, the synopsis lists one student; only that student is named.

## Rebuild commands
`scripts/run_pipeline.py` -> `reports/build_paper.py` -> `reports/build_front_page.py` ->
`presentation/export_slide_data.py` -> `node presentation/build_slides.js` -> `presentation/export_pdf.py`.
