"""Create notebooks/CKD_XAI_walkthrough.ipynb (then execute it with nbconvert).

    python scripts/build_notebook.py
    python -m jupyter nbconvert --to notebook --execute --inplace notebooks/CKD_XAI_walkthrough.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Explainable ML for chronic kidney disease: a guided walkthrough\n\n"
       "This notebook walks through the project **using the same code as the scripts and the app** "
       "(the `ckd_xai` package in `src/`). It does not re-run the full experiment; it loads the saved "
       "outputs of `python scripts/run_pipeline.py` and re-checks the key claims.\n\n"
       "Sections: 1) data, 2) leak-free pipeline, 3) model selection, 4) final test, "
       "5) reload and reproduce, 6) SHAP explanations, 7) limitations."),
    code("import sys, json, warnings\nfrom pathlib import Path\n"
         "warnings.filterwarnings('ignore', message='IProgress not found')   # harmless tqdm notice from shap\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT / 'src'))\n\n"
         "import numpy as np, pandas as pd\n"
         "from IPython.display import Image, display\n"
         "from ckd_xai import data, schema\n"
         "from ckd_xai.config import load_config, TABLES_DIR, FIGURES_DIR, PROCESSED_DIR\n"
         "pd.set_option('display.width', 160); pd.set_option('display.max_columns', 30)\n"
         "cfg = load_config()\n"
         "print('random seed:', cfg['project']['random_seed'])"),
    md("## 1. The data\n\nThe raw UCI file has stray tabs, `?` for missing values, two lines with a trailing "
       "comma and one with a doubled comma. `data.clean()` fixes each of these and logs every change. "
       "No record is removed."),
    code("df = data.load_clean()\n"
         "summary = json.loads((PROCESSED_DIR / 'cleaning_summary.json').read_text())\n"
         "print('records:', len(df), '| classes:', summary['class_counts'])\n"
         "print('line repairs:', summary['line_repairs'])\n"
         "print('impossible values set to missing:', [(d['feature'], d['value']) for d in summary['implausible_values_set_missing']])\n"
         "print('cells that differ from UCI data.csv:', summary['cross_check']['n_cells_different'])\n"
         "df.head()"),
    code("pd.DataFrame(schema.feature_dictionary_rows())[['code', 'name', 'type', 'unit', 'values / limits']]"),
    code("display(Image(filename=str(FIGURES_DIR / 'eda_missing_per_record_train.png'), width=420))"),
    md("CKD records have many more missing values than non-CKD records. Keep this in mind: missingness "
       "itself carries class information (see section 3)."),
    md("## 2. A leak-free pipeline\n\nThe test split is made once and locked. Every learned step "
       "(imputation, scaling, SMOTENC, classifier) lives inside one `imblearn` pipeline, so in "
       "cross-validation it is refitted on each training fold only. SMOTENC runs only during `fit`."),
    code("split = data.make_or_verify_split(df, cfg)\n"
         "X_train, y_train = data.get_partition(df, split, 'train')\n"
         "X_test, y_test = data.get_partition(df, split, 'test')\n"
         "print('train', X_train.shape, y_train.value_counts().to_dict(), '| test', X_test.shape, y_test.value_counts().to_dict())\n\n"
         "from ckd_xai.preprocessing import build_pipeline\n"
         "pipe = build_pipeline('random_forest', cfg)\n"
         "pipe"),
    code("# Show that SMOTENC adds synthetic non-CKD records during fit, but never at prediction time\n"
         "pipe.fit(X_train, y_train)\n"
         "prep, smote = pipe.named_steps['prep'], pipe.named_steps['resample']\n"
         "X_res, y_res = smote.fit_resample(prep.transform(X_train), y_train)\n"
         "print('training rows before / after SMOTENC:', len(X_train), '/', len(X_res), np.bincount(y_res))\n"
         "print('prediction rows for the test set:', len(pipe.predict_proba(X_test)))"),
    md("## 3. Model selection used training-set cross-validation only\n\n"
       "Rule fixed in `config.yaml` before the experiment: highest mean 5-fold CV ROC-AUC; models within "
       "0.005 are tied and ranked by mean CV recall, then F1."),
    code("cv = pd.read_csv(TABLES_DIR / 'cv_model_comparison.csv')\n"
         "cols = [f'{s}_{m}' for m in ['roc_auc', 'recall', 'f1'] for s in ('mean', 'std')]\n"
         "display(cv[['display_name'] + cols].round(4))\n"
         "sel = json.loads((ROOT / 'models' / 'selection.json').read_text())\n"
         "print(sel['reason'])\n"
         "print('test set used for selection:', sel['test_set_used'])"),
    code("from ckd_xai.train import select_model\n"
         "winner, ranking, _ = select_model(cv, cfg)   # re-apply the rule to the saved CV table\n"
         "print('rule re-applied ->', winner)\n"
         "display(pd.read_csv(TABLES_DIR / 'cv_resampling_ablation.csv')[['model', 'resampling', 'mean_roc_auc', 'mean_recall', 'mean_f1']].round(4))\n"
         "display(pd.read_csv(TABLES_DIR / 'cv_missingness_probe.csv')[['probe', 'mean_roc_auc', 'std_roc_auc']].round(3))"),
    md("SMOTENC gives no consistent improvement here, and a model that sees only *which* values are missing "
       "already reaches a high CV ROC-AUC."),
    md("## 4. Final evaluation on the untouched test set"),
    code("tm = pd.read_csv(TABLES_DIR / 'test_metrics.csv')\n"
         "display(tm[['display_name', 'roc_auc', 'accuracy', 'precision', 'recall', 'f1', 'fp', 'fn',\n"
         "            'accuracy_exact_lo', 'accuracy_exact_hi']].round(4))\n"
         "display(pd.read_csv(TABLES_DIR / 'leakage_audit.csv')[['check', 'passed']])"),
    code("display(Image(filename=str(FIGURES_DIR / 'test_score_distributions.png'), width=480))"),
    md("With only 80 test records, even 80/80 correct is compatible with a true accuracy as low as the exact "
       "95% lower bound shown above."),
    md("## 5. Reload the saved model and reproduce its predictions"),
    code("import joblib\n"
         "from ckd_xai.config import SELECTED_MODEL_PATH\n"
         "from ckd_xai.utils import ckd_probability\n"
         "model = joblib.load(SELECTED_MODEL_PATH)\n"
         "saved = pd.read_csv(TABLES_DIR / 'test_predictions.csv').set_index('record_id').loc[X_test.index]\n"
         "p = ckd_probability(model, X_test)\n"
         "print('max difference from saved predictions:', np.abs(p - saved[f\"{sel['selected_model']}_p_ckd\"]).max())"),
    md("## 6. SHAP explanations\n\nThe explainer uses a background of 100 real training records. For the "
       "Random Forest the output explained is the **probability of CKD**, so the baseline plus the "
       "contributions must equal the predicted probability."),
    code("from ckd_xai.inference import CKDPredictor\n"
         "predictor = CKDPredictor(cfg)\n"
         "record = {k: (None if pd.isna(v) else v) for k, v in X_test.loc[26].items()}\n"
         "res = predictor.predict(record)\n"
         "ex = predictor.explain(res['record'])\n"
         "print(f\"P(CKD) = {res['p_ckd']:.3f} -> {res['class_name']} | imputed: {res['imputed_values']}\")\n"
         "print(f\"baseline {ex['base_value']:.3f} + sum {ex['values'].sum():+.3f} = {ex['reconstructed_output']:.3f}; consistent: {ex['consistent']}\")\n"
         "ex['table'].head(8)"),
    code("display(Image(filename=str(FIGURES_DIR / 'shap_beeswarm.png'), width=460))"),
    code("imp = pd.read_csv(TABLES_DIR / 'shap_global_importance.csv')\n"
         "imp[['label', 'mean_abs_shap_test', 'rank_test', 'rank_train']].head(10).round(4)"),
    md("## 7. Limitations to remember\n\n"
       "* 400 records from one hospital; 80 test records.\n"
       "* Classes are very easy to separate in this dataset; it says little about early or borderline CKD.\n"
       "* Missingness is informative, so part of the signal may reflect which tests were ordered.\n"
       "* Probabilities come from a model trained on balanced data and are not calibrated risks.\n"
       "* SHAP explains the model's behaviour, not clinical causation.\n"
       "* No external validation has been done; this is an educational prototype, not a diagnostic tool."),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "CKD_XAI_walkthrough.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"wrote {out}")
