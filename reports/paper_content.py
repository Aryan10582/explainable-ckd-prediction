"""Content of the IEEE paper: text blocks, tables, figures, equations and references.

Every number is read from the saved experiment outputs via ckd_xai.report_data,
so the paper always matches results/ and models/.

Inline markup used in text:  *italic*  **bold**  x_{sub}  10^{sup}
Citations: [@key] or [@key1;key2]   Cross-references: {fig:key} {tab:key} {eq:key}
"""
from __future__ import annotations

import numpy as np

from ckd_xai import schema
from ckd_xai.models import DISPLAY_NAMES, MODEL_NAMES
from ckd_xai.report_data import median_iqr

FIG = "results/figures/"
TITLE = "Explainable Machine Learning for Chronic Kidney Disease Prediction Using SHAP-Based Feature Attribution"
AUTHOR = {
    "name": "Vedank Singh",
    "affiliation": ["School of Computer Engineering, Manipal Institute of Technology",
                    "Manipal Academy of Higher Education, Manipal, India"],
}

# IEEE-style references (verified; see docs/REFERENCE_VERIFICATION.md). *...* = italics.
REFERENCES = {
    "gbd2023": 'GBD 2023 Chronic Kidney Disease Collaborators, "Global, regional, and national burden of chronic kidney disease in adults, 1990–2023, and its attributable risk factors: A systematic analysis for the Global Burden of Disease Study 2023," *Lancet*, vol. 406, no. 10518, pp. 2461–2482, Nov. 2025, doi: 10.1016/S0140-6736(25)01853-7.',
    "levey2005": 'A. S. Levey et al., "Definition and classification of chronic kidney disease: A position statement from Kidney Disease: Improving Global Outcomes (KDIGO)," *Kidney Int.*, vol. 67, no. 6, pp. 2089–2100, Jun. 2005, doi: 10.1111/j.1523-1755.2005.00365.x.',
    "kdigo2024": 'Kidney Disease: Improving Global Outcomes (KDIGO) CKD Work Group, "KDIGO 2024 clinical practice guideline for the evaluation and management of chronic kidney disease," *Kidney Int.*, vol. 105, no. 4S, pp. S117–S314, Apr. 2024, doi: 10.1016/j.kint.2023.10.018.',
    "webster2017": 'A. C. Webster, E. V. Nagler, R. L. Morton, and P. Masson, "Chronic kidney disease," *Lancet*, vol. 389, no. 10075, pp. 1238–1252, Mar. 2017, doi: 10.1016/S0140-6736(16)32064-5.',
    "charleonnan2016": 'A. Charleonnan, T. Fufaung, T. Niyomwong, W. Chokchueypattanakit, S. Suwannawach, and N. Ninchawee, "Predictive analytics for chronic kidney disease using machine learning techniques," in *Proc. Manage. Innov. Technol. Int. Conf. (MITicon)*, Bang-Saen, Thailand, 2016, pp. MIT-80–MIT-83, doi: 10.1109/MITICON.2016.8025242.',
    "yang2024": 'W. Yang, N. Ahmed, and A. L. C. Barczak, "Comparative analysis of machine learning algorithms for CKD risk prediction," *IEEE Access*, vol. 12, pp. 171205–171220, 2024, doi: 10.1109/ACCESS.2024.3499355.',
    "phuong2025": 'N. D. Phuong, N. T. Tuyen, V. T. T. Linh, N. N. Nguyen, and T. Q. Nguyen, "Machine learning techniques in chronic kidney diseases: A comparative study of classification model performance," *Bioinform. Biol. Insights*, vol. 19, Art. no. 11779322251356563, 2025, doi: 10.1177/11779322251356563.',
    "loh2022": 'H. W. Loh, C. P. Ooi, S. Seoni, P. D. Barua, F. Molinari, and U. R. Acharya, "Application of explainable artificial intelligence for healthcare: A systematic review of the last decade (2011–2022)," *Comput. Methods Programs Biomed.*, vol. 226, Art. no. 107161, Nov. 2022, doi: 10.1016/j.cmpb.2022.107161.',
    "ribeiro2016": 'M. T. Ribeiro, S. Singh, and C. Guestrin, "\'Why should I trust you?\': Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, San Francisco, CA, USA, 2016, pp. 1135–1144, doi: 10.1145/2939672.2939778.',
    "lundberg2017": 'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems*, vol. 30, Long Beach, CA, USA, 2017.',
    "chawla2002": 'N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *J. Artif. Intell. Res.*, vol. 16, pp. 321–357, Jun. 2002, doi: 10.1613/jair.953.',
    "uci2015": 'L. Rubini, P. Soundarapandian, and P. Eswaran, "Chronic kidney disease," UCI Machine Learning Repository, 2015. [Dataset]. doi: 10.24432/C5G020.',
    "khalid2023": 'H. Khalid, A. Khan, M. Z. Khan, G. Mehmood, and M. S. Qureshi, "Machine learning hybrid model for the prediction of chronic kidney disease," *Comput. Intell. Neurosci.*, vol. 2023, Art. no. 9266889, 2023, doi: 10.1155/2023/9266889.',
    "khan2024": 'N. Khan et al., "Unveiling the predictive power: A comprehensive study of machine learning model for anticipating chronic kidney disease," *Front. Artif. Intell.*, vol. 6, Art. no. 1339988, Jan. 2024, doi: 10.3389/frai.2023.1339988.',
    "singamsetty2024": 'S. Singamsetty, S. Ghanta, S. Biswas, and A. Pradhan, "Enhancing machine learning-based forecasting of chronic renal disease with explainable AI," *PeerJ Comput. Sci.*, vol. 10, Art. no. e2291, Sep. 2024, doi: 10.7717/peerj-cs.2291.',
    "dharmarathne2024": 'G. Dharmarathne, M. Bogahawaththa, M. McAfee, U. Rathnayake, and D. P. P. Meddage, "On the diagnosis of chronic kidney disease using a machine learning-based interface with explainable artificial intelligence," *Intell. Syst. Appl.*, vol. 22, Art. no. 200397, Jun. 2024, doi: 10.1016/j.iswa.2024.200397.',
    "ghosh2024": 'S. K. Ghosh and A. H. Khandoker, "Investigation on explainable machine learning models to predict chronic kidney diseases," *Sci. Rep.*, vol. 14, no. 1, Art. no. 3687, Feb. 2024, doi: 10.1038/s41598-024-54375-4.',
    "chouit2026": 'E. M. Chouit, M. Rachdi, M. Bellafkih, and B. Raouyane, "Interpretable machine learning for chronic kidney disease prediction: Insights from SHAP and LIME analyses," *PLOS One*, vol. 21, no. 2, Art. no. e0343205, Feb. 2026, doi: 10.1371/journal.pone.0343205.',
    "rezk2025": 'N. G. Rezk, S. Alshathri, A. Sayed, and E. E.-D. Hemdan, "Explainable AI for chronic kidney disease prediction in medical IoT: Integrating GANs and few-shot learning," *Bioengineering*, vol. 12, no. 4, Art. no. 356, Mar. 2025, doi: 10.3390/bioengineering12040356.',
    "lemaitre2017": 'G. Lemaître, F. Nogueira, and C. K. Aridas, "Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning," *J. Mach. Learn. Res.*, vol. 18, no. 17, pp. 1–5, 2017.',
    "pedregosa2011": 'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.',
    "cox1958": 'D. R. Cox, "The regression analysis of binary sequences," *J. Roy. Statist. Soc. Ser. B*, vol. 20, no. 2, pp. 215–232, Jul. 1958, doi: 10.1111/j.2517-6161.1958.tb00292.x.',
    "cortes1995": 'C. Cortes and V. Vapnik, "Support-vector networks," *Mach. Learn.*, vol. 20, no. 3, pp. 273–297, Sep. 1995, doi: 10.1007/BF00994018.',
    "breiman2001": 'L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, Oct. 2001, doi: 10.1023/A:1010933404324.',
    "chen2016": 'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, San Francisco, CA, USA, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.',
    "varma2006": 'S. Varma and R. Simon, "Bias in error estimation when using cross-validation for model selection," *BMC Bioinf.*, vol. 7, Art. no. 91, 2006, doi: 10.1186/1471-2105-7-91.',
    "kaufman2012": 'S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," *ACM Trans. Knowl. Discovery Data*, vol. 6, no. 4, pp. 1–21, Dec. 2012, doi: 10.1145/2382577.2382579.',
    "goorbergh2022": 'R. van den Goorbergh, M. van Smeden, D. Timmerman, and B. Van Calster, "The harm of class imbalance corrections for risk prediction models: Illustration and simulation using logistic regression," *J. Amer. Med. Inform. Assoc.*, vol. 29, no. 9, pp. 1525–1534, Sep. 2022, doi: 10.1093/jamia/ocac093.',
    "groenwold2020": 'R. H. H. Groenwold, "Informative missingness in electronic health record systems: The curse of knowing," *Diagn. Progn. Res.*, vol. 4, Art. no. 8, 2020, doi: 10.1186/s41512-020-00077-0.',
    "lundberg2020": 'S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," *Nature Mach. Intell.*, vol. 2, no. 1, pp. 56–67, Jan. 2020, doi: 10.1038/s42256-019-0138-9.',
    "levey2009": 'A. S. Levey et al., "A new equation to estimate glomerular filtration rate," *Ann. Intern. Med.*, vol. 150, no. 9, pp. 604–612, May 2009, doi: 10.7326/0003-4819-150-9-200905050-00006.',
    "stauffer2014": 'M. E. Stauffer and T. Fan, "Prevalence of anemia in chronic kidney disease in the United States," *PLoS One*, vol. 9, no. 1, Art. no. e84943, Jan. 2014, doi: 10.1371/journal.pone.0084943.',
    "rudin2019": 'C. Rudin, "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead," *Nature Mach. Intell.*, vol. 1, no. 5, pp. 206–215, May 2019, doi: 10.1038/s42256-019-0048-x.',
    "collins2024": 'G. S. Collins et al., "TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods," *BMJ*, vol. 385, Art. no. e078378, Apr. 2024, doi: 10.1136/bmj-2023-078378.',
}

# (LaTeX for Markdown, MathML for Word)
EQUATIONS = {
    "smote": (r"x' = x_i + \lambda\,(\hat{x}_i - x_i), \quad \lambda \sim U(0,1)",
              "<mi>x</mi><mo>′</mo><mo>=</mo><msub><mi>x</mi><mi>i</mi></msub><mo>+</mo><mi>λ</mi><mo>(</mo>"
              "<msub><mover accent=\"true\"><mi>x</mi><mo>^</mo></mover><mi>i</mi></msub><mo>−</mo>"
              "<msub><mi>x</mi><mi>i</mi></msub><mo>)</mo><mo>,</mo><mspace width=\"0.6em\"/><mi>λ</mi><mo>∼</mo>"
              "<mi>U</mi><mo>(</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>)</mo>"),
    "pr": (r"P = \frac{TP}{TP+FP}, \qquad R = \frac{TP}{TP+FN}",
           "<mi>P</mi><mo>=</mo><mfrac><mrow><mi>TP</mi></mrow><mrow><mi>TP</mi><mo>+</mo><mi>FP</mi></mrow></mfrac>"
           "<mo>,</mo><mspace width=\"1em\"/><mi>R</mi><mo>=</mo><mfrac><mrow><mi>TP</mi></mrow><mrow><mi>TP</mi>"
           "<mo>+</mo><mi>FN</mi></mrow></mfrac>"),
    "f1": (r"F_1 = \frac{2PR}{P+R}",
           "<msub><mi>F</mi><mn>1</mn></msub><mo>=</mo><mfrac><mrow><mn>2</mn><mi>P</mi><mi>R</mi></mrow><mrow><mi>P</mi>"
           "<mo>+</mo><mi>R</mi></mrow></mfrac>"),
    "shapley": (r"\phi_j = \sum_{S \subseteq F \setminus \{j\}} w_S\,\big[v(S \cup \{j\}) - v(S)\big]",
                "<msub><mi>φ</mi><mi>j</mi></msub><mo>=</mo><munder><mo>∑</mo><mrow><mi>S</mi><mo>⊆</mo><mi>F</mi>"
                "<mo>∖</mo><mo>{</mo><mi>j</mi><mo>}</mo></mrow></munder><msub><mi>w</mi><mi>S</mi></msub>"
                "<mo>[</mo><mi>v</mi><mo>(</mo><mi>S</mi><mo>∪</mo><mo>{</mo><mi>j</mi><mo>}</mo><mo>)</mo><mo>−</mo>"
                "<mi>v</mi><mo>(</mo><mi>S</mi><mo>)</mo><mo>]</mo>"),
    "value": (r"v(S) = \frac{1}{|B|} \sum_{\mathbf{z} \in B} f(\mathbf{x}_S, \mathbf{z}_{\bar S})",
              "<mi>v</mi><mo>(</mo><mi>S</mi><mo>)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mo>|</mo><mi>B</mi><mo>|</mo>"
              "</mrow></mfrac><munder><mo>∑</mo><mrow><mi>z</mi><mo>∈</mo><mi>B</mi></mrow></munder><mi>f</mi>"
              "<mo>(</mo><msub><mi>x</mi><mi>S</mi></msub><mo>,</mo><msub><mi>z</mi><mover><mi>S</mi><mo>¯</mo></mover>"
              "</msub><mo>)</mo>"),
    "additive": (r"f(\mathbf{x}) = \phi_0 + \sum_{j=1}^{M} \phi_j",
                 "<mi>f</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><msub><mi>φ</mi><mn>0</mn></msub><mo>+</mo>"
                 "<munderover><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>M</mi></munderover><msub>"
                 "<mi>φ</mi><mi>j</mi></msub>"),
}


def _pm(m, s, d=4):
    return f"{m:.{d}f} ± {s:.{d}f}"


def _sci(x):
    mant, exp = f"{x:.1e}".split("e")
    return f"{mant} × 10^{{{int(exp)}}}"


def _contribs(example, n):
    """'serum creatinine of 2.4 mg/dL (+0.067)' phrases from the saved local-explanation table."""
    out = []
    for part in [s.strip() for s in example.top_contributions.split(";")][:n]:
        label_value, shap = part.rsplit(": ", 1)
        label, value = label_value.split(" = ", 1)
        value = value.replace(" (imputed)", ", imputed")
        joiner = " = " if value.split(",")[0] in {"yes", "no", "normal", "abnormal", "present", "notpresent",
                                                  "good", "poor"} else " of "
        out.append(f"{label.lower()}{joiner}{value} ({shap})")
    return ", ".join(out[:-1]) + " and " + out[-1]


def _passed_tests():
    from ckd_xai.config import LOGS_DIR

    last = (LOGS_DIR / "pytest_report.txt").read_text(encoding="utf-8").strip().splitlines()[-1]
    return int(last.split(" passed")[0].split()[-1])


NUMBER_WORDS = {40: "Forty", 41: "Forty-one", 42: "Forty-two", 43: "Forty-three", 44: "Forty-four",
                45: "Forty-five", 46: "Forty-six", 47: "Forty-seven", 48: "Forty-eight", 49: "Forty-nine", 50: "Fifty"}


def _ci(lo, hi):
    return f"{lo:.3f}–{hi:.3f}"


def build_blocks(R: dict) -> dict:
    cv, test, W = R["cv"], R["test"], R["winner"]
    wname = R["winner_name"]
    tw = test.loc[W]
    clean, split, lat = R["clean"], R["split"], R["latency"]
    imp = R["shap_imp"]
    loc = R["local"]
    pk = R["env"]["packages"]
    nm = R["near_miss_id"]
    shapc = R["shap_checks"]
    n_changed, n_total = [int(t) for t in shapc["explainer_note"].split(" split")[0].split(" of ")]
    lr_err = R["errors"][R["errors"].model == "logistic_regression"]
    lr_ids = lr_err.record_id.tolist()
    rec = R["records"]
    sv, recs = R["shap_values"], R["records"].loc[R["shap_values"].index]

    def thr(f):
        obs = recs[f].notna()
        s, x = sv.loc[obs, f], recs.loc[obs, f]
        return x[s < 0], x[s > 0]

    sc_neg, sc_pos = thr("sc")
    hb_neg, hb_pos = thr("hemo")

    # Claims in the text that are checked here rather than assumed.
    abl = R["ablation"]
    within_sd = all(abs(abl[m][k]) < cv.loc[m, f"std_{k}"] for m in MODEL_NAMES
                    for k in ("roc_auc", "accuracy", "recall", "f1", "specificity") if cv.loc[m, f"std_{k}"] > 0
                    ) and all(abs(abl[m][k]) < 1e-12 for m in MODEL_NAMES for k in ("roc_auc", "specificity")
                              if cv.loc[m, f"std_{k}"] == 0)
    assert within_sd, "Ablation text claims every change is within one fold SD"
    assert sc_neg.max() < sc_pos.min() and hb_pos.max() < hb_neg.min()
    assert set(R["binary_zero_in_noncdk"]) == set(schema.BINARY)
    assert R["audit_passed"] == R["audit_total"]
    tied = [m for m in MODEL_NAMES if cv.loc[m, "mean_roc_auc"] >= R["cv_auc_max"] - R["selection"]["rule"]["auc_tie_tolerance"]]
    perfect = [m for m in MODEL_NAMES if test.loc[m, "accuracy"] == 1.0]
    rec_cols = "missing_pct_train_ckd", "missing_pct_train_nonckd"
    mb = R["miss_by_class"]
    (c1a, c1b), c1v = R["corr_top"].index[0], R["corr_top"].iloc[0]
    train_ids = R["records"].index.difference(R["preds"].index)
    hemo_median = R["records"].loc[train_ids, "hemo"].median()
    ckd_share = 100 * clean["class_counts"]["ckd"] / clean["n_records_clean"]
    from ckd_xai.config import load_config
    k = load_config()["cv"]["n_splits"]
    fold_ckd = split["class_counts"]["train"]["ckd"] // k
    fold_non = split["class_counts"]["train"]["notckd"] // k
    n_tests = _passed_tests()
    # Statements written in words below are checked against the saved results here.
    assert R["shap_top5"] == ["Serum creatinine", "Specific gravity", "Haemoglobin", "Albumin grade",
                              "Packed cell volume"], R["shap_top5"]
    assert cv["mean_roc_auc"].idxmax() == "svm" and cv["mean_recall"].idxmax() == W
    assert all(abl[m]["recall"] <= 0 and abl[m]["f1"] < 0 for m in MODEL_NAMES if m != "random_forest")
    assert abl["random_forest"]["f1"] > 0 and abl["random_forest"]["recall"] >= 0
    assert all(rec.loc[i, c] == "no" for i in lr_ids for c in ("htn", "dm", "ane"))
    assert (sv["sg"][recs["sg"] <= 1.015] > 0).all() and (sv["sg"][recs["sg"] >= 1.020] < 0).all()
    assert (sv["al"][recs["al"] >= 1] > 0).all() and (sv["al"][recs["al"] == 0] < 0).all()
    assert abs(hemo_median - 12.6) < 1e-9

    abstract = (
        "Chronic kidney disease often progresses silently, and machine learning models that flag it from routine "
        "tests are only useful to clinicians if their predictions can be explained. This paper presents a "
        "reproducible pipeline that compares logistic regression, a support vector machine, a random forest and "
        "extreme gradient boosting on the 400-record University of California, Irvine chronic kidney disease "
        "dataset, and explains the selected model with Shapley additive explanations. Missing values were imputed, "
        "classes were balanced with a synthetic oversampling method for mixed numeric and categorical data, and "
        "every learned step was fitted inside the training folds only. A rule fixed in advance chose the model by "
        "its mean five-fold cross-validated area under the receiver operating characteristic curve, with recall "
        f"and F1-score as tie-breakers. All four models scored between {R['cv_auc_min']:.4f} and "
        f"{R['cv_auc_max']:.4f}; the random forest was selected and classified all {split['n_test']} untouched "
        f"test records correctly (exact 95% confidence interval for accuracy {_ci(tw.accuracy_exact_lo, tw.accuracy_exact_hi)}). "
        "Serum creatinine, specific gravity, haemoglobin, albumin and packed cell volume drove its predictions, in "
        "line with clinical markers of kidney damage and anaemia. Oversampling gave no consistent benefit, and the "
        "near-perfect separability, informative missingness and single-centre data mean that these results do "
        "not establish clinical validity."
    )
    index_terms = ("Chronic kidney disease, class imbalance, explainable artificial intelligence, machine learning, "
                   "Shapley additive explanations")

    feature_rows = [
        ["Numeric (11)", "age (years); blood pressure (mm Hg); random blood glucose, blood urea, serum creatinine "
                         "(mg/dL); sodium, potassium (mEq/L); haemoglobin (g/dL); packed cell volume (no unit "
                         "documented); white cell count (cells/cumm); red cell count (millions/cmm)"],
        ["Ordinal (3)", "specific gravity (1.005–1.025, five levels); albumin and sugar grades (0–5)"],
        ["Binary (10)", "red blood cells, pus cells (normal/abnormal); pus cell clumps, bacteria (present/absent); "
                        "hypertension, diabetes mellitus, coronary artery disease, pedal oedema, anaemia (yes/no); "
                        "appetite (good/poor)"],
        ["Target", f"CKD = 1 (positive, n = {clean['class_counts']['ckd']}); non-CKD = 0 "
                   f"(n = {clean['class_counts']['notckd']})"],
    ]

    def hp(m):
        p = cv.loc[m, "best_params"]
        p = eval(p) if isinstance(p, str) else p  # noqa: S307 (our own saved dict)
        if m == "logistic_regression":
            return f"{'L1' if p['l1_ratio'] == 1 else 'L2'} penalty, C = {p['C']:g}"
        if m == "svm":
            return f"{p['kernel'].upper()} kernel, C = {p['C']:g}, γ = {p['gamma']}"
        if m == "random_forest":
            depth = "unlimited" if p["max_depth"] is None or (isinstance(p["max_depth"], float) and np.isnan(p["max_depth"])) else p["max_depth"]
            return (f"{p['n_estimators']} trees, depth {depth}, max features {p['max_features']}, "
                    f"min leaf {p['min_samples_leaf']}")
        return (f"{p['n_estimators']} trees, depth {p['max_depth']}, learning rate {p['learning_rate']:g}, "
                f"subsample {p['subsample']:g}")

    # Models as columns so the comparison tables fit one IEEE column (no section breaks needed).
    short = {"logistic_regression": "LR", "svm": "SVM", "random_forest": "RF", "xgboost": "XGBoost"}
    model_hdr = ["Metric"] + [short[m] + ("*" if m == W else "") for m in MODEL_NAMES]
    metric_lbl = {"roc_auc": "ROC-AUC", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                  "f1": "F1-score", "specificity": "Specificity"}
    cv_rows = [[metric_lbl[k]] + [f"{cv.loc[m, f'mean_{k}']:.3f}\n± {cv.loc[m, f'std_{k}']:.3f}" for m in MODEL_NAMES]
               for k in ("roc_auc", "accuracy", "precision", "recall", "f1")]
    hp_text = "; ".join(f"{DISPLAY_NAMES[m]}, {hp(m)}" for m in MODEL_NAMES)

    def signed(x):
        return "0.000" if abs(x) < 0.0005 else f"{x:+.3f}".replace("-", "−")

    abl_rows = [[DISPLAY_NAMES[m]] + [signed(abl[m][k]) for k in ("roc_auc", "recall", "f1", "specificity")]
                for m in MODEL_NAMES]

    def with_ci(m, k):
        t = test.loc[m]
        return f"{t[k]:.3f}\n{_ci(t[f'{k}_exact_lo'], t[f'{k}_exact_hi'])}"

    test_rows = [["ROC-AUC"] + [f"{test.loc[m, 'roc_auc']:.3f}" for m in MODEL_NAMES],
                 ["Accuracy"] + [with_ci(m, "accuracy") for m in MODEL_NAMES],
                 ["Precision"] + [f"{test.loc[m, 'precision']:.3f}" for m in MODEL_NAMES],
                 ["Recall"] + [with_ci(m, "recall") for m in MODEL_NAMES],
                 ["Specificity"] + [with_ci(m, "specificity") for m in MODEL_NAMES],
                 ["F1-score"] + [f"{test.loc[m, 'f1']:.3f}" for m in MODEL_NAMES],
                 ["FP / FN"] + [f"{int(test.loc[m, 'fp'])} / {int(test.loc[m, 'fn'])}" for m in MODEL_NAMES]]

    direction = {"sc": "high", "sg": "low (≤ 1.015)", "hemo": "low", "al": "grade ≥ 1", "pcv": "low",
                 "htn": "yes", "dm": "yes", "rbcc": "low", "bgr": "high", "sod": "low", "bu": "high", "bp": "high"}
    shap_rows = [[str(int(r.rank_test)), r.label, f"{r.mean_abs_shap_test:.3f}", f"{r.share_of_total_pct:.1f}",
                  direction.get(r.feature, "")] for r in imp.head(6).itertuples()]

    ex = {k: loc.loc[k] for k in loc.index}
    c_ckd, c_non, c_bor = ex["typical_ckd"], ex["typical_non_ckd"], ex["borderline"]

    low = [t.lower() for t in R["shap_top5"]]
    top5_txt = (", ".join(low[:-1]) + " and " + low[-1]).capitalize()

    B = []
    add = B.append
    add(("h1", "Introduction"))
    add(("h2", "Problem and Motivation"))
    add(("p", "Chronic kidney disease (CKD) is defined by abnormalities of kidney structure or function, such as a "
              "reduced glomerular filtration rate (GFR) or albuminuria, that persist for more than three months "
              "[@levey2005;kdigo2024]. The Global Burden of Disease Study estimated that 788 million adults had CKD "
              "in 2023, an age-standardised prevalence of 14.2% [@gbd2023]. Because early CKD rarely causes "
              "symptoms, many patients are identified late, when fewer options remain to slow its progression "
              "[@webster2017]. Routine blood and urine tests carry much of the information needed to flag likely "
              "CKD, which has made CKD prediction a popular machine learning (ML) task "
              "[@charleonnan2016;yang2024;phuong2025]."))
    add(("p", "High accuracy alone is not enough in a clinical setting: a clinician must see why a model reached a "
              "decision before judging whether to trust it, and model opacity remains a main barrier to ML in "
              "healthcare [@loh2022]. Post-hoc methods such as local interpretable model-agnostic explanations "
              "(LIME) [@ribeiro2016] and Shapley additive explanations (SHAP) [@lundberg2017] attribute each "
              "prediction to the input features and so open the model's behaviour to inspection."))
    add(("h2", "Objectives"))
    add(("numbered", [
        "To acquire, validate and preprocess the UCI CKD dataset, handling missing values, categorical encoding, "
        "feature scaling and class imbalance without leaking information from evaluation data.",
        "To tune and compare logistic regression (LR), a support vector machine (SVM), a random forest (RF) and "
        "XGBoost using accuracy, precision, recall, F1-score and the area under the receiver operating "
        "characteristic curve (ROC-AUC), and to select one model.",
        "To explain the selected model globally and for individual patients with SHAP, and to compare the most "
        "influential features with the clinical literature.",
    ]))
    add(("h2", "Contribution"))
    add(("p", "The contribution is a reproducible comparative prediction and explanation pipeline, not a new "
              "algorithm. Three refinements were made to the project synopsis: the final model is chosen by "
              "cross-validation on the training set instead of by test-set performance, so the untouched test set "
              "gives an unbiased final estimate; SMOTE is replaced by its mixed-data variant SMOTE-NC "
              "[@chawla2002], so synthetic records never contain impossible category values; and every reported "
              "number is produced by scripts with a fixed seed, an automated leakage audit and a SHAP additivity "
              "check. A local Streamlit application demonstrates the saved model. {fig:workflow} summarises the "
              "workflow."))
    add(("fig", FIG + "workflow_diagram.png", "workflow",
         "System workflow. Blue steps use training records only; the test partition (orange) is locked after the "
         "split and used once, for the final evaluation of the frozen pipelines.", 3.0))

    add(("h1", "Related Work"))
    add(("p", "The UCI CKD dataset [@uci2015] is the most widely used benchmark for CKD classification. "
              "Charleonnan et al. [@charleonnan2016] compared k-nearest neighbours, SVM, LR and decision trees. "
              "Yang et al. [@yang2024] compared eight classical and three neural-network classifiers on the UCI data "
              "with SMOTE-NC balancing and feature selection, and found almost no difference between the balanced "
              "and unbalanced versions. Phuong et al. [@phuong2025] reported 100% accuracy for RF, XGBoost, SVM "
              "and LR after adding polynomial features. Khalid et al. [@khalid2023] proposed a stacked hybrid "
              "ensemble, and Khan et al. [@khan2024] compared five classifier families on a Pakistani cohort."))
    add(("p", "Several studies add explanations. Singamsetty et al. [@singamsetty2024] tuned models by grid search "
              "and applied LIME and SHAP (99.07% accuracy). Dharmarathne et al. [@dharmarathne2024] explained "
              "XGBoost with SHAP and partial dependence plots behind a graphical interface. Ghosh and Khandoker "
              "[@ghosh2024] found creatinine, glycated haemoglobin and age most influential in a 491-patient "
              "hospital cohort. Chouit et al. [@chouit2026] combined XGBoost, SMOTE, SHAP and LIME; on the UCI "
              "data, specific gravity, haemoglobin and serum creatinine led. Rezk et al. [@rezk2025] added "
              "generative adversarial imputation and few-shot learning. This project keeps to conventional "
              "classifiers but combines what these studies rarely report together: a predeclared training-only "
              "selection rule, mixed-type oversampling inside cross-validation with an ablation, exact uncertainty "
              "intervals and a verified SHAP decomposition."))

    add(("h1", "Dataset and Preprocessing"))
    add(("h2", "Data Source"))
    add(("p", "The data are the UCI Chronic Kidney Disease dataset [@uci2015], donated in 2015 and, according to "
              "its documentation, collected over about two months at a hospital in Tamil Nadu, India. It has "
              f"{clean['n_records_clean']} records ({clean['class_counts']['ckd']} CKD, "
              f"{clean['class_counts']['notckd']} non-CKD) and 24 predictors ({{tab:features}}). CKD is the "
              "positive class. Units were copied from the documentation; where none is given, none is assumed."))
    add(("table", "features", "Predictors in the UCI CKD dataset (units as documented)", ["Type", "Features"],
         feature_rows, [0.75, 2.45], False, 8.5))
    add(("h2", "Acquisition, Cleaning and Validation"))
    add(("p", "A script downloads the official archive, caches it and records SHA-256 checksums; a local-file "
              "option supports offline use. The original attribute-relation (ARFF) file contained stray tabs and "
              f"spaces in {R['whitespace_cells']} cells (including two class labels), \"?\" for missing values, two "
              "lines with a trailing comma and one with a doubled comma. Each defect was repaired by an explicit, "
              "logged rule, and the result matched UCI's own CSV conversion cell for cell. Three physiologically "
              "impossible values (sodium 4.5 mEq/L; potassium 39 and 47 mEq/L) fell outside plausibility limits "
              "fixed before modelling and were set to missing; no record was removed. No two records share a "
              f"predictor vector. Overall, {clean['missing_cells_total']} of {clean['n_records_clean'] * 24} "
              f"values ({R['missing_pct_cells']:.1f}%) are missing, and {clean['records_with_any_missing']} records "
              "have at least one gap."))
    add(("h2", "Exploratory Analysis"))
    add(("p", f"Class-conditioned exploration used only the {split['n_train']} training records. CKD records had "
              f"higher serum creatinine (median {median_iqr(R, 'sc', 'CKD').split(' ')[0]} vs "
              f"{median_iqr(R, 'sc', 'Non-CKD').split(' ')[0]} mg/dL) and albumin grades, and lower haemoglobin "
              f"({median_iqr(R, 'hemo', 'CKD').split(' ')[0]} vs {median_iqr(R, 'hemo', 'Non-CKD').split(' ')[0]} g/dL), "
              f"packed cell volume ({median_iqr(R, 'pcv', 'CKD').split(' ')[0]} vs "
              f"{median_iqr(R, 'pcv', 'Non-CKD').split(' ')[0]}) and specific gravity (1.015 vs 1.020). Strikingly, "
              "none of the ten binary abnormalities, including hypertension and diabetes, occurs in any non-CKD "
              "training record, which suggests that the non-CKD group consists of screened healthy controls. "
              "Missingness also differs by class ({fig:missing}): the red cell count, for example, is missing in "
              f"{mb.loc['rbcc', rec_cols[0]]:.1f}% of CKD but {mb.loc['rbcc', rec_cols[1]]:.1f}% of non-CKD training "
              f"records. The strongest correlation links {schema.FEATURE_BY_NAME[c1a].label.lower()} and "
              f"{schema.FEATURE_BY_NAME[c1b].label.lower()} (Spearman ρ = {c1v:.2f}). These are associations "
              "within this sample, not causal effects."))
    add(("fig", FIG + "eda_missing_per_record_train.png", "missing",
         "Missing values per record in the training set, by class. Non-CKD records are mostly complete, so the "
         "pattern of missingness itself carries class information."))
    add(("h2", "Preprocessing"))
    add(("p", "All preprocessing is fitted on training data inside one pipeline per model. Numeric features are "
              "imputed with the training median and standardised; ordinal grades are imputed with the training "
              "mode and standardised; binary features are encoded as 0/1 (1 = abnormal, present, yes or poor), "
              "and a category unseen at prediction time is treated as missing. Each predictor maps to exactly one "
              "model column, so SHAP values need no aggregation over one-hot columns. Scaling is needed by LR, SVM "
              "and the distance computation in SMOTE-NC; it does not change the splits of the tree models."))
    add(("h2", "Class Imbalance with SMOTE-NC"))
    add(("p", f"The training set has a {split['class_counts']['train']['ckd']}:{split['class_counts']['train']['notckd']} "
              "class ratio, so the minority class is non-CKD. SMOTE [@chawla2002] creates a synthetic minority "
              "record by interpolating between a minority record x_{i} and one of its k nearest minority "
              "neighbours x̂_{i}:"))
    add(("eq", "smote"))
    add(("p", "Interpolating 0/1 columns would create fractional \"patients\" with, say, 0.4 hypertension. "
              "SMOTE-NC, defined in the same paper, interpolates only continuous features, gives each nominal "
              "feature the most frequent value among the k neighbours and penalises nominal mismatches in the "
              "distance. The ordinal grades were also treated as nominal during resampling, so every synthetic "
              "record has valid levels (verified by the test suite). SMOTE-NC (k = 5) is a step of the "
              "imbalanced-learn pipeline [@lemaitre2017], so it runs only when a pipeline is fitted on training "
              "data and never touches validation or test records."))

    add(("h1", "Proposed Methodology"))
    add(("h2", "Data Split and Classifiers"))
    add(("p", f"Following {{fig:workflow}}, a stratified test partition of {split['n_test']} records "
              f"({split['class_counts']['test']['ckd']} CKD, {split['class_counts']['test']['notckd']} non-CKD; "
              "20%, seed 42) was created and locked, with a checksum of its record identifiers, before any tuning. "
              "Four classifiers spanning linear, kernel, bagging and boosting approaches were implemented with "
              "scikit-learn [@pedregosa2011] and XGBoost: LR [@cox1958] with an L1 or L2 penalty; an SVM "
              "[@cortes1995] with a radial-basis (RBF) or linear kernel, whose decision scores are turned into "
              "probabilities by sigmoid calibration fitted through an internal five-fold split of the training "
              "data; RF [@breiman2001]; and XGBoost [@chen2016] with the binary logistic objective and the CPU "
              "histogram tree method. These are also the models most often compared on CKD data "
              "[@yang2024;phuong2025]."))
    add(("h2", "Tuning and Predeclared Selection"))
    add(("p", "Hyperparameters were tuned by exhaustive grid search with stratified five-fold cross-validation "
              f"(CV) on the training set ({R['n_candidates']} candidates, {R['n_fits']} fits): LR with "
              "C ∈ {0.01, …, 100} and an L1 or L2 penalty; SVM with an RBF kernel (C ∈ {0.1, …, 100}, "
              "γ ∈ {scale, 0.01, 0.1}) or a linear kernel; RF with 300 trees, depth ∈ {unlimited, 4, 8}, "
              "minimum leaf ∈ {1, 3, 5} and √p or 0.5 features per split; and XGBoost with 100 or 300 trees, depth "
              "∈ {2, 3, 4}, learning rate ∈ {0.05, 0.1} and 0.8 row and column subsampling. All metrics use the "
              "predicted CKD probability at the default threshold of 0.5, which was not tuned."))
    add(("p", "The selection rule was written into the configuration file before any score was computed: the "
              "highest mean CV ROC-AUC wins; models within 0.005 of the best are tied and ranked by mean CV "
              f"recall, then F1-score, then simplicity. A validation fold holds about {fold_ckd} CKD and {fold_non} "
              f"non-CKD records, so one mis-ordered pair moves its ROC-AUC by about {1 / (fold_ckd * fold_non):.3f}. "
              "The synopsis proposed choosing the model with the best held-out test score; that would bias the "
              "test estimate upwards, because the same records would select and grade the model [@varma2006]. "
              "Here the test set is used once, after selection, to evaluate all four frozen models."))
    add(("h2", "Leakage Prevention"))
    add(("p", "Leakage [@kaufman2012] was prevented by design and then audited. Imputation, scaling, resampling "
              "and fitting all happen inside the pipeline; the record identifier and target are excluded from the "
              "predictors; the test set is locked; and the selection is saved before the evaluation code reads any "
              f"test record. An automated audit checks {R['audit_total']} properties, including that the fitted "
              "imputation medians equal the training-only medians, that predictions bypass the resampler, that no "
              "predictor vector appears in both partitions, and that training on shuffled labels gives "
              f"chance-level CV ROC-AUC (mean {R['perm_auc']:.3f})."))
    add(("h2", "Evaluation Metrics and Uncertainty"))
    add(("p", "With CKD as the positive class and TP, FP and FN the true positives, false positives and false "
              "negatives, precision P and recall R are"))
    add(("eq", "pr"))
    add(("p", "and the F1-score is 2PR/(P + R). A false negative is a missed CKD case, the more harmful error in "
              "screening; a false positive sends a person without CKD for unnecessary follow-up. ROC-AUC measures "
              "ranking quality independently of the threshold. Because the test set is small, 95% Clopper-Pearson "
              "exact intervals are reported for proportions, and stratified bootstrap intervals (2000 resamples) "
              "were also computed.", "noindent"))

    add(("h1", "Experimental Setup and Results"))
    add(("h2", "Setup and Reproducibility"))
    add(("p", "Experiments ran on a laptop with an AMD Ryzen 7 7840HS processor and 16 GB of memory under "
              f"Windows 11, using Python {R['env']['python'].rsplit('.', 1)[0]}, scikit-learn {pk['scikit-learn']}, "
              f"imbalanced-learn {pk['imbalanced-learn']}, XGBoost {pk['xgboost']} and SHAP {pk['shap']}. All "
              "settings live in one configuration file with seed 42. The whole pipeline took "
              f"{R['pipeline_seconds']:.0f} s (grid searches {R['grid_seconds']:.0f} s), and a full re-run "
              f"reproduced the CV results and test predictions byte for byte. One prediction took a median of "
              f"{lat['predict_ms_median']:.0f} ms and its explanation {lat['explain_ms_median']:.0f} ms. "
              f"{NUMBER_WORDS.get(n_tests, n_tests)} automated tests cover cleaning, schema validation, leakage, "
              "saved-model consistency, inference, SHAP consistency and the application."))
    tied_txt = "all four" if len(tied) == 4 else ", ".join(DISPLAY_NAMES[m] for m in tied)
    add(("h2", "Cross-Validation Comparison and Selection"))
    add(("p", f"The selected configurations were: {hp_text}. {{tab:cv}} gives their CV scores. Mean CV ROC-AUC "
              f"ranged from {R['cv_auc_min']:.4f} to {R['cv_auc_max']:.4f}, so {tied_txt} models were tied under "
              f"the predeclared tolerance. {wname} had the highest mean CV recall ({cv.loc[W, 'mean_recall']:.3f}) "
              f"and F1-score ({cv.loc[W, 'mean_f1']:.3f}) and was selected; SVM had the highest ROC-AUC "
              f"({cv.loc['svm', 'mean_roc_auc']:.4f}) but a recall of {cv.loc['svm', 'mean_recall']:.3f}. Pooled "
              "out-of-fold ROC-AUCs were between "
              f"{min(R['oof_auc'].values()):.4f} and {max(R['oof_auc'].values()):.4f}."))
    add(("table", "cv", "Five-fold CV results of the tuned models on the training set (mean ± SD; * = selected)",
         model_hdr, cv_rows, [0.75, 0.62, 0.62, 0.62, 0.62], False, 8.5))
    add(("h2", "Effect of SMOTE-NC"))
    add(("p", "{tab:ablation} compares each tuned model with and without SMOTE-NC on the same folds. The "
              "differences are tiny and inconsistent: resampling slightly lowered recall and F1-score for LR, SVM "
              "and XGBoost and slightly raised them for RF, and every change is smaller than one fold standard "
              f"deviation. The imbalance is mild ({ckd_share:.1f}:{100 - ckd_share:.1f}), and Yang et al. reported a "
              "similar null result on this dataset [@yang2024]. Resampling also shifts predicted probabilities away "
              "from the true prevalence [@goorbergh2022], so the probabilities below are read as rankings rather "
              "than risks. SMOTE-NC was kept, as the synopsis specifies, but no benefit is claimed."))
    add(("table", "ablation", "Change in mean CV score when SMOTE-NC is used (SMOTE-NC minus no resampling)",
         ["Model", "ROC-AUC", "Recall", "F1-score", "Specificity"], abl_rows, [0.98, 0.55, 0.47, 0.55, 0.68], False,
         8.5))
    add(("p", "A further probe asks how much class information lies in missingness alone: an LR model trained only "
              f"on the 24 missing-value indicators reached a mean CV ROC-AUC of {R['probe_auc']:.3f} "
              f"(SD {R['probe_auc_sd']:.3f}). Which values are missing, probably reflecting which tests were "
              "ordered, is itself highly predictive, and a model may partly exploit it [@groenwold2020]."))
    add(("h2", "Final Test Results"))
    lr_desc = " and ".join(str(i) for i in lr_ids)
    lr_sc = " and ".join(f"{rec.loc[i, 'sc']:g}" for i in lr_ids)
    add(("p", f"{{tab:test}} reports the frozen models on the {split['n_test']} test records. "
              f"{', '.join(DISPLAY_NAMES[m] for m in perfect[:-1])} and {DISPLAY_NAMES[perfect[-1]]} classified "
              f"every record correctly (ROC-AUC 1.000); LR missed two CKD cases (records {lr_desc}) that had serum "
              f"creatinine of {lr_sc} mg/dL but no hypertension, diabetes or anaemia. For the selected {wname}, the "
              f"exact 95% interval is {_ci(tw.accuracy_exact_lo, tw.accuracy_exact_hi)} for accuracy and "
              f"{_ci(tw.recall_exact_lo, tw.recall_exact_hi)} for recall, so the data remain compatible with a few "
              "errors per hundred patients; the bootstrap intervals collapse to a single value when no error "
              "occurs and are therefore not shown."))
    add(("table", "test", f"Frozen models on the held-out test set (n = {split['n_test']}; exact 95% CI below "
                          "each proportion; * = selected)",
         model_hdr, test_rows, [0.62, 0.65, 0.65, 0.65, 0.65], False, 8))
    add(("p", f"{{fig:scores}} shows that perfect accuracy hides a near miss: non-CKD record {nm} scored "
              f"{R['near_miss_p']['random_forest']:.3f} under RF and {R['near_miss_p']['xgboost']:.3f} under "
              "XGBoost, just below the threshold."))
    add(("h2", "Investigating the Perfect Scores"))
    add(("p", f"Perfect test scores prompted a search for leakage. The audit passed all {R['audit_total']} "
              "checks, no record is duplicated across the split, and shuffled labels gave chance-level performance. "
              "The explanation lies in the data: the classes are almost separable by single features (serum "
              f"creatinine interquartile ranges of {median_iqr(R, 'sc', 'CKD').split('[')[1].rstrip(']')} vs "
              f"{median_iqr(R, 'sc', 'Non-CKD').split('[')[1].rstrip(']')} mg/dL), and non-CKD records contain no "
              "abnormal binary finding. Other studies on this dataset report similar accuracy "
              "[@singamsetty2024;phuong2025]. Such separability means the results say little about early or "
              "borderline CKD and do not demonstrate clinical readiness."))
    add(("fig", FIG + "test_score_distributions.png", "scores",
         "Predicted CKD probabilities on the test set by true class (circles: non-CKD, triangles: CKD). Records "
         "near the threshold are the least certain predictions."))

    add(("h1", "SHAP Explainability and Discussion"))
    add(("h2", "Method"))
    add(("p", "For a record x with M = 24 features, SHAP [@lundberg2017] gives feature j the Shapley value"))
    add(("eq", "shapley"))
    add(("p", "where F is the set of all features, the weight w_{S} = |S|!(M − |S| − 1)!/M! averages over all "
              "orders in which features can be added, and v(S) is the expected model output when the features in S "
              "keep the record's values and the others are taken from a background set B of training records:",
         "noindent"))
    add(("eq", "value"))
    add(("p", "The values satisfy local accuracy,", "noindent"))
    add(("eq", "additive"))
    add(("p", f"where the baseline φ_{{0}} is the average model output over B. The explained output is the {wname}'s "
              "probability of CKD, so a positive φ_{j} pushes a prediction towards CKD. Interventional Tree SHAP "
              f"[@lundberg2020] was used with a background of {shapc['background'].split(' ')[0]} real training "
              f"records (no synthetic records), giving φ_{{0}} = {shapc['base_value']:.3f}. Attributions are "
              "reported per original feature, with values converted back to clinical units after imputation.",
         "noindent"))
    add(("p", "Identity {eq:additive} was checked for every explained record. It first failed by up to "
              f"{shapc['max_additivity_error_without_threshold_fix']:.4f}, because the SHAP library routed background "
              "values lying within single-precision rounding of a split threshold differently from scikit-learn; "
              "SMOTE-NC makes such near-ties likely by creating values almost identical to real ones. The explainer "
              f"therefore uses a copy of the forest with its thresholds ({n_changed} of {n_total}) rounded down to "
              "single precision, which provably leaves every prediction unchanged; the maximum error then fell to "
              f"{_sci(shapc['max_additivity_error'])}."))
    add(("h2", "Global Importance"))
    add(("p", f"{{fig:beeswarm}} and {{tab:shap}} summarise the SHAP values on the {shapc['n_explained_test']} test "
              f"records. {top5_txt} are the five most influential features, with {R['shap_top5_share']:.1f}% of "
              f"the total attribution; the ranking on the {shapc['n_explained_train_for_stability']} training "
              f"records is nearly identical (Spearman ρ = {shapc['spearman_rank_corr_test_vs_train_importance']:.3f}). "
              f"The effects are threshold-like: creatinine contributions are negative for every test record at or "
              f"below {sc_neg.max():g} mg/dL and positive from {sc_pos.min():g} mg/dL, and haemoglobin "
              f"contributions are positive up to {hb_pos.max():g} g/dL and negative from {hb_neg.min():g} g/dL. "
              "Specific gravity of 1.005–1.015 and any albumin grade above 0 push towards CKD."))
    add(("table", "shap", "Top six SHAP features (test set)",
         ["#", "Feature", "Mean |SHAP|", "%", "CKD when"], shap_rows, [0.2, 1.15, 0.75, 0.35, 0.8],
         False, 8.5))
    add(("fig", FIG + "shap_beeswarm.png", "beeswarm",
         f"SHAP summary plot of the selected {wname} on the test set. Each dot is one record; its position is the "
         "contribution to the probability of CKD and its colour the feature value (red = high; for binary "
         "features, red = the condition in brackets)."))
    add(("h2", "Local Explanations"))
    add(("p", "Examples were chosen by a fixed rule: for each class, the correctly classified test record whose "
              "probability is closest to that class's median, plus the record closest to the threshold. CKD "
              f"record {int(c_ckd.record_id)} ({{fig:wf_ckd}}) reaches P(CKD) = {c_ckd.p_ckd:.3f} mainly through "
              f"{_contribs(c_ckd, 3)}. The typical non-CKD record {int(c_non.record_id)} is pushed down to "
              f"{c_non.p_ckd:.3f} by {_contribs(c_non, 3)}. For the least certain record, {int(c_bor.record_id)} "
              f"({{fig:wf_bor}}; true non-CKD, P = {c_bor.p_ckd:.3f}), low creatinine and a specific gravity of "
              "1.020 push the output down, but its missing haemoglobin, imputed with the training median of "
              f"{hemo_median:g} g/dL, pushes it towards CKD. That median is low because CKD records dominate the "
              "training set, which shows how median imputation interacts with informative missingness. The "
              f"selected {wname} made no test error, so there is no misclassified record to explain."))
    add(("h2", "Comparison with Clinical Literature"))
    add(("p", "The leading features agree with clinical knowledge. Serum creatinine is the basis of estimated GFR "
              "[@levey2009], and reduced GFR is one of the two criteria that define CKD [@kdigo2024]. Albumin in "
              "urine is the marker of kidney damage behind the albuminuria categories [@levey2005;kdigo2024]. Low "
              "haemoglobin, packed cell volume and red cell count reflect anaemia, which becomes more common as CKD "
              "advances [@stauffer2014]. Hypertension and diabetes are leading risk factors for CKD "
              "[@gbd2023;webster2017]. Specific gravity is not part of the definition of CKD [@kdigo2024]; its high "
              "rank matches the earlier SHAP analysis of this dataset by Chouit et al. [@chouit2026] and may reflect "
              "a reduced ability to concentrate urine, although these data cannot test that explanation."))
    add(("fig", FIG + f"shap_waterfall_typical_ckd_{int(c_ckd.record_id)}.png", "wf_ckd",
         f"SHAP waterfall plot for typical CKD test record {int(c_ckd.record_id)}. Red bars push the output towards "
         "CKD, blue bars towards non-CKD; the bars add up exactly from the baseline to f(x)."))
    add(("fig", FIG + f"shap_waterfall_borderline_{int(c_bor.record_id)}.png", "wf_bor",
         f"SHAP waterfall plot for non-CKD test record {int(c_bor.record_id)}, the prediction closest to the "
         "threshold. Haemoglobin was missing and imputed."))
    add(("h2", "Limits of the Interpretation"))
    add(("p", "SHAP explains how this model uses its inputs, not how kidney disease arises, and post-hoc "
              "explanations should not be over-trusted [@rudin2019]. Correlated features (haemoglobin, packed cell "
              "volume, red cell count) share credit in model-dependent ways, the interventional background treats "
              "features as independent, and imputed values are attributed as if measured. Agreement with the "
              "literature suggests the model relies on plausible signals; it does not validate it clinically."))

    add(("h1", "Demonstration System"))
    add(("p", "A five-page Streamlit application loads the saved pipeline and SHAP background and never retrains. "
              "Its form lists the 24 inputs with units and allowed categories, rejects impossible values, imputes "
              "blank fields exactly as the pipeline does and reports them, and warns about values outside the "
              "training range. It shows the CKD probability, the predicted class and a waterfall explanation whose "
              "baseline plus contributions is checked against the displayed probability. Every page states that "
              "the tool is an educational prototype, not a diagnostic device; no treatment advice is given."))

    add(("h1", "Limitations and Conclusion"))
    add(("h2", "Limitations"))
    add(("bullets", [
        "Data: 400 single-centre records with an apparently healthy non-CKD group, an easier task than real "
        "screening.",
        f"Sample size: {split['n_test']} test records; perfect accuracy is compatible with a true accuracy of "
        f"{tw.accuracy_exact_lo * 100:.1f}%.",
        "Labels: no estimated GFR or repeat measurements, so the three-month chronicity criterion cannot be "
        "checked [@kdigo2024].",
        f"Missingness alone predicts the class (CV ROC-AUC {R['probe_auc']:.3f}); part of the signal may reflect "
        "testing practice.",
        "Probabilities are uncalibrated and should be read as rankings.",
        "No external validation was performed, which TRIPOD+AI expects before clinical claims [@collins2024].",
    ]))
    add(("h2", "Conclusion and Future Work"))
    add(("p", "A reproducible pipeline compared LR, SVM, RF and XGBoost for CKD prediction on the UCI dataset with "
              "leakage-free preprocessing, SMOTE-NC inside the training folds and a selection rule fixed in "
              f"advance. All four models were nearly perfect in cross-validation; the selected {wname} classified "
              f"all {split['n_test']} test records correctly, and SHAP showed that it relies mainly on serum "
              "creatinine, specific gravity, haemoglobin, albumin and packed cell volume, in agreement with clinical "
              "knowledge and earlier work. The same checks exposed the limits of the dataset: oversampling did not "
              "help, missingness is informative, and the classes are unrealistically separable. Future work should "
              "validate the pipeline on larger, independent cohorts with GFR-based labels, assess calibration, "
              "model missingness explicitly, and compare the stability of explanations across models."))

    add(("h1u", "Acknowledgment"))
    add(("p", "The author thanks Dr. Narendra V G, Professor, and Ms Vibha, Assistant Professor - Senior Scale, "
              "School of Computer Engineering, Manipal Institute of Technology, for their guidance."))
    add(("refs",))
    add(("h1u", "About the Author"))
    add(("p", "Vedank Singh is a fifth-semester B.Tech. (CSE, AI and ML) student at Manipal Institute of "
              "Technology, India."))

    return {"title": TITLE, "author": AUTHOR, "abstract": abstract, "index_terms": index_terms, "blocks": B}
