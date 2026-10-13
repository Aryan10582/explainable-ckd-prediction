# Viva questions and answers

Short, defensible answers about **this** implementation. Numbers come from `results/`.

## Priority questions

**1. Why these four classifiers?**
They cover the main families of supervised learning on tabular data: a linear model (Logistic Regression), a kernel
method (SVM), bagged trees (Random Forest) and boosted trees (XGBoost). They differ in how they draw decision
boundaries and handle interactions, they are the models most often compared on this dataset in the literature, and all
run quickly on a laptop. Logistic Regression is also an interpretable baseline: if it were as good as the others, the
simpler model would be preferable.

**2. Why does preprocessing differ between models?**
In this project the *same* preprocessing pipeline is used for all four, but for different reasons. LR and SVM need
standardised features: LR's penalty and SVM's RBF distance treat all features on one scale, and specific gravity
(1.005–1.025) would otherwise be swamped by white cell count (thousands). Trees do not need scaling, because a split on
a standardised feature is the same split as on the raw value. Scaling is still applied before SMOTE-NC, because SMOTE-NC
finds nearest neighbours by distance. So scaling is harmless for trees and necessary for LR, SVM and the resampler.

**3. How are missing values handled?**
Missing values ("?" in the file, 10.6% of all values) are imputed inside the pipeline using the **training fold only**:
median for the 11 numeric features, most frequent value for the 3 ordinal grades and 10 binary features. Unknown
categories at prediction time are treated as missing. No record was dropped. Three impossible values (sodium 4.5,
potassium 39 and 47 mEq/L) were set to missing before modelling by fixed plausibility limits. In the app, every
imputed field is shown to the user. Limitation: missingness is informative (CKD records miss more values), and median
imputation can push predictions; for example, test record 324's imputed haemoglobin (12.6 g/dL) pushes it towards CKD.

**4. Class imbalance and SMOTE/SMOTE-NC?**
The training set is 200 CKD : 120 non-CKD, so the minority class is *non-CKD*. SMOTE creates synthetic minority
records by interpolating between a record and one of its k nearest minority neighbours. Interpolating binary columns
would create impossible values such as 0.4 hypertension, so SMOTE-NC is used: it interpolates only continuous features
and gives each categorical feature the most common value among the neighbours. The ordinal grades were treated as
categorical too, so synthetic records always have valid levels. SMOTE-NC runs only on training folds. Honest result:
it gave **no consistent benefit** (recall and F1 changed by at most 0.01, within one fold SD); it was kept because the
synopsis specified it, but no improvement is claimed. Resampling also makes predicted probabilities uncalibrated.

**5. How is leakage prevented?**
(a) The 80-record test set is split off first, locked with a checksum and never used before final evaluation.
(b) Imputation, scaling, encoding, SMOTE-NC and the classifier are one imbalanced-learn pipeline, so in every CV fold
they are fitted on the fold's training part only. (c) Resampling happens only during `fit`, never on validation or test
records. (d) The record ID and target are excluded from the predictors. (e) The model is chosen from training-set CV and
the selection is saved before the test set is read. An automated audit checks 11 of these properties (all passed), and
training on shuffled labels gives CV ROC-AUC 0.505, i.e. chance, as expected without leakage.

**6. Why are both cross-validation and an untouched test set needed?**
They answer different questions. Cross-validation on the training set is used to *make choices* (hyperparameters,
which model). Any data used to make choices gives an optimistic estimate of performance, because the choice was
fitted to it. The untouched test set is used *once*, after all choices are frozen, to give an unbiased estimate of how
the final pipeline performs on new records. The synopsis suggested choosing the model by test score; that would have
turned the test set into a second validation set and biased the result upwards.

**7. Accuracy versus recall, F1 and ROC-AUC?**
Accuracy counts all correct predictions, so with 62.5% CKD a useless "always CKD" model already scores 62.5%.
Recall (sensitivity) is the share of CKD patients found: the key metric for screening. Precision is the share of CKD
predictions that are correct. F1 is the harmonic mean of precision and recall. ROC-AUC measures how well the
model *ranks* CKD above non-CKD across all thresholds (1.0 = perfect ranking, 0.5 = chance). This project selected on
ROC-AUC with recall and F1 as tie-breakers, and reports all of them.

**8. False positives and false negatives?**
A false negative is a CKD patient predicted as non-CKD: a missed case, who might not get follow-up tests and treatment
to slow progression; this is the more harmful error in screening. A false positive is a person without CKD flagged as
CKD: it causes worry and extra tests (e.g. repeat creatinine, urine albumin), which are inexpensive and confirmatory.
On the test set, LR made 2 false negatives (records 102 and 224) and no false positives; the other models made none.

**9. How was the winner selected?**
By a rule written into `config.yaml` before any score was computed: highest mean 5-fold CV ROC-AUC; models within
0.005 of the best are tied and ranked by CV recall, then F1, then simplicity. All four had CV ROC-AUC between 0.9988 and
1.0000, so all were tied; Random Forest had the highest CV recall (1.000) and F1 (0.995) and was selected. SVM had the
highest ROC-AUC (1.0000) but lower recall (0.985). The 0.005 tolerance reflects the metric's resolution: a validation
fold has about 40 × 24 CKD/non-CKD pairs, so one mis-ordered pair changes its AUC by about 0.001.

**10. How does SHAP work, and what is the baseline?**
SHAP gives each feature a Shapley value: its average contribution to the prediction over all orders in which features
could be "revealed" to the model. A feature's contribution is the change in the expected model output when its true
value is added to a subset of known features, with the unknown features filled from a background set. Here the
background is 100 real training records and the output is the Random Forest's probability of CKD. The **baseline**
φ0 = 0.667 is the average predicted probability over that background, i.e. the prediction before looking at the
patient. For every record, baseline + sum of contributions = the model's probability (checked to 1.8 × 10⁻⁸).
Positive values push towards CKD, negative values towards non-CKD.

**11. Global versus local explanations?**
A local explanation describes one prediction (a waterfall plot for one patient: which features moved *this* probability
and by how much). A global explanation summarises the model's behaviour over many records: mean absolute SHAP value per
feature (importance) and the beeswarm (direction and spread). Global importance here: serum creatinine, specific
gravity, haemoglobin, albumin grade and packed cell volume (63.7% of total attribution); the ranking on the training set
is almost the same (Spearman 0.995).

**12. Why is SHAP not causal evidence?**
SHAP explains how the *model* uses its inputs, not how the disease works. A feature can be important because it is
correlated with the true cause, because of how the data were collected (e.g. which tests were ordered), or because the
model happened to split on it instead of a correlated feature (haemoglobin, packed cell volume and red cell count share
credit). The interventional background also combines feature values independently. Changing a patient's creatinine in
the app changes the *prediction*, not their kidney function. Agreement with clinical knowledge only shows the model is
using plausible signals.

**13. Why does strong test performance not establish clinical validity?**
(a) Only 80 test records: with 80/80 correct, the exact 95% interval for accuracy is 0.955–1.000. (b) One small,
single-centre dataset from about two months; patients elsewhere differ. (c) The non-CKD group looks like healthy
controls (no hypertension, diabetes or any abnormal finding), which makes the task much easier than real screening,
where many people without CKD have such conditions. (d) Missingness alone predicts the class (CV ROC-AUC 0.864).
(e) The labels are not defined by eGFR and chronicity. (f) There was no external validation, calibration study or
prospective evaluation, which guidance such as TRIPOD+AI expects before clinical use.

**14. How does a new patient record reach the saved model and the explanation?**
Form → `validate_record()` (allowed categories, plausibility limits, blanks become missing) → one-row table in the
training column order → the saved pipeline's preprocessing with *training* medians, modes and scaling → Random Forest
probability → label at 0.5 → SHAP with the saved 100-record training background → waterfall in clinical units, with
imputed values marked and a check that baseline + contributions equals the displayed probability. Nothing is
retrained; SMOTE-NC is not used at prediction time.

## Likely follow-up questions

**Why did the test scores come out perfect? Is something wrong?**
That was checked: no shared records or IDs between partitions, imputation statistics equal training-only values,
resampler not applied at prediction, shuffled labels give 0.505, and a full rerun is byte-identical. The data are
simply very separable (serum creatinine IQR 1.4–4.6 vs 0.7–1.1 mg/dL) and other studies on this dataset report similar
accuracy. Even so, one non-CKD record (324) scored 0.497 under XGBoost and 0.397 under RF: close to a mistake.

**Why stratified splitting and 5 folds?**
Stratification keeps the 62.5/37.5 class ratio in every partition and fold, so no fold is unusually easy or hard.
Five folds leave about 64 records (≈40 CKD, ≈24 non-CKD) per validation fold, enough for stable metrics with 320
records, at modest cost.

**Why median and mode imputation, not something smarter?**
They are simple, robust to outliers, learned only from training data, and easy to explain. More complex imputation
(k-NN, iterative) adds variance on 320 records and would not fix the main issue, informative missingness. Future work
could add missing-value indicators explicitly.

**What does the SVM calibration do?**
SVC outputs decision scores, not probabilities. `CalibratedClassifierCV(SVC, method="sigmoid", ensemble=False)` fits a
logistic (Platt-type) mapping from scores to probabilities using an internal 5-fold split of the training data, then
refits one SVC on all training data. It never sees validation or test records.

**What was the SHAP additivity problem?**
At first, base value + SHAP values differed from the Random Forest's probability by up to 0.0065. The cause: scikit-learn
compares float32 inputs with float64 split thresholds, while the SHAP library routed a few background records that lay
within float32 rounding of a threshold differently. SMOTE-NC creates near-duplicate values, so such near-ties occur.
The explainer now uses a copy of the forest with thresholds rounded down to float32, which provably leaves every
prediction unchanged; the error dropped to 1.8 × 10⁻⁸. This is checked automatically for every record.

**Why is specific gravity so important? It is not a CKD criterion.**
True; it is not in the KDIGO definition. In this dataset CKD records have lower specific gravity (median 1.015 vs
1.020), and an earlier SHAP study on this dataset also ranked it in the top three. It may reflect a reduced ability to
concentrate urine, but these data cannot confirm that; it is reported as a model association.

**Why are the probabilities not risks?**
The model was trained on SMOTE-NC-balanced data (non-CKD oversampled to 50%), so its probabilities are shifted relative
to any real population's prevalence, and calibration was never assessed on an external population. They are useful
for ranking within this dataset only.

**What would you do with more time?**
External validation on a larger, independent cohort with eGFR- and albuminuria-based labels; calibration analysis;
explicit modelling of missingness; comparing explanation stability across models; reporting following TRIPOD+AI.
