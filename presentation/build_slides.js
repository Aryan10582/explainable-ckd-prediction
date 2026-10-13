// Builds presentation/CKD_XAI_presentation.pptx from presentation/slide_data.json.
//   python presentation/export_slide_data.py   (numbers from the saved experiment)
//   node presentation/build_slides.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "slide_data.json"), "utf8"));
const OUT = path.join(__dirname, "CKD_XAI_presentation.pptx");

const THEME = {
  name: "CKD XAI",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2430", lt1: "FFFFFF", dk2: "0F3D4C", lt2: "EAF1F4",
    accent1: "EB6834", accent2: "2A78D6", accent3: "1BAF7A", accent4: "4A3AA7",
    accent5: "E34948", accent6: "5F6B73", hlink: "2A78D6", folHlink: "4A3AA7",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.author = "Vedank Singh";
pres.title = "Explainable Machine Learning for Chronic Kidney Disease Prediction";
pres.subject = "Mini-project presentation";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const HEX = THEME.colors;
const f = (x, d = 3) => Number(x).toFixed(d);

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.2, w: 8.7, h: 2.2, fontFace: THEME.headFontFace,
      fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.55, w: 9.6, h: 0.7, fontSize: 22,
      color: C.accent1, margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.4, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.8, fontFace: THEME.headFontFace,
      fontSize: 30, bold: true, color: C.text2, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "Explainable ML for CKD prediction  |  Vedank Singh", options: { x: 0.6, y: 7.0, w: 8, h: 0.3,
      fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.25, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.8, fontFace: THEME.headFontFace,
      fontSize: 30, bold: true, color: C.background1, valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.25, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.background2, align: "right" },
});

// ------------------------------------------------------------------ helpers
function card(slide, x, y, w, h, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12,
    fill: { color: opts.fill || C.background2 }, line: opts.line ? { color: opts.line, width: 1.5 } : { type: "none" },
    objectName: opts.name || "card" });
}
function badge(slide, x, y, label, color, d = 0.5) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { type: "none" }, objectName: "badge" });
  slide.addText(label, { x, y, w: d, h: d, align: "center", valign: "middle", fontSize: 16, bold: true,
    color: C.background1, margin: 0, isTextBox: true });
}
function stat(slide, x, y, w, value, label, color = C.accent1, size = 40) {
  slide.addText(value, { x, y, w, h: 0.8, fontSize: size, bold: true, color, fontFace: THEME.headFontFace, margin: 0,
    valign: "bottom", isTextBox: true });
  slide.addText(label, { x, y: y + 0.85, w, h: 0.6, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
}
function title(slide, t) { slide.addText(t, { placeholder: "title" }); }
const RF = D.models.find((m) => m.selected);

// ------------------------------------------------------------------ 1. title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addShape(pres.shapes.OVAL, { x: 10.2, y: 0.9, w: 3.6, h: 3.6, fill: { color: C.accent1, transparency: 15 }, line: { type: "none" }, objectName: "motif-1" });
s.addShape(pres.shapes.OVAL, { x: 11.3, y: 3.3, w: 2.4, h: 2.4, fill: { color: C.accent2, transparency: 25 }, line: { type: "none" }, objectName: "motif-2" });
s.addShape(pres.shapes.OVAL, { x: 9.7, y: 4.2, w: 1.3, h: 1.3, fill: { color: C.accent3, transparency: 20 }, line: { type: "none" }, objectName: "motif-3" });
title(s, "Explainable Machine Learning for Chronic Kidney Disease Prediction");
s.addText("SHAP-based feature attribution on the UCI CKD dataset", { placeholder: "body" });
s.addText([
  { text: `${D.student.name}  |  Reg. No. ${D.student.reg}`, options: { fontSize: 20, bold: true, breakLine: true } },
  { text: D.student.program, options: { fontSize: 15, breakLine: true } },
  { text: `Guides: ${D.student.guides}`, options: { fontSize: 15, breakLine: true } },
  { text: `${D.student.school}  |  ${D.student.year}`, options: { fontSize: 15 } },
], { x: 0.8, y: 4.7, w: 9.2, h: 1.9, color: C.background2, margin: 0, valign: "top", paraSpaceAfter: 6, isTextBox: true });
s.addNotes("Introduce the project: an explainable pipeline that predicts CKD from routine clinical data and shows why each prediction was made. Mention it is an educational prototype, not a diagnostic tool.");

// ------------------------------------------------------------------ 2. problem + objectives
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
title(s, "Why CKD prediction needs explanations");
s.addText(`${D.gbd.adults_millions} million`, { x: 0.6, y: 1.35, w: 5.6, h: 0.95, fontSize: 54, bold: true, color: C.accent1,
  fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
s.addText(`adults had CKD in 2023 (age-standardised prevalence ${D.gbd.prevalence_pct}%), GBD 2023 Study`,
  { x: 0.6, y: 2.3, w: 5.6, h: 0.6, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
card(s, 0.6, 3.2, 5.6, 1.6);
s.addText([{ text: "Often silent early", options: { bold: true, fontSize: 16, breakLine: true } },
  { text: "Routine blood and urine tests can flag likely CKD before symptoms appear.", options: { fontSize: 14 } }],
  { x: 0.85, y: 3.35, w: 5.1, h: 1.3, color: C.text1, margin: 0, valign: "top", isTextBox: true });
card(s, 0.6, 5.0, 5.6, 1.6);
s.addText([{ text: "Black boxes are hard to trust", options: { bold: true, fontSize: 16, breakLine: true } },
  { text: "A clinician needs to see why a model flags a patient. SHAP attributes each prediction to the inputs.", options: { fontSize: 14 } }],
  { x: 0.85, y: 5.15, w: 5.1, h: 1.3, color: C.text1, margin: 0, valign: "top", isTextBox: true });
s.addText("Objectives", { x: 6.9, y: 1.35, w: 5.8, h: 0.5, fontSize: 22, bold: true, color: C.text2, margin: 0, isTextBox: true });
const objectives = [
  "Acquire, clean and preprocess the UCI CKD data without leakage: missing values, encoding, scaling, class imbalance",
  "Tune and compare Logistic Regression, SVM, Random Forest and XGBoost on accuracy, precision, recall, F1 and ROC-AUC",
  "Explain the chosen model with SHAP, globally and per patient, and compare its key features with clinical literature",
];
objectives.forEach((t, i) => {
  const y = 2.05 + i * 1.55;
  badge(s, 6.9, y + 0.1, String(i + 1), [C.accent1, C.accent2, C.accent3][i]);
  s.addText(t, { x: 7.6, y, w: 5.1, h: 1.3, fontSize: 16, color: C.text1, margin: 0, valign: "top", isTextBox: true });
});
s.addNotes("Motivation: CKD is common and silent early. ML can screen from routine tests, but without explanations clinicians cannot judge predictions. Read the three objectives from the synopsis.");

// ------------------------------------------------------------------ 3. data
pres.addSection({ title: "Data and method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data and method" });
title(s, "The data: 400 patients, 24 routine clinical inputs");
const tiles = [
  [String(D.data.records), "records (UCI repository, id 336)"],
  [`${D.data.ckd} / ${D.data.nonckd}`, "CKD / non-CKD"],
  ["24", "inputs: 11 numeric, 3 ordinal, 10 binary"],
  [`${D.data.missing_pct}%`, "of values missing"],
];
tiles.forEach(([v, l], i) => {
  card(s, 0.6 + i * 3.05, 1.3, 2.85, 1.55);
  stat(s, 0.8 + i * 3.05, 1.3, 2.5, v, l, [C.accent1, C.accent2, C.accent4, C.accent3][i], 36);
});
card(s, 0.6, 3.1, 5.9, 3.6);
s.addText("Cleaning: nothing deleted", { x: 0.85, y: 3.25, w: 5.4, h: 0.45, fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true });
s.addText([
  { text: "Official archive cached with SHA-256; matches UCI's CSV cell for cell", options: { bullet: true, breakLine: true } },
  { text: `${D.data.line_repairs} malformed lines and ${D.data.whitespace} stray tab/space cells repaired`, options: { bullet: true, breakLine: true } },
  { text: `${D.data.implausible} impossible values set to missing (sodium 4.5; potassium 39 and 47 mEq/L)`, options: { bullet: true, breakLine: true } },
  { text: "No duplicate patient records; 80-record test set locked before tuning", options: { bullet: true } },
], { x: 0.85, y: 3.8, w: 5.45, h: 2.8, fontSize: 15, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 8, isTextBox: true });
s.addImage({ path: D.fig.missing, x: 6.8, y: 3.1, w: 5.9, h: 5.9 * 1.65 / 3.25, altText: "Missing values per record by class" });
s.addText(`In training data, CKD records miss far more values, and no non-CKD record shows any abnormal finding: the classes are unusually easy to separate (creatinine median ${D.data.sc_ckd} vs ${D.data.sc_non} mg/dL).`,
  { x: 6.8, y: 6.15, w: 5.9, h: 0.75, fontSize: 13, color: C.accent6, margin: 0, valign: "top", isTextBox: true });
s.addNotes("The UCI file has formatting errors that were fixed by explicit rules, all logged. Missing values are common and strongly class-dependent, which matters later. Non-CKD records look like healthy controls.");

// ------------------------------------------------------------------ 4. workflow + leakage
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data and method" });
title(s, "Pipeline designed so the test set stays untouched");
s.addImage({ path: D.fig.workflow, x: 0.7, y: 1.2, w: 3.3, h: 3.3 * 5.6 / 3.25, altText: "System workflow diagram" });
s.addText("How leakage is prevented", { x: 4.6, y: 1.3, w: 8.1, h: 0.5, fontSize: 22, bold: true, color: C.text2, margin: 0, isTextBox: true });
const checks = [
  `Test set (${D.data.test} records) split off and locked, with a checksum, before any tuning`,
  "Imputation, scaling and SMOTE-NC are refitted inside every training fold",
  "SMOTE-NC runs only during fitting: validation and test records are never resampled",
  "The model is chosen from training-set cross-validation; the test set is used once",
  "Record ID and target are excluded from the predictors",
  `Automated audit: ${D.checks.audit_passed} / ${D.checks.audit_total} checks passed; shuffled labels give CV ROC-AUC ${f(D.checks.perm_auc)}`,
];
checks.forEach((t, i) => {
  const y = 2.0 + i * 0.8;
  s.addShape(pres.shapes.OVAL, { x: 4.6, y: y + 0.08, w: 0.36, h: 0.36, fill: { color: C.accent3 }, line: { type: "none" }, objectName: "check" });
  s.addText("✓", { x: 4.6, y: y + 0.08, w: 0.36, h: 0.36, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText(t, { x: 5.15, y, w: 7.6, h: 0.6, fontSize: 16, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
});
s.addNotes("Walk down the workflow. The key point is that every learned step lives inside one pipeline that is refitted on each training fold, so nothing learned from validation or test data can leak in. SMOTE-NC creates synthetic non-CKD records only during fitting.");

// ------------------------------------------------------------------ 5. models + selection
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data and method" });
title(s, "Four models, one selection rule fixed in advance");
D.models.forEach((m, i) => {
  const x = 0.6 + i * 3.05;
  card(s, x, 1.3, 2.85, 1.25, m.selected ? { line: C.accent3 } : {});
  s.addText([{ text: m.name + (m.selected ? "  (selected)" : ""), options: { bold: true, fontSize: 16, breakLine: true, color: m.selected ? C.accent3 : C.text2 } },
    { text: m.params, options: { fontSize: 13, color: C.text1 } }],
    { x: x + 0.2, y: 1.42, w: 2.5, h: 1.05, margin: 0, valign: "top", isTextBox: true });
});
const hdr = { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 14, align: "center" };
const rows = [[{ text: "5-fold CV (mean ± SD)", options: { ...hdr, align: "left" } }, { text: "ROC-AUC", options: hdr },
  { text: "Recall", options: hdr }, { text: "F1-score", options: hdr }]];
D.models.forEach((m) => {
  const o = { fontSize: 14, align: "center", bold: m.selected, fill: { color: m.selected ? C.background2 : C.background1 } };
  rows.push([{ text: m.name, options: { ...o, align: "left" } }, { text: `${f(m.cv_auc)} ± ${f(m.cv_auc_sd)}`, options: o },
    { text: `${f(m.cv_recall)} ± ${f(m.cv_recall_sd)}`, options: o }, { text: `${f(m.cv_f1)} ± ${f(m.cv_f1_sd)}`, options: o }]);
});
s.addTable(rows, { x: 0.6, y: 2.95, w: 7.5, colW: [2.4, 1.7, 1.7, 1.7], rowH: 0.55, border: { type: "solid", pt: 0.5, color: "D5DEE3" },
  color: C.text1, valign: "middle" });
card(s, 8.4, 2.95, 4.3, 2.75);
s.addText("Rule (written before any score)", { x: 8.6, y: 3.1, w: 3.9, h: 0.45, fontSize: 17, bold: true, color: C.text2, margin: 0, isTextBox: true });
s.addText([
  { text: "Highest mean CV ROC-AUC wins", options: { bullet: { type: "number" }, breakLine: true } },
  { text: "Within 0.005: tied, ranked by recall, then F1", options: { bullet: { type: "number" }, breakLine: true } },
  { text: `All four tied (${f(Math.min(...D.models.map((m) => m.cv_auc)), 4)}–${f(Math.max(...D.models.map((m) => m.cv_auc)), 4)}): ${D.winner} wins on recall (${f(RF.cv_recall)})`, options: { bold: true, color: C.accent3 } },
], { x: 8.6, y: 3.65, w: 3.95, h: 1.95, fontSize: 15, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 8, isTextBox: true });
s.addText("The synopsis proposed picking the best test score; selecting on training CV keeps the test estimate unbiased.",
  { x: 0.6, y: 5.95, w: 12.1, h: 0.6, fontSize: 14, italic: true, color: C.accent6, margin: 0, valign: "top", isTextBox: true });
s.addNotes("All four were tuned by grid search with stratified 5-fold CV on the 320 training records. The selection rule was fixed in the config file before running. The synopsis said 'pick the best on the test set'; that biases the test estimate, so selection used CV only.");

// ------------------------------------------------------------------ 6. test results
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
title(s, "Untouched test set: near-perfect, with honest uncertainty");
const testTiles = [
  [`${D.data.test} / ${D.data.test}`, `test records correct for ${D.winner} (also SVM and XGBoost)`, C.accent3],
  [`${f(D.rf_test.acc_lo)}–${f(D.rf_test.acc_hi)}`, "exact 95% CI for Random Forest accuracy", C.accent2],
  [String(D.models.find((m) => m.key === "logistic_regression").fn), "CKD cases missed by Logistic Regression (no FP)", C.accent1],
];
testTiles.forEach(([v, l, c], i) => {
  card(s, 0.6, 1.3 + i * 1.8, 4.6, 1.6);
  stat(s, 0.85, 1.3 + i * 1.8, 4.2, v, l, c, 36);
});
s.addImage({ path: D.fig.scores, x: 5.6, y: 1.3, w: 7.1, h: 7.1 * 2.15 / 3.25, altText: "Predicted CKD probabilities by true class" });
s.addText(`Perfect accuracy hides a near miss: non-CKD record ${D.near_miss.id} scored ${f(D.near_miss.rf)} (Random Forest) and ${f(D.near_miss.xgb)} (XGBoost), just below the 0.5 threshold.`,
  { x: 5.6, y: 6.1, w: 7.1, h: 0.75, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
s.addNotes("Explain FN vs FP: a false negative is a missed CKD patient, the more harmful error. With only 80 test records, 80/80 still allows a true accuracy as low as about 95.5%. Perfect scores are not clinical validation.");

// ------------------------------------------------------------------ 7. honest checks
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
title(s, "Checks that keep the results honest");
const maxAbl = Math.max(...D.models.map((m) => Math.max(Math.abs(m.abl_recall), Math.abs(m.abl_f1))));
const honest = [
  ["No gain", "SMOTE-NC vs no resampling", `Same folds and settings: recall and F1 changed by at most ${f(maxAbl, 3)}, within one fold SD. Kept as specified, no benefit claimed.`, C.accent6],
  [f(D.checks.probe_auc), "Missingness alone", "CV ROC-AUC from which values are missing, ignoring the values: testing practice carries class signal.", C.accent1],
  [f(D.checks.perm_auc), "Shuffled labels", "CV ROC-AUC after permuting the labels: chance level, so the pipeline has no hidden leak.", C.accent2],
  ["0%", "Abnormal findings in non-CKD", "No non-CKD training record has hypertension, diabetes, anaemia or any other abnormal finding.", C.accent4],
];
honest.forEach(([v, h, t, c], i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 1.3 + Math.floor(i / 2) * 2.85;
  card(s, x, y, 5.95, 2.6);
  s.addText(v, { x: x + 0.3, y: y + 0.2, w: 2.3, h: 0.9, fontSize: 40, bold: true, color: c, fontFace: THEME.headFontFace, margin: 0, valign: "middle", isTextBox: true });
  s.addText(h, { x: x + 2.7, y: y + 0.25, w: 3.0, h: 0.8, fontSize: 17, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
  s.addText(t, { x: x + 0.3, y: y + 1.2, w: 5.35, h: 1.25, fontSize: 15, color: C.text1, margin: 0, valign: "top", isTextBox: true });
});
s.addNotes("These checks matter more than the headline numbers. SMOTE-NC did not help, consistent with Yang et al. 2024. Missingness is informative. Shuffled labels collapse to 0.5, showing no leakage. The data are simply very separable.");

// ------------------------------------------------------------------ 8. global SHAP
pres.addSection({ title: "Explainability" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Explainability" });
title(s, "What drives the predictions: global SHAP");
s.addImage({ path: D.fig.beeswarm, x: 0.6, y: 1.2, w: 5.6 * 3.34 / 3.21, h: 5.6, altText: "SHAP beeswarm summary plot" });
const top = D.shap.top;
s.addChart(pres.charts.BAR, [{ name: "Share of attribution (%)", labels: top.map((t) => t.label).reverse(), values: top.map((t) => Number(t.share.toFixed(1))).reverse() }], {
  x: 6.7, y: 1.2, w: 6.0, h: 3.0, barDir: "bar", chartColors: [HEX.accent2], showValue: true, dataLabelFormatCode: "0.0",
  dataLabelFontSize: 12, dataLabelColor: HEX.dk1, catAxisLabelFontSize: 13, valAxisHidden: true, valGridLine: { style: "none" },
  catGridLine: { style: "none" }, showLegend: false, showTitle: true, title: "Share of total |SHAP| attribution, test set (%)",
  titleFontSize: 14, titleColor: HEX.dk2, catAxisLabelColor: HEX.dk1, catAxisLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  dataLabelFontFace: "+mn-lt", barGapWidthPct: 60,
});
s.addText([
  { text: "Creatinine: basis of eGFR, which defines CKD", options: { bullet: true, breakLine: true } },
  { text: "Albumin: marker of kidney damage", options: { bullet: true, breakLine: true } },
  { text: "Haemoglobin, packed cell volume: anaemia, common as CKD advances", options: { bullet: true, breakLine: true } },
  { text: "Hypertension, diabetes: leading CKD risk factors", options: { bullet: true, breakLine: true } },
  { text: "Specific gravity: not a diagnostic criterion; matches earlier SHAP work on this dataset", options: { bullet: true } },
], { x: 6.7, y: 4.4, w: 6.0, h: 2.0, fontSize: 14, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 4, isTextBox: true });
s.addText(`Ranking stable between test and training records (Spearman ${f(D.shap.spearman)}). SHAP explains the model, not the disease.`,
  { x: 6.7, y: 6.45, w: 6.0, h: 0.45, fontSize: 12, italic: true, color: C.accent6, margin: 0, isTextBox: true });
s.addNotes("The top five features carry about 64% of all attribution. Each dot in the beeswarm is a test patient: red means a high feature value. High creatinine and low specific gravity or haemoglobin push towards CKD. These agree with clinical knowledge, but SHAP is not causal evidence.");

// ------------------------------------------------------------------ 9. local SHAP
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Explainability" });
title(s, "Why one patient got their prediction: local SHAP");
const wfH = 4.55, wfW = wfH * 3.25 / 2.72;
s.addImage({ path: D.fig.wf_ckd, x: 0.6, y: 1.2, w: wfW, h: wfH, altText: "Waterfall plot for a CKD record" });
s.addImage({ path: D.fig.wf_border, x: 12.7 - wfW, y: 1.2, w: wfW, h: wfH, altText: "Waterfall plot for the borderline record" });
s.addText([{ text: `Record ${D.shap.local_ckd.id} (true CKD), P = ${f(D.shap.local_ckd.p)}`, options: { bold: true, breakLine: true } },
  { text: "High creatinine, low specific gravity and low haemoglobin push the output up from the 0.667 baseline.", options: {} }],
  { x: 0.6, y: 5.9, w: 5.9, h: 1.0, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
s.addText([{ text: `Record ${D.shap.local_border.id} (true non-CKD), P = ${f(D.shap.local_border.p)}`, options: { bold: true, breakLine: true } },
  { text: "Normal creatinine pushes down, but missing haemoglobin, imputed at the training median 12.6 g/dL, pushes up.", options: {} }],
  { x: 6.8, y: 5.9, w: 5.9, h: 1.0, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
s.addNotes(`Read a waterfall bottom to top: start at the baseline (mean prediction over 100 training records), add each feature's contribution, end at the model output. The contributions add up exactly (max error ${D.shap.additivity.toExponential(1)}). The second example shows how imputation can influence an explanation.`);

// ------------------------------------------------------------------ 10. demo
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Explainability" });
title(s, "Demonstration: a local Streamlit prototype");
const shotW = 6.2, shotH = shotW * 450 / 800;
s.addImage({ path: D.fig.app_prediction_top, x: 0.6, y: 1.3, w: 5.7, h: 5.7 * 184 / 720, altText: "App screenshot: prediction for test record 26" });
s.addImage({ path: D.fig.app_explanation, x: 12.7 - shotW, y: 1.3, w: shotW, h: shotH, altText: "App screenshot: SHAP explanation for test record 26" });
const steps = ["Validate inputs against the schema", "Impute blanks and list them", "Predict P(CKD) with the saved model",
  "Explain the prediction with SHAP", "Check baseline + contributions = output"];
steps.forEach((t, i) => {
  const y = 3.0 + i * 0.62;
  badge(s, 0.6, y, String(i + 1), [C.accent2, C.accent4, C.accent1, C.accent3, C.accent6][i], 0.45);
  s.addText(t, { x: 1.2, y: y - 0.05, w: 5.1, h: 0.55, fontSize: 15, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
});
s.addText(`Screenshots from the running app (dataset record 26). It loads saved artifacts only: about ${D.runtime.predict_ms} ms per prediction and ${D.runtime.explain_ms} ms per explanation. Every page states that it is an educational prototype, not a diagnostic tool, with no treatment advice.`,
  { x: 6.5, y: 5.0, w: 6.2, h: 1.4, fontSize: 14, color: C.accent6, margin: 0, valign: "top", isTextBox: true });
s.addNotes("Live demo: run 'streamlit run app/streamlit_app.py', load dataset record 26, press Predict and explain, show the probability, the imputed field and the waterfall, then point to the consistency check line. Screenshots are from the running app.");

// ------------------------------------------------------------------ 11. conclusion
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Conclusion" });
title(s, "Conclusion, limitations and next steps");
s.addText("What we found", { x: 0.6, y: 1.35, w: 5.9, h: 0.5, fontSize: 20, bold: true, color: C.accent1, margin: 0, isTextBox: true });
s.addText([
  { text: "A reproducible, leakage-audited pipeline for four classifiers", options: { bullet: true, breakLine: true } },
  { text: `${D.winner} chosen by a predeclared CV rule; ${D.data.test}/${D.data.test} test records correct`, options: { bullet: true, breakLine: true } },
  { text: "SHAP: creatinine, specific gravity, haemoglobin, albumin, packed cell volume", options: { bullet: true, breakLine: true } },
  { text: "Oversampling gave no consistent benefit", options: { bullet: true } },
], { x: 0.6, y: 1.95, w: 5.9, h: 3.2, fontSize: 16, color: C.background1, margin: 0, valign: "top", paraSpaceAfter: 10, isTextBox: true });
s.addText("Limits", { x: 6.9, y: 1.35, w: 5.8, h: 0.5, fontSize: 20, bold: true, color: C.accent1, margin: 0, isTextBox: true });
s.addText([
  { text: "400 single-centre records; healthy-looking controls", options: { bullet: true, breakLine: true } },
  { text: "80 test records: wide confidence intervals", options: { bullet: true, breakLine: true } },
  { text: "No eGFR or repeat tests; missingness is informative", options: { bullet: true, breakLine: true } },
  { text: "Probabilities uncalibrated; SHAP is not causal", options: { bullet: true, breakLine: true } },
  { text: "No external validation: not clinically validated", options: { bullet: true } },
], { x: 6.9, y: 1.95, w: 5.8, h: 3.2, fontSize: 16, color: C.background1, margin: 0, valign: "top", paraSpaceAfter: 10, isTextBox: true });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.45, w: 12.1, h: 1.2, rectRadius: 0.12, fill: { color: C.background1, transparency: 88 }, line: { type: "none" }, objectName: "next-steps" });
s.addText([{ text: "Next steps: ", options: { bold: true, color: C.accent1 } },
  { text: "external validation on larger cohorts with eGFR-based labels, calibration, explicit missingness modelling, and explanation stability across models.", options: { color: C.background1 } }],
  { x: 0.85, y: 5.55, w: 11.6, h: 1.0, fontSize: 16, margin: 0, valign: "middle", isTextBox: true });
s.addNotes("Close with the honest message: the pipeline is sound and the explanations are clinically sensible, but the dataset is too small and too easy to support clinical claims. Invite questions.");

// ------------------------------------------------------------------ write + theme colours
async function applyThemeColors(file) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const name = "ppt/theme/theme1.xml";
  let xml = await zip.file(name).async("string");
  for (const [slot, hex] of Object.entries(THEME.colors)) {
    xml = xml.replace(new RegExp(`<a:${slot}>[\\s\\S]*?</a:${slot}>`), `<a:${slot}><a:srgbClr val="${hex}"/></a:${slot}>`);
  }
  xml = xml.replace(/<a:clrScheme name="[^"]*">/, `<a:clrScheme name="${THEME.name}">`);
  zip.file(name, xml);
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyThemeColors(OUT);
  console.log(`wrote ${OUT}`);
})();
