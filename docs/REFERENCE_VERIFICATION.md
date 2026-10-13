# Reference verification

Every reference used in the IEEE paper was checked against the registrar or publisher record, not
copied from memory or from the synopsis. Raw evidence:

- `docs/references/doi_metadata.json`: Crossref / DataCite metadata for every DOI (`python scripts/verify_references.py`)
- `docs/references/abstracts_europepmc.json`: abstracts from Europe PMC, Semantic Scholar and OpenAlex, used
  to describe the related studies accurately

Checked on 13 October 2026.

## Synopsis references

| # | Reference (short) | DOI resolves | Details checked | Action |
|---|---|---|---|---|
| 1 | GBD 2023 CKD Collaborators, *Lancet* 406(10518):2461–2482, 2025 | yes | yes | used; 788 million / 14.2 % figures taken from the abstract |
| 2 | Levey et al., KDIGO position statement, *Kidney Int.* 67(6):2089–2100, 2005 | yes | yes | used |
| 3 | Lundberg & Lee, NeurIPS 2017 (SHAP) | no DOI | title, authors, volume 30 confirmed on proceedings.neurips.cc | used **without page numbers**: the official BibTeX lists none, so the synopsis' "pp. 4765–4774" could not be confirmed |
| 4 | Ribeiro et al., LIME, KDD 2016, pp. 1135–1144 | yes | yes | used |
| 5 | Chen & Guestrin, XGBoost, KDD 2016, pp. 785–794 | yes | yes | used |
| 6 | Breiman, Random forests, *Mach. Learn.* 45:5–32, 2001 | yes | yes | used; issue number (no. 1) added |
| 7 | Chawla et al., SMOTE, *JAIR* 16:321–357, 2002 | yes | yes | used (SMOTE-NC is defined in this paper) |
| 8 | Loh et al., XAI in healthcare review, *CMPB* 226:107161, 2022 | yes | yes | used |
| 9 | Rezk et al., *Bioengineering* 12(4):356, 2025 | yes | yes | used |
| 10 | Dharmarathne et al., *Intell. Syst. Appl.* 22:200397, 2024 | yes | yes | used |
| 11 | Ghosh & Khandoker, *Sci. Rep.* 14:3687, 2024 | yes | yes | used; note: uses a UAE hospital cohort, not the UCI data |
| 12 | Singamsetty et al., *PeerJ Comput. Sci.* 10:e2291, 2024 | yes | yes | used |
| 13 | Chouit et al., *PLOS One* 21(2):e0343205, 2026 | yes | yes | used |
| 14 | Phuong et al., *Bioinform. Biol. Insights* 19, 2025 | yes | article number 11779322251356563 | used; synopsis "pp. 1–18" replaced by the article number (pages not in the registrar record) |
| 15 | Yang et al., *IEEE Access* 12:171205–171220, 2024 | yes | yes | used |
| 16 | Charleonnan et al., MITicon 2016, pp. MIT-80–MIT-83 | yes | yes | used |
| 17 | Khalid et al., *Comput. Intell. Neurosci.* 2023:9266889 | yes | yes | used; month omitted (registrar gives print issue date only) |
| 18 | Khan et al., *Front. Artif. Intell.* 6:1339988, 2024 | yes | yes | used; note: uses a Pakistani case-control cohort, not the UCI data |

No synopsis reference was found to be fabricated or wrong in substance; the corrections above are bibliographic.

## References added for the final paper

| Purpose | Reference | Verified via |
|---|---|---|
| Dataset | L. Rubini, P. Soundarapandian, P. Eswaran, *Chronic Kidney Disease*, UCI ML Repository, 2015, doi:10.24432/C5G020 | DataCite + UCI API metadata (three creators) |
| Tree SHAP | Lundberg et al., *Nat. Mach. Intell.* 2(1):56–67, 2020 | Crossref |
| SVM | Cortes & Vapnik, *Mach. Learn.* 20(3):273–297, 1995 | Crossref |
| Logistic regression | Cox, *J. R. Stat. Soc. B* 20(2):215–232, 1958 | Crossref |
| scikit-learn | Pedregosa et al., *JMLR* 12:2825–2830, 2011 | jmlr.org |
| imbalanced-learn | Lemaître et al., *JMLR* 18(17):1–5, 2017 | jmlr.org |
| CKD definition/classification | KDIGO 2024 guideline, *Kidney Int.* 105(4S):S117–S314, 2024 | Crossref |
| CKD clinical overview | Webster et al., *Lancet* 389(10075):1238–1252, 2017 | Crossref |
| Creatinine-based eGFR | Levey et al., CKD-EPI, *Ann. Intern. Med.* 150(9):604–612, 2009 | Crossref |
| Anaemia in CKD | Stauffer & Fan, *PLoS One* 9(1):e84943, 2014 | Crossref |
| Limits of post-hoc explanation | Rudin, *Nat. Mach. Intell.* 1(5):206–215, 2019 | Crossref |
| Data leakage | Kaufman et al., *ACM TKDD* 6(4):1–21, 2012 | Crossref |
| Selection bias in CV | Varma & Simon, *BMC Bioinformatics* 7:91, 2006 | Crossref |
| Reporting / external validation | Collins et al., TRIPOD+AI, *BMJ* 385:e078378, 2024 | Crossref |
| Resampling and calibration | van den Goorbergh et al., *JAMIA* 29(9):1525–1534, 2022 | Crossref |
| Informative missingness | Groenwold, *Diagn. Progn. Res.* 4:8, 2020 | Crossref |

## Note on the college guideline

The mini-project guidelines ask for journal references from 2020 onwards for the problem statement.
The paper still cites the original algorithm papers (several are conference papers from before 2020)
because the methods must be attributed to their sources; the remaining references are mostly recent
peer-reviewed journal articles.
