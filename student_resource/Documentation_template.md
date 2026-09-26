# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Complete before submission]

**Team Members:** [Complete before submission]
**Submission Date:** 2026-09-26

---

## 1. Executive Summary
This solution treats entity resolution as a two-stage, one-to-many linking task. A disk-backed multi-key blocker produces the exact candidate set passed to an explainable pairwise matcher; the matcher then applies a threshold tuned for the official macro F_0.5 metric. The implementation uses only the supplied TSV files and does not perform any external lookup or data augmentation.

---

## 2. Methodology

### 2.1 Problem Analysis
The training data contains 2,206,821 S1 rows, 5,034,616 S2 rows and 5,285,603 S3 rows. It has 7,638,365 positive links; 123,247 S1 records are labelled singletons. Address is missing for 168,967 S2 rows and 175,916 S3 rows, so the pipeline must not require it. All normalization is Unicode NFKC plus case folding; it deliberately preserves multilingual text rather than transliterating it. Country is an open string field, so unseen France records pass through with no country whitelist.

### 2.2 Solution Strategy
*Outline your high-level approach.*

**Approach Type:** Blocking + pairwise classifier
**Core Innovation:** A small, auditable union of exact and country-aware normalized name/address blocks, recorded exactly in `candidate_pairs.tsv` before model inference.

---

## 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

- **Blocking keys used:** Exact normalized name, exact normalized address, exact name+address, normalized-name prefix, meaningful normalized name tokens, and address-number tokens, each country-aware.
- **Candidate pairs generated:** Determined at final test inference; report the exact file count and average candidates per S1 after running the reproducible command.
- **How you ensured true matches were not lost:** On two deterministic, disjoint 1,500-S1 holdouts, exact-only blocking recovered 49.84% and 49.50% of labelled links. The selected multi-key union recovered 98.32% and 97.73%, respectively. Common keys are excluded rather than arbitrarily truncated; final candidates are deterministically ranked and capped before inference.

---

## 4. Matching Model

**Features used:**
- Name features: sequence similarity, token-set similarity, Jaccard overlap and exact normalized-name indicator.
- Address features: sequence similarity, token-set similarity, Jaccard overlap, exact normalized-address indicator and numeric-token overlap.
- Other: country equality and name/address missingness indicators.

**Model type:** Dependency-free logistic regression trained with deterministic SGD. XGBoost is retained as an optional experiment only; it must not replace this model without an improvement on untouched held-out S1 entities.
**Threshold selection method:** Select the threshold maximizing macro F_0.5 on a deterministic development holdout. The measured selected threshold was 0.470.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.996884 on a separate 1,500-S1 labelled-pair holdout. This is an internal estimate, not a leaderboard score.
- **Common false positives (wrong merges):** Must be inspected from the production candidate set; similar businesses that share common names or an address number are the expected risk.
- **Common false negatives (missed matches):** Candidates lost when both name and address have substantial independent variation, particularly where address is missing.

---

## 6. Conclusion
The selected multi-key blocker is materially better than exact-only matching while remaining explainable and scalable through SQLite. Final F_0.5 prioritises avoiding false merges, so the decision threshold is learned from holdout data rather than assumed. The full test artifacts must be generated and validated before submitting.

---

## Appendix

### A. Code Artefacts
`preprocessing/normalization.py` applies Unicode-safe normalization. `blocking/keys.py` defines the candidate keys and `blocking/sqlite_index.py` creates the disk-backed test index. `features/pair_features.py` creates the pair features, `models/logistic_matcher.py` trains/scores the selected matcher, and `inference/run_pipeline.py` writes both output TSV files. Exact commands are in `code/business_entity_resolution/README.md`.

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
