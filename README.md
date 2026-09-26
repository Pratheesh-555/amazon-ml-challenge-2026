# Amazon ML Challenge 2026 — Business Entity Resolution

Match noisy business records from **Source 2** and **Source 3** to every **Source 1** entity. Source 1 is the deduplicated reference; each S1 record may match zero, one, or many S2/S3 records.

This repo follows the official student-resource layout from **ML 2.pdf** so the final zip can be assembled without reshuffling folders. Challenge-window and artefact rules from **AMZN ML 1.pdf** are summarized at the end.

## Repository layout

```
amazon-ml-challenge-2026/
├── student_resource/                 # Official data + validator + methodology template
│   ├── dataset/
│   │   ├── train/
│   │   │   ├── train_source1.tsv
│   │   │   ├── train_source2.tsv
│   │   │   ├── train_source3.tsv
│   │   │   └── train_ground_truth.tsv
│   │   └── test/
│   │       ├── test_source1.tsv      # generate a row for every entity here
│   │       ├── test_source2.tsv
│   │       └── test_source3.tsv
│   ├── utils/validate_submission.py
│   ├── Documentation_template.md
│   └── README.md                     # full problem statement
├── code/
│   └── business_entity_resolution/   # runnable pipeline (required in the submission zip)
│       ├── src/                      # preprocessing, blocking, evaluation
│       ├── tests/
│       ├── README.md                 # reproduce data → blocking → matching → output
│       └── requirements.txt
├── scripts/
│   └── baseline_v1.py                # current end-to-end test inference
├── output/                           # files the Portal / zip expect
│   ├── matching_results.tsv          # leaderboard file
│   └── candidate_pairs.tsv           # blocking set (required in the final zip)
└── experiments/
    └── results.csv                   # blocking benchmark logs
```

All challenge files are **tab-separated `.tsv`**. Always read and write with `sep="\t"`. Addresses and ID lists contain commas, so a CSV read will silently collapse each row into one column.

## What you must produce

Two files in `output/`:

| File | Role | Columns |
| --- | --- | --- |
| `matching_results.tsv` | **Only file scored on the Portal** | `source1_entity_id`, `matched_entity_ids` |
| `candidate_pairs.tsv` | Blocking set fed to the matcher; reviewed for final ranking | `source1_entity_id`, `candidate_entity_ids` |

Example (`matching_results.tsv` — one tab between columns, commas inside the ID list, no quotes):

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

Rules (submission is rejected if these fail):

- Exactly one row per Source 1 test entity (France included).
- Empty `matched_entity_ids` / `candidate_entity_ids` for singletons / no candidates.
- ID lists are S2-/S3- IDs that exist in the test set; no S1 self-matches; no duplicates in a list or as rows.
- Final matches must be a **subset** of candidates.
- `candidate_pairs.tsv` is the **last** candidate list the matching model actually scores — not an earlier blocking pass.

## Setup

Python 3.10+ recommended. From the **repository root** (`amazon-ml-challenge-2026/`):

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r code/business_entity_resolution/requirements.txt
```

Challenge data already lives under `student_resource/dataset/`. Do not look up businesses via APIs, government registries, geocoders, or other external data (immediate disqualification).

## Run inference (current working path)

A trained pairwise matcher is **not in the repo yet**. The working end-to-end path is the exact-name / exact-address baseline, which writes `output/matching_results.tsv` for the **test** split.

From the repository root:

```bash
python scripts/baseline_v1.py
```

This loads `student_resource/dataset/test/test_source{1,2,3}.tsv`, indexes S2/S3 on normalized name and address, and writes one row per S1 test entity.

Then validate format before uploading (run from `student_resource/` as the PDFs specify):

```bash
python utils/validate_submission.py `
    --matching ../output/matching_results.tsv `
    --candidate ../output/candidate_pairs.tsv `
    --test-dir dataset/test
```

From bash:

```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

`PASS` (exit 0) means the files are safe to submit. The validator does **not** compute F0.5. Add `--check-ids` to verify S2/S3 IDs exist in the test files (uses more RAM).

Portal upload during the contest: **`output/matching_results.tsv` only**. Keep `candidate_pairs.tsv` for the final zip.

## Train / evaluate on labeled data

Ground truth exists only for **train**. The test set has no labels. The PDFs require you to hold out a validation split from train and score it with **macro F0.5**.

Official metric (per Source 1 entity, then averaged over **all** S1 entities, including singletons):

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- Singleton with empty prediction → **1.0**; any predicted match → **0.0**.
- Precision is weighted 2× recall. False merges hurt more than missed links.

Implemented helpers:

- `code/business_entity_resolution/src/evaluation/metrics.py` — `f05_per_entity`, `f05_macro`, `blocking_metrics`
- `code/business_entity_resolution/src/evaluation/benchmark.py` — blocking recall / candidate-size on a train sample

### Blocking experiments (no learned model)

Must run from the **repository root** so `student_resource/dataset` resolves:

```bash
# Windows PowerShell
$env:PYTHONPATH = "code/business_entity_resolution"
python -m src.evaluation.benchmark --method exact_name --sample-size 10000 --out experiments/results.csv

python -m src.evaluation.benchmark --method exact_address --sample-size 10000
python -m src.evaluation.benchmark --method token_name --sample-size 10000
python -m src.evaluation.benchmark --method hybrid --sample-size 10000
```

```bash
# macOS / Linux
PYTHONPATH=code/business_entity_resolution python -m src.evaluation.benchmark --method hybrid --sample-size 10000
```

Indexes are built on **full** train S2 and S3, then queried on `--sample-size` S1 rows. Results append to `experiments/results.csv`.

### Matching model (planned; MIT/Apache, ≤ 8B params)

The intended train loop, once the classifier lands under `src/`:

1. Hold out a validation slice of train S1 + `train_ground_truth.tsv`.
2. Run blocking → candidate pairs (this list is what you later write as `candidate_pairs.tsv`).
3. Build pair features (Jaccard, Levenshtein / RapidFuzz, TF-IDF cosine on name and address; country as an **open** string, never a US/India-only one-hot).
4. Train a licensed matcher (e.g. LightGBM / sklearn) on labeled pairs: true matches from ground truth vs negatives sampled from candidates.
5. Choose the score threshold that **maximizes macro F0.5** on validation (empty predictions for low scores — do not force a match).
6. Retrain on all train labels if desired, then run the same pipeline on test and write both output TSVs.

Until that code exists, do **not** treat `baseline_v1.py` as a trained model: it keeps every exact-normalized candidate as a match.

## Tests

From `code/business_entity_resolution/`:

```bash
cd code/business_entity_resolution
python -m pytest tests/ -v
```

## Final submission zip (ML 2.pdf)

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

Fill `student_resource/Documentation_template.md` (methodology, blocking, model/features, other notes) and place it at the zip root. **AMZN ML 1.pdf** also asks for a 1–2 page ML write-up plus commented training/inference source; the filled template covers the methodology side.

## Constraints from the PDFs

- Challenge window (AMZN ML 1): 25 Sep 2026 12:00 AM IST – 27 Sep 2026 11:59 PM IST; **5 submissions per day**.
- Public + private leaderboards; final ranking uses the private split. Smaller, high-quality candidate sets are reviewed for ranking beyond the score.
- Final model: **MIT or Apache 2.0**, **≤ 8 billion parameters**.
- No external identity lookup.
- Desktop/laptop only; one machine per participant.

## Scoring status on the Portal

A correctly formatted `matching_results.tsv` should show **SCORED** with your F0.5. Format failures are not evaluated.
