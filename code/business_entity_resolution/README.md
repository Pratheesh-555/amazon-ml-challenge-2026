# Business Entity Resolution Pipeline

Self-contained pipeline for Amazon ML Challenge 2026 (see **ML 2.pdf** / `student_resource/README.md`).

Goal: for every Source 1 entity, find matching Source 2 / Source 3 records. Produce:

- `output/matching_results.tsv` — final matches (Portal / leaderboard)
- `output/candidate_pairs.tsv` — last candidate set scored by the matcher (required in the final zip)

Stages:

1. **Preprocessing** — normalize names, addresses, and country labels (open set: train has US/India; test also has France). Expand abbreviations, strip legal suffixes. Do not hard-code countries.
2. **Blocking** — index S2/S3 so each S1 entity gets a small candidate set instead of a full cross-product.
3. **Matching** — not trained in-repo yet. Current inference (`scripts/baseline_v1.py` at repo root) treats exact-normalized name/address hits as matches.
4. **Output** — tab-separated TSVs with one row per S1 entity.

## 1. Setup

Install from this folder:

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

Data is **not** inside this folder. It lives at the repository root:

```
../../student_resource/dataset/train/
../../student_resource/dataset/test/
```

All reads/writes use `sep="\t"`, `encoding="utf-8"`.

## 2. Blocking methods

| `--method` | Behavior |
| --- | --- |
| `exact_name` | Exact match on normalized name, partitioned by country |
| `exact_address` | Exact match on normalized address, partitioned by country |
| `token_name` | Inverted index on name tokens; keep IDs sharing at least *N* tokens |
| `hybrid` | Union of exact name + exact address |

## 3. Benchmark blocking on train (labeled)

Ground truth is only in `train_ground_truth.tsv`. Hold out / sample S1 and score **candidate recall** (upper bound on match recall).

Run from the **repository root** (`amazon-ml-challenge-2026/`), not this directory — paths are `student_resource/dataset/...`:

```bash
# Windows PowerShell
$env:PYTHONPATH = "code/business_entity_resolution"
python -m src.evaluation.benchmark --method exact_name --sample-size 10000 --out experiments/results.csv

python -m src.evaluation.benchmark --method hybrid --sample-size 10000
```

```bash
# macOS / Linux
PYTHONPATH=code/business_entity_resolution python -m src.evaluation.benchmark --method hybrid --sample-size 10000
```

Metrics go to `experiments/results.csv`. Macro F0.5 for a matcher is in `src/evaluation/metrics.py` (`f05_macro`).

## 4. Train a matching model

There is **no** `train.py` yet. When you add one under `src/`, it should:

1. Split train S1 + ground truth into train/validation.
2. Build candidates with a blocker (`candidate_pairs` = whatever the model will score).
3. Featurize pairs (string similarity on name/address; country as a free-form label).
4. Train an **MIT/Apache 2.0** model with **≤ 8B** parameters (sklearn / LightGBM-class models fit this).
5. Pick a threshold that maximizes **macro F0.5** on validation, including singletons (empty prediction = 1.0 if truth is empty).
6. Run the same blocker + scorer on **test** and write both output TSVs.

Until that exists, generate leaderboard-shaped matches with the repo-root baseline:

```bash
# from amazon-ml-challenge-2026/
python scripts/baseline_v1.py
# writes output/matching_results.tsv
```

Expected `matching_results.tsv` schema:

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

`candidate_pairs.tsv` uses `source1_entity_id` and `candidate_entity_ids` with the same row/ID rules. Final matches must be a subset of candidates.

## 5. Validate expected output

From `student_resource/`:

```bash
python utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv \
    --test-dir dataset/test
```

`PASS` means format is safe to upload. It does not compute F0.5.

## 6. Tests

```bash
cd code/business_entity_resolution
python -m pytest tests/ -v
```

## 7. Reproduce end-to-end (data → blocking → matching → output)

From repository root:

```bash
pip install -r code/business_entity_resolution/requirements.txt
python scripts/baseline_v1.py
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --test-dir student_resource/dataset/test
```

Upload `output/matching_results.tsv` to the Portal. Include both TSVs plus this `src/` tree in `<team_name>_submission.zip` as specified in ML 2.pdf.
