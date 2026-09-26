# Business Entity Resolution

This solution uses only the supplied TSV data. It is deliberately a **blocking + classifier** system: the candidate list is compact and audited, then a learned matcher is thresholded for the official macro F_0.5 metric.

## Why these algorithms

- `rule_baseline.py` is an intentionally conservative exact-field benchmark. It is useful for detecting whether a complex model really improves on a high-precision baseline.
- `keys.py` is the selected candidate generator. It unions exact normalized name/address keys with country-aware name prefixes, meaningful name tokens and address numbers. This covers formatting and abbreviation noise without hard-coding the training countries; `France` passes through unchanged.
- `logistic_matcher.py` is the selected dependency-free calibrated matcher. It learns from explainable similarities and is reliable to reproduce here. `xgb_matcher.py` is kept as an optional nonlinear experiment; it must only replace the selected model if it improves an untouched holdout.
- Semantic ANN/transformers are not enabled by default. They would require a separately licensed multilingual model and must earn their cost by improving the held-out benchmark. No external entity data or lookup is used.

## Reproduce

From the project root, create a virtual environment and install the pinned dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt
export PYTHONPATH="$PWD/code/business_entity_resolution/src"
```

First benchmark blocking on a deterministic S1 holdout. It reports positive-pair capture for exact-only and multi-key alternatives and writes a feature set with positive and country-preserving negative pairs:

```bash
.venv/bin/python -m evaluation.benchmark --train student_resource/dataset/train --output artifacts/blocking_benchmark.json --maximum 5000
.venv/bin/python -m models.train --pairs artifacts/blocking_benchmark.pairs.pkl --model artifacts/logistic_matcher.json
```

Only after inspecting these numbers, run inference. The SQLite index is disk-backed so it does not attempt to keep the 10M candidate-side rows in RAM. It can take substantial time and disk; do not interrupt it mid-build. `candidate_pairs.tsv` is written from precisely the final capped candidate set passed into the model.

```bash
.venv/bin/python -m inference.run_pipeline --build-index \
  --test student_resource/dataset/test --index artifacts/test_candidates.sqlite \
  --model artifacts/logistic_matcher.json --output output
python3 student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
```

## Decision required before final submission

The chosen `max-per-key` and `max-candidates` values are deliberately command-line settings, not hidden assumptions. Compare candidate recall, average candidates and held-out macro F_0.5 for several values, then record the winning settings and observed metrics in the methodology document.
