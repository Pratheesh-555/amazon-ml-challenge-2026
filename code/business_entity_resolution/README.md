# Business Entity Resolution Pipeline

This repository contains a modular pipeline for the Amazon ML Challenge 2026. 

The pipeline is split into independent stages to strictly control memory usage and maximize reproducibility:
1. **Preprocessing**: Normalizes names, addresses, and country codes (handling English, French, and Indian scripts) while expanding abbreviations and stripping legal suffixes.
2. **Blocking (Candidate Generation)**: Reduces the 10M x 1.7M cross-product space into a manageable candidate set using scalable indexers.
3. **Feature Engineering** *(Upcoming)*: Computes pair-wise similarities (Jaccard, Levenshtein, phonetic) between Source 1 and candidates.
4. **Matching Model** *(Upcoming)*: A classifier (e.g., LightGBM) to score candidates and pick the final matches using an F0.5 optimized threshold.

## 1. Setup

Navigate to the `code/business_entity_resolution` directory and install the required dependencies:

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

*(Note: Data should be placed in `student_resource/dataset/` relative to the repository root, as per the challenge setup).*

## 2. Implemented Blocking Algorithms

You can swap between different algorithms to find the best candidate recall vs. memory trade-off:

*   **`exact_name`**: Strict exact matching on the normalized business name. 
*   **`exact_address`**: Strict exact matching on the normalized address (with 50+ abbreviations expanded).
*   **`token_name`**: Inverted index on name tokens. Retrieves candidates that share at least *N* tokens. Automatically filters out "stop word" tokens (like "services", "inc") that appear in >100,000 records.
*   **`hybrid`**: A composite blocker that runs multiple strategies (e.g., Exact Name + Exact Address) and takes the union of their candidates.

## 3. Benchmarking Algorithms

To test blocking algorithms and see their Candidate Recall, Average Candidates, and Runtime without generating massive output files, use the benchmark script. It outputs results directly to `experiments/results.csv`.

```bash
# Run exact_name matching on a subset of 10,000 entities
python -m src.evaluation.benchmark --method exact_name --sample-size 10000

# Run token overlap blocking
python -m src.evaluation.benchmark --method token_name --sample-size 10000

# Run a hybrid combination
python -m src.evaluation.benchmark --method hybrid --sample-size 10000
```

## 4. Applying the Model (End-to-End)

*(This component is actively in development as per the engineering plan)*

Once the best hybrid blocker is identified, you will run the full inference script to generate the submission files:

```bash
# Coming soon:
python -m src.inference.generate_candidates --method hybrid --out output/candidate_pairs.tsv
python -m src.inference.score_matches --candidates output/candidate_pairs.tsv --out output/matching_results.tsv
```

## 5. Development & Testing

We enforce strict test coverage for normalizers and metric calculators to ensure recall ceilings are accurate.

To run the test suite:
```bash
python -m pytest tests/ -v
```
