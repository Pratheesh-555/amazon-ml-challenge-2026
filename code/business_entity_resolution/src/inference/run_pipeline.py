"""Build the test candidate index and write both required TSVs in one reproducible run."""
from __future__ import annotations

import argparse, csv, sqlite3
from pathlib import Path
import json
from blocking.keys import normalized
from blocking.sqlite_index import build, candidates
from features.pair_features import FEATURE_NAMES, make
from models.logistic_matcher import predict


def write_predictions(test: Path, db: Path, model_file: Path, output: Path, max_per_key: int, max_candidates: int) -> None:
    artifact = json.loads(model_file.read_text()); threshold = artifact["threshold"]
    output.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    with (test / "test_source1.tsv").open(encoding="utf-8", newline="") as source, \
         (output / "candidate_pairs.tsv").open("w", encoding="utf-8", newline="") as candidate_file, \
         (output / "matching_results.tsv").open("w", encoding="utf-8", newline="") as match_file:
        candidate_writer, match_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n"), csv.writer(match_file, delimiter="\t", lineterminator="\n")
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"]); match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for i, row in enumerate(csv.DictReader(source, delimiter="\t"), 1):
            left, possible = normalized(row), candidates(con, row, max_per_key, max_candidates)
            # candidate_pairs is written before the model: exactly the rows scored below.
            candidate_writer.writerow([row["entity_id"], ",".join(x["entity_id"] for x in possible)])
            if possible:
                scores = predict(artifact, [make(left, x) for x in possible])
                selected = [x["entity_id"] for x, score in zip(possible, scores) if score >= threshold]
            else: selected = []
            match_writer.writerow([row["entity_id"], ",".join(selected)])
            if i % 10_000 == 0: print(f"scored {i:,} S1 records", flush=True)
    con.close()


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--test",type=Path,required=True); p.add_argument("--index",type=Path,required=True); p.add_argument("--model",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--build-index",action="store_true"); p.add_argument("--max-per-key",type=int,default=80); p.add_argument("--max-candidates",type=int,default=250); a=p.parse_args()
    if a.build_index: build([a.test / "test_source2.tsv", a.test / "test_source3.tsv"], a.index)
    write_predictions(a.test,a.index,a.model,a.output,a.max_per_key,a.max_candidates)
