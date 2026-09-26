"""Repeatable held-out comparison of strict and multi-key blocking."""
from __future__ import annotations

import argparse, csv, hashlib, json, random
from pathlib import Path
from blocking.keys import keys, normalized
from features.pair_features import make


def selected(entity_id: str, modulus: int, remainder: int) -> bool:
    return int(hashlib.blake2b(entity_id.encode(), digest_size=8).hexdigest(), 16) % modulus == remainder


def load_sample(train: Path, modulus: int, remainder: int, maximum: int):
    wanted, truth = set(), {}
    with (train / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            ids = set(filter(None, r["matched_entity_ids"].split(",")))
            if ids and selected(r["source1_entity_id"], modulus, remainder) and len(truth) < maximum:
                truth[r["source1_entity_id"]] = ids; wanted.update(ids)
    s1 = {}
    with (train / "train_source1.tsv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["entity_id"] in truth: s1[r["entity_id"]] = normalized(r)
    targets = {}
    for filename in ("train_source2.tsv", "train_source3.tsv"):
        with (train / filename).open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["entity_id"] in wanted: targets[r["entity_id"]] = normalized(r)
    if len(s1) != len(truth) or len(targets) != len(wanted): raise RuntimeError("Could not resolve every held-out training ID")
    return s1, targets, truth


def run(train: Path, output: Path, maximum: int, modulus: int, remainder: int) -> None:
    s1, targets, truth = load_sample(train, modulus, remainder, maximum)
    exact = multi = total = 0
    feature_rows, labels, groups, ids = [], [], [], []
    # Positives plus country-preserving shuffled hard-ish negatives from another S1 group.
    pool = list(targets.items()); rng = random.Random(2026)
    for entity, actual in truth.items():
        left = s1[entity]
        for target in actual:
            right = targets[target]; total += 1
            left_keys, right_keys = set(keys({"business_name":left["name"], "business_address":left["address"], "country":left["country"]})), set(keys({"business_name":right["name"], "business_address":right["address"], "country":right["country"]}))
            exact += bool({x for x in left_keys if x[0] in {"exact_name", "exact_address", "exact_pair"}} & right_keys)
            multi += bool(left_keys & right_keys)
            feature_rows.append(make(left, right)); labels.append(1); groups.append(entity); ids.append(target)
        negatives = [x for x in pool if x[0] not in actual and x[1]["country"] == left["country"]]
        for target, right in rng.sample(negatives, min(3, len(negatives))):
            feature_rows.append(make(left, right)); labels.append(0); groups.append(entity); ids.append(target)
    report = {"validation_s1":len(truth), "positive_pairs":total,
              "exact_block_recall":exact/total, "multi_key_block_recall":multi/total,
              "note":"Candidate-size measurement requires the full SQLite index; recalls are exact over held-out labelled pairs."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    import pickle
    with output.with_suffix(".pairs.pkl").open("wb") as f: pickle.dump((feature_rows, labels, groups, ids, truth), f)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--train",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--maximum",type=int,default=5000); p.add_argument("--modulus",type=int,default=10); p.add_argument("--remainder",type=int,default=0); a=p.parse_args(); run(a.train,a.output,a.maximum,a.modulus,a.remainder)
