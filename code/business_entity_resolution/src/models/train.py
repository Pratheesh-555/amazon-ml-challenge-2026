from __future__ import annotations

import argparse, pickle
from pathlib import Path
from models.logistic_matcher import fit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit the dependency-free logistic matcher and tune macro F0.5 on held-out S1 entities.")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    with args.pairs.open("rb") as f: rows, labels, groups, target_ids, truth = pickle.load(f)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    threshold, score = fit(rows, labels, groups, target_ids, truth, args.model)
    print(f"validation macro F0.5={score:.6f}; selected threshold={threshold:.3f}")
