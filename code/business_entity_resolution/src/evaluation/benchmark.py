"""
Benchmark script for blocking strategies.
"""
import argparse
import time
import os
import csv
import pandas as pd
from tqdm import tqdm

from src.preprocessing.loader import load_source, load_ground_truth_sample, dataset_paths
from src.evaluation.metrics import blocking_metrics
from src.blocking.exact_name import ExactNameBlocker
from src.blocking.exact_address import ExactAddressBlocker
from src.blocking.token_inverted import TokenInvertedBlocker
from src.blocking.hybrid import HybridBlocker

def get_blocker(method_name: str):
    if method_name == "exact_name":
        return ExactNameBlocker()
    elif method_name == "exact_address":
        return ExactAddressBlocker()
    elif method_name == "token_name":
        return TokenInvertedBlocker(min_overlap=2, field="business_name")
    elif method_name == "hybrid":
        return HybridBlocker([
            ExactNameBlocker(),
            ExactAddressBlocker(),
        ])
    else:
        raise ValueError(f"Unknown method: {method_name}")

def run_benchmark(method: str, sample_size: int, output_csv: str):
    print(f"--- Benchmarking {method} on {sample_size} S1 entities ---")
    paths = dataset_paths("student_resource/dataset", "train")
    
    print("Loading S1 sample...")
    s1_df = load_source(paths["source1"]).head(sample_size)
    s1_ids = set(s1_df.entity_id)
    
    print("Loading Ground Truth for sample...")
    truth_dict = load_ground_truth_sample(paths["ground_truth"], s1_ids)
    
    # We need to collect all valid matches to load a subset of S2/S3?
    # No, to accurately test blocking, we must build the index on ALL of S2/S3.
    # We can chunk S2/S3 or just load it entirely if RAM permits (10M rows total).
    # Since we want accurate memory/time, let's load full S2/S3 into RAM for the index.
    print("Loading full S2 and S3 for indexing (this takes memory)...")
    t0 = time.time()
    s2_df = load_source(paths["source2"])
    s3_df = load_source(paths["source3"])
    load_time = time.time() - t0
    print(f"Data load time: {load_time:.2f}s")
    
    blocker = get_blocker(method)
    
    print("Building index...")
    t0 = time.time()
    blocker.build_index(s2_df, s3_df)
    index_time = time.time() - t0
    print(f"Index time: {index_time:.2f}s")
    
    print("Generating candidates...")
    candidates_dict = {}
    
    t0 = time.time()
    for row in tqdm(s1_df.itertuples(index=False), total=len(s1_df)):
        candidates_dict[row.entity_id] = blocker.generate_candidates(row._asdict())
    query_time = time.time() - t0
    print(f"Query time: {query_time:.2f}s ({len(s1_df)/query_time:.1f} queries/s)")
    
    metrics = blocking_metrics(candidates_dict, truth_dict)
    
    print("\nResults:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    # Write to CSV
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "method", "sample_size", "candidate_recall", 
                "avg_candidates", "median_candidates", "p90_candidates",
                "p95_candidates", "p99_candidates", "zero_candidate_rate",
                "index_time", "query_time"
            ])
        writer.writerow([
            method, sample_size, metrics["candidate_recall"],
            metrics["avg_candidates"], metrics["median_candidates"],
            metrics["p90_candidates"], metrics["p95_candidates"],
            metrics["p99_candidates"], metrics["zero_candidate_rate"],
            index_time, query_time
        ])
    print(f"Saved results to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--out", type=str, default="experiments/results.csv")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    run_benchmark(args.method, args.sample_size, args.out)
