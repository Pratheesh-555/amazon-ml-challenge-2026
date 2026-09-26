"""
Evaluation metrics for business entity resolution.

Includes the official F0.5 macro metric, plus blocking-specific metrics
like candidate recall and candidate set statistics.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Set


def f05_per_entity(predicted: Set[str], truth: Set[str]) -> float:
    """Calculate F0.5 for a single Source 1 entity.

    Rules:
    - If truth is empty (singleton):
        - Predicted empty -> 1.0
        - Predicted non-empty -> 0.0
    - If truth is non-empty but predicted is empty -> 0.0
    - Otherwise standard F0.5: (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    """
    if not truth:
        return 1.0 if not predicted else 0.0
    
    if not predicted:
        return 0.0

    intersection = len(predicted & truth)
    if intersection == 0:
        return 0.0

    precision = intersection / len(predicted)
    recall = intersection / len(truth)

    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05


def f05_macro(predictions_dict: Dict[str, Set[str]], truth_dict: Dict[str, Set[str]]) -> float:
    """Calculate macro-averaged F0.5 across all entities in predictions_dict."""
    if not predictions_dict:
        return 0.0

    scores = []
    for s1_id, predicted in predictions_dict.items():
        truth = truth_dict.get(s1_id, set())
        scores.append(f05_per_entity(predicted, truth))

    return float(np.mean(scores))


def blocking_metrics(candidates_dict: Dict[str, Set[str]], truth_dict: Dict[str, Set[str]]) -> dict:
    """Calculate metrics for a blocking / candidate generation step.
    
    Metrics include:
    - candidate_recall: Proportion of true match pairs successfully retrieved.
    - avg_candidates: Mean number of candidates per S1 entity.
    - median_candidates: Median candidates.
    - p90_candidates, p95_candidates, p99_candidates: Percentiles of candidate counts.
    - zero_candidate_rate: Fraction of S1 entities that got 0 candidates.
    """
    if not candidates_dict:
        return {}

    total_true_pairs = 0
    found_true_pairs = 0
    candidate_counts = []
    zero_count = 0

    for s1_id, candidates in candidates_dict.items():
        truth = truth_dict.get(s1_id, set())
        
        # Candidate counting
        c_len = len(candidates)
        candidate_counts.append(c_len)
        if c_len == 0:
            zero_count += 1
            
        # Recall counting (only for non-singletons)
        if truth:
            total_true_pairs += len(truth)
            found_true_pairs += len(truth & candidates)

    counts = np.array(candidate_counts)
    
    candidate_recall = found_true_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
    
    return {
        "candidate_recall": candidate_recall,
        "avg_candidates": float(np.mean(counts)),
        "median_candidates": float(np.median(counts)),
        "p90_candidates": float(np.percentile(counts, 90)),
        "p95_candidates": float(np.percentile(counts, 95)),
        "p99_candidates": float(np.percentile(counts, 99)),
        "zero_candidate_rate": zero_count / len(candidates_dict),
    }
