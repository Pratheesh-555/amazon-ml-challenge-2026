import pytest
from src.evaluation.metrics import f05_per_entity, f05_macro, blocking_metrics

def test_f05_per_entity():
    # Exact match
    assert f05_per_entity({"A", "B"}, {"A", "B"}) == 1.0
    
    # Singleton exact match
    assert f05_per_entity(set(), set()) == 1.0
    
    # Singleton false positive
    assert f05_per_entity({"A"}, set()) == 0.0
    
    # Missed match
    assert f05_per_entity(set(), {"A"}) == 0.0
    
    # Partial match
    # Predicted: A, B, C (3)
    # Truth: A, B (2)
    # Precision: 2/3, Recall: 2/2 = 1.0
    # F0.5 = (1.25 * 0.666 * 1.0) / (0.25 * 0.666 + 1.0) = 0.833 / 1.166 = 0.714
    score = f05_per_entity({"A", "B", "C"}, {"A", "B"})
    assert abs(score - 0.714) < 0.001

def test_f05_macro():
    predictions = {
        "S1-1": {"A", "B"},
        "S1-2": set(),
        "S1-3": {"C"}
    }
    truth = {
        "S1-1": {"A", "B"},  # 1.0
        "S1-2": set(),       # 1.0
        "S1-3": set()        # 0.0
    }
    assert abs(f05_macro(predictions, truth) - 0.666) < 0.001

def test_blocking_metrics():
    candidates = {
        "S1-1": {"A", "B", "C"},
        "S1-2": {"D"},
        "S1-3": set(),
        "S1-4": {"E"}
    }
    truth = {
        "S1-1": {"A", "B"}, # 2/2 found
        "S1-2": {"X"},      # 0/1 found
        "S1-3": set(),      # singleton (ignored in recall calculation)
        "S1-4": {"E", "F"}  # 1/2 found
    }
    
    metrics = blocking_metrics(candidates, truth)
    # True pairs = 2 + 1 + 2 = 5
    # Found pairs = 2 + 0 + 1 = 3
    # Recall = 3/5 = 0.6
    assert metrics["candidate_recall"] == 0.6
    assert metrics["zero_candidate_rate"] == 0.25 # 1/4
    # Counts: 3, 1, 0, 1 -> avg: 1.25
    assert metrics["avg_candidates"] == 1.25
