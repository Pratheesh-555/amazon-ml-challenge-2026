"""Explainable pairwise features; all are computed from supplied fields only."""
from __future__ import annotations

from difflib import SequenceMatcher
from preprocessing.normalization import informative_tokens, numbers


def _jaccard(a: str, b: str) -> float:
    left, right = set(a.split()), set(b.split())
    return len(left & right) / len(left | right) if left or right else 0.0


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _token_set_ratio(a: str, b: str) -> float:
    """Dependency-free fallback; installing RapidFuzz makes this easy to replace for speed."""
    left, right = " ".join(sorted(set(a.split()))), " ".join(sorted(set(b.split())))
    return _ratio(left, right)


def make(left: dict[str, str], right: dict[str, str]) -> dict[str, float]:
    n1, a1, c1 = left["name"], left["address"], left["country"]
    n2, a2, c2 = right["name"], right["address"], right["country"]
    ns1, ns2 = set(numbers(a1)), set(numbers(a2))
    return {
        "country_equal": float(bool(c1) and c1 == c2),
        "name_ratio": _ratio(n1, n2), "name_token_ratio": _token_set_ratio(n1, n2),
        "name_jaccard": _jaccard(n1, n2), "address_ratio": _ratio(a1, a2),
        "address_token_ratio": _token_set_ratio(a1, a2), "address_jaccard": _jaccard(a1, a2),
        "name_exact": float(bool(n1) and n1 == n2), "address_exact": float(bool(a1) and a1 == a2),
        "number_overlap": float(bool(ns1 & ns2)),
        "left_name_missing": float(not n1), "right_name_missing": float(not n2),
        "left_address_missing": float(not a1), "right_address_missing": float(not a2),
    }


FEATURE_NAMES = list(make({"name":"a","address":"1","country":"x"}, {"name":"b","address":"2","country":"x"}))
