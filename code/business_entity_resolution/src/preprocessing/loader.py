"""
Streaming data loader for business entity resolution.

All reads use sep='\\t', dtype=str, encoding='utf-8' to match the challenge format.
Ground truth is parsed into a dict for O(1) lookup.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Iterator, Optional, Set

import pandas as pd


def load_source(
    path: str,
    chunksize: Optional[int] = None,
    usecols: Optional[list] = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Load a source TSV (source1/2/3).

    Parameters
    ----------
    path:
        Absolute or relative path to a .tsv file.
    chunksize:
        If set, returns an iterator of DataFrames each with this many rows.
    usecols:
        If set, only load these columns (reduces memory).

    Returns
    -------
    DataFrame (if chunksize is None) or TextFileReader iterator.
    """
    kwargs = dict(
        sep="\t",
        dtype=str,
        encoding="utf-8",
        na_filter=False,  # keep empty strings as "", not NaN
    )
    if usecols:
        kwargs["usecols"] = usecols
    if chunksize:
        kwargs["chunksize"] = chunksize
    return pd.read_csv(path, **kwargs)


def load_ground_truth(path: str) -> Dict[str, Set[str]]:
    """Load ground truth into {source1_entity_id: set(matched_ids)}.

    Singletons (empty matched_entity_ids) map to an empty set.
    Streaming read to avoid loading the full GT string into memory at once.
    """
    result: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"].strip()
            raw = row["matched_entity_ids"].strip()
            if raw:
                result[s1_id] = set(raw.split(","))
            else:
                result[s1_id] = set()
    return result


def load_ground_truth_sample(path: str, s1_ids: Set[str]) -> Dict[str, Set[str]]:
    """Load ground truth for a specific subset of S1 IDs (fast for sampling)."""
    result: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"].strip()
            if s1_id not in s1_ids:
                continue
            raw = row["matched_entity_ids"].strip()
            result[s1_id] = set(raw.split(",")) if raw else set()
    return result


def iter_source_rows(path: str, chunksize: int = 200_000):
    """Yield individual rows as dicts from a source TSV, chunked for memory."""
    for chunk in load_source(path, chunksize=chunksize):
        for row in chunk.itertuples(index=False):
            yield row._asdict()


def count_rows(path: str) -> int:
    """Count data rows (excluding header) without loading into memory."""
    with open(path, encoding="utf-8", newline="") as f:
        return sum(1 for _ in f) - 1  # subtract header


def dataset_paths(root: str, split: str = "train") -> dict:
    """Return dict of standard dataset file paths for a given split."""
    base = os.path.join(root, split)
    paths = {
        "source1": os.path.join(base, f"{split}_source1.tsv"),
        "source2": os.path.join(base, f"{split}_source2.tsv"),
        "source3": os.path.join(base, f"{split}_source3.tsv"),
    }
    if split == "train":
        paths["ground_truth"] = os.path.join(base, "train_ground_truth.tsv")
    return paths
