"""Ultra-fast inference via a single sequential blocks-table scan.

Algorithm
─────────
Traditional approach: for each S1 row, fire N SQL queries to retrieve candidates.
  → 1.73M × 8 keys × 2 queries = ~28M SQL round-trips  (slow)

This approach: one full sequential scan of the blocks table.
  Pass 1 – Scan test_source1.tsv and build an in-memory mapping:
             s1_index[(kind, block_key)] = [s1_entity_id, ...]
             Simultaneously collect all normalized S1 records for feature computation.
  Pass 2 – Sequential full scan of blocks table (no index, fastest mode).
             For every (kind, block_key, s2_entity_id) row:
               if (kind, block_key) is in s1_index and block_cnt <= max_per_key:
                 emit (s1_id, s2_id) candidate pair
             Block counts are counted in real time using a streaming counter;
             oversized blocks are discarded after the fact.
  Pass 3 – For each S1 entity, fetch its candidate full records from the
             records table in large batches (avoids per-row query overhead).
  Pass 4 – Score and write output.

Time estimate
─────────────
  Pass 1 (S1 scan):      ~95s  (already measured)
  Pass 2 (blocks scan):  ~190s (sequential, no index; ~3.1 min measured)
  Pass 3 (batch fetch):  ~60s  (a handful of large IN queries)
  Pass 4 (score+write):  ~30s  (pure Python, no I/O)
  Total:  ~6 min  (vs 15+ hours original)
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from blocking.keys import keys as blocking_keys, normalized
from features.pair_features import FEATURE_NAMES, make
from models.logistic_matcher import predict

# ── tunables ────────────────────────────────────────────────────────────────────
BLOCK_BATCH   = 500_000   # rows to fetch per blocks-scan iteration
RECORD_BATCH  = 50_000    # entity_ids per IN-query for records
WRITE_BUFFER  = 50_000    # output rows to buffer before flushing
PROGRESS_ROWS = 50_000    # how often to log blocks-scan progress


def _open_db(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.execute("PRAGMA mmap_size=4294967296")   # 4 GB mmap
    con.execute("PRAGMA cache_size=-131072")     # 128 MB page cache
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA query_only=ON")
    return con


# ── PASS 1: build S1 in-memory index ─────────────────────────────────────────

def _build_s1_index(
    test_source1: Path,
) -> tuple[dict[tuple, list[str]], dict[str, dict[str, str]]]:
    """
    Returns
    -------
    s1_key_index : {(kind, block_key): [s1_entity_id, ...]}
    s1_records   : {s1_entity_id: {name, address, country}}
    """
    print("Pass 1: building S1 key index …", flush=True)
    t0 = time.perf_counter()
    s1_key_index: dict[tuple, list[str]] = defaultdict(list)
    s1_records: dict[str, dict[str, str]] = {}

    with test_source1.open(encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh, delimiter="\t"), 1):
            eid  = row["entity_id"]
            norm = normalized(row)
            s1_records[eid] = norm
            for kv in blocking_keys(row):
                s1_key_index[kv].append(eid)
            if i % 100_000 == 0:
                print(f"  S1 scan: {i:,} rows, {len(s1_key_index):,} unique keys", flush=True)

    elapsed = time.perf_counter() - t0
    print(
        f"Pass 1 done: {len(s1_records):,} S1 entities, "
        f"{len(s1_key_index):,} unique blocking keys in {elapsed:.1f}s",
        flush=True,
    )
    return dict(s1_key_index), s1_records


# ── PASS 2: sequential blocks scan → candidate pairs ─────────────────────────

def _build_candidates(
    con: sqlite3.Connection,
    s1_key_index: dict[tuple, list[str]],
    max_per_key: int,
) -> dict[str, set[str]]:
    """
    Two-pass sequential scan of the blocks table (memory-efficient).

    Pass 2a: count block sizes for all keys that appear in s1_key_index.
    Pass 2b: re-scan blocks, skipping oversized keys, and build candidate map.

    This avoids storing raw_pairs in memory (which could be multi-GB).
    """
    relevant = set(s1_key_index)

    # ── Pass 2a: count block sizes ──────────────────────────────────────────
    print("Pass 2a: counting block sizes (sequential scan) …", flush=True)
    t0 = time.perf_counter()
    block_counts: dict[tuple, int] = defaultdict(int)
    scanned = 0
    for kind, bkey in con.execute("SELECT kind, block_key FROM blocks"):
        kv = (kind, bkey)
        if kv in relevant:
            block_counts[kv] += 1
        scanned += 1
        if scanned % (PROGRESS_ROWS * 10) == 0:
            print(f"  2a: {scanned:,} rows scanned ({time.perf_counter()-t0:.0f}s)", flush=True)

    oversized = frozenset(kv for kv, cnt in block_counts.items() if cnt > max_per_key)
    small_keys = relevant - oversized
    elapsed_2a = time.perf_counter() - t0
    print(
        f"Pass 2a done: {scanned:,} rows in {elapsed_2a:.1f}s | "
        f"oversized={len(oversized):,} | usable={len(small_keys):,}",
        flush=True,
    )

    # ── Pass 2b: collect entity_ids for non-oversized keys ──────────────────
    print("Pass 2b: collecting candidates (sequential scan) …", flush=True)
    t1 = time.perf_counter()
    candidates: dict[str, set[str]] = defaultdict(set)
    scanned = 0
    for kind, bkey, eid in con.execute("SELECT kind, block_key, entity_id FROM blocks"):
        kv = (kind, bkey)
        if kv in small_keys:
            for s1_eid in s1_key_index[kv]:
                candidates[s1_eid].add(eid)
        scanned += 1
        if scanned % (PROGRESS_ROWS * 10) == 0:
            print(f"  2b: {scanned:,} rows scanned | {len(candidates):,} S1 with cands ({time.perf_counter()-t1:.0f}s)", flush=True)

    elapsed_2b = time.perf_counter() - t1
    total_pairs = sum(len(v) for v in candidates.values())
    print(
        f"Pass 2b done: {len(candidates):,} S1 entities with candidates "
        f"({total_pairs:,} total pairs) in {elapsed_2b:.1f}s",
        flush=True,
    )
    return dict(candidates)


# ── PASS 3: batch-fetch full records for candidate S2/S3 entities ─────────────

def _fetch_records(
    con: sqlite3.Connection,
    entity_ids: set[str],
) -> dict[str, dict[str, str]]:
    """Fetch name/address/country for a set of entity_ids in batched IN queries."""
    result: dict[str, dict[str, str]] = {}
    ids = list(entity_ids)
    for start in range(0, len(ids), RECORD_BATCH):
        chunk = ids[start : start + RECORD_BATCH]
        placeholders = ",".join("?" * len(chunk))
        for eid, name, addr, cty in con.execute(
            f"SELECT entity_id, name, address, country FROM records WHERE entity_id IN ({placeholders})",
            chunk,
        ):
            result[eid] = {"entity_id": eid, "name": name, "address": addr, "country": cty}
    return result


# ── PASS 4: score and write output ───────────────────────────────────────────

def _score_and_write(
    s1_records: dict[str, dict[str, str]],
    candidates: dict[str, set[str]],
    s2s3_records: dict[str, dict[str, str]],
    artifact: dict,
    max_candidates: int,
    output: Path,
) -> None:
    threshold = artifact["threshold"]
    output.mkdir(parents=True, exist_ok=True)
    print("Pass 4: scoring and writing output …", flush=True)
    t0 = time.perf_counter()

    def _score_pair(left: dict, cand: dict) -> tuple[float, str]:
        """Preliminary rank: exact matches score highest."""
        return (
            (cand["name"] == left["name"]) * 4
            + (cand["address"] == left["address"]) * 4
            + len(set(cand["name"].split()) & set(left["name"].split()))
            + len(set(cand["address"].split()) & set(left["address"].split())),
            cand["entity_id"],
        )

    cand_path  = output / "candidate_pairs.tsv"
    match_path = output / "matching_results.tsv"

    cand_buf:  list[list] = []
    match_buf: list[list] = []
    written = 0

    with (
        cand_path.open("w",  encoding="utf-8", newline="") as cand_fh,
        match_path.open("w", encoding="utf-8", newline="") as match_fh,
    ):
        cw = csv.writer(cand_fh,  delimiter="\t", lineterminator="\n")
        mw = csv.writer(match_fh, delimiter="\t", lineterminator="\n")
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])
        mw.writerow(["source1_entity_id", "matched_entity_ids"])

        for s1_eid, left in s1_records.items():
            possible_ids = candidates.get(s1_eid)
            if possible_ids:
                possible = [s2s3_records[e] for e in possible_ids if e in s2s3_records]
                # Rank and cap
                possible = sorted(possible, key=lambda r: _score_pair(left, r), reverse=True)[:max_candidates]
                cand_ids  = [r["entity_id"] for r in possible]
                if possible:
                    scores   = predict(artifact, [make(left, r) for r in possible])
                    selected = [r["entity_id"] for r, s in zip(possible, scores) if s >= threshold]
                else:
                    selected = []
            else:
                cand_ids = []
                selected = []

            cand_buf.append([s1_eid, ",".join(cand_ids)])
            match_buf.append([s1_eid, ",".join(selected)])
            written += 1

            if written % WRITE_BUFFER == 0:
                cw.writerows(cand_buf)
                mw.writerows(match_buf)
                cand_fh.flush()
                match_fh.flush()
                cand_buf.clear()
                match_buf.clear()
                print(f"  Written {written:,} S1 rows …", flush=True)

        if cand_buf:
            cw.writerows(cand_buf)
            mw.writerows(match_buf)

    print(f"Pass 4 done: {written:,} rows written in {time.perf_counter()-t0:.1f}s", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def run(
    test: Path,
    db: Path,
    model_file: Path,
    output: Path,
    max_per_key: int,
    max_candidates: int,
) -> None:
    t_total = time.perf_counter()
    artifact = json.loads(model_file.read_text())

    # Pass 1
    s1_key_index, s1_records = _build_s1_index(test / "test_source1.tsv")

    # Pass 2
    con = _open_db(db)
    candidates = _build_candidates(con, s1_key_index, max_per_key)

    # Pass 3: collect all unique S2/S3 candidate ids to fetch
    all_s2s3 = set()
    for eids in candidates.values():
        all_s2s3.update(eids)
    print(f"Pass 3: fetching records for {len(all_s2s3):,} unique S2/S3 candidates …", flush=True)
    t3 = time.perf_counter()
    s2s3_records = _fetch_records(con, all_s2s3)
    print(f"Pass 3 done in {time.perf_counter()-t3:.1f}s", flush=True)
    con.close()

    # Pass 4
    _score_and_write(s1_records, candidates, s2s3_records, artifact, max_candidates, output)

    print(
        f"\n✓ Pipeline complete in {(time.perf_counter()-t_total)/60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Full-scan inference (single blocks-table pass).")
    p.add_argument("--test",           type=Path, required=True)
    p.add_argument("--index",          type=Path, required=True)
    p.add_argument("--model",          type=Path, required=True)
    p.add_argument("--output",         type=Path, required=True)
    p.add_argument("--max-per-key",    type=int,  default=80)
    p.add_argument("--max-candidates", type=int,  default=250)
    a = p.parse_args()
    run(a.test, a.index, a.model, a.output, a.max_per_key, a.max_candidates)
