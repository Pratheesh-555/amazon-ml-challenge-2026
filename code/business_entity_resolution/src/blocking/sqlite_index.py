"""A disk-backed index: avoids holding 10M records in Python memory."""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Iterable

from blocking.keys import keys, normalized


SCHEMA = """
CREATE TABLE records (entity_id TEXT PRIMARY KEY, name TEXT, address TEXT, country TEXT);
CREATE TABLE blocks (kind TEXT, block_key TEXT, entity_id TEXT);
"""


def build(files: Iterable[Path], database: Path, batch_size: int = 50_000) -> None:
    if database.exists():
        raise FileExistsError(f"Refusing to overwrite existing index: {database}")
    con = sqlite3.connect(database)
    con.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA temp_store=FILE;" + SCHEMA)
    record_batch, block_batch = [], []
    for path in files:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                value = normalized(row); entity = row["entity_id"]
                record_batch.append((entity, value["name"], value["address"], value["country"]))
                block_batch.extend((kind, key, entity) for kind, key in keys(row))
                if len(record_batch) >= batch_size:
                    con.executemany("INSERT INTO records VALUES (?, ?, ?, ?)", record_batch)
                    con.executemany("INSERT INTO blocks VALUES (?, ?, ?)", block_batch)
                    con.commit(); record_batch.clear(); block_batch.clear()
    if record_batch:
        con.executemany("INSERT INTO records VALUES (?, ?, ?, ?)", record_batch)
        con.executemany("INSERT INTO blocks VALUES (?, ?, ?)", block_batch)
        con.commit()
    # Building this index after the bulk insert is much faster than maintaining it
    # for every one of the tens of millions of block rows.
    print("building SQLite lookup index", flush=True)
    con.execute("CREATE INDEX blocks_lookup ON blocks(kind, block_key)")
    con.execute("ANALYZE"); con.commit(); con.close()


def candidates(con: sqlite3.Connection, row: dict[str, str], max_per_key: int = 80, max_total: int = 250) -> list[dict[str, str]]:
    """Returns the *final* candidate list, after common-key controls, before scoring."""
    entity_ids: set[str] = set()
    for kind, key in keys(row):
        # Very common blocks are deliberately ignored rather than silently truncated.
        count = con.execute("SELECT count(*) FROM blocks WHERE kind=? AND block_key=?", (kind, key)).fetchone()[0]
        if count <= max_per_key:
            entity_ids.update(x[0] for x in con.execute("SELECT entity_id FROM blocks WHERE kind=? AND block_key=?", (kind, key)))
    if not entity_ids: return []
    # Ranking/capping before ML is deterministic and therefore faithfully recorded in candidate_pairs.tsv.
    query = "SELECT entity_id, name, address, country FROM records WHERE entity_id IN (%s)" % ",".join("?" * len(entity_ids))
    rows = [{"entity_id": x[0], "name": x[1], "address": x[2], "country": x[3]} for x in con.execute(query, tuple(entity_ids))]
    n = normalized(row)
    def preliminary(r: dict[str, str]) -> tuple[float, str]:
        return ((r["name"] == n["name"]) * 4 + (r["address"] == n["address"]) * 4 +
                len(set(r["name"].split()) & set(n["name"].split())) + len(set(r["address"].split()) & set(n["address"].split())), r["entity_id"])
    return sorted(rows, key=preliminary, reverse=True)[:max_total]
