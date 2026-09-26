import os
import re
import pandas as pd
from rapidfuzz.fuzz import ratio

ROOT = "student_resource/dataset"
TRAIN = f"{ROOT}/train"
TEST = f"{ROOT}/test"
OUT = "output"

os.makedirs(OUT, exist_ok=True)


def norm(x):
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = x.replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def load(path):
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


print("Loading test data...")

s1 = load(f"{TEST}/test_source1.tsv")
s2 = load(f"{TEST}/test_source2.tsv")
s3 = load(f"{TEST}/test_source3.tsv")

for df in (s1, s2, s3):
    df["n"] = df["business_name"].map(norm)
    df["a"] = df["business_address"].map(norm)

print("Building exact indexes...")


def make_index(df):
    idx_n = {}
    idx_a = {}
    idx_na = {}

    for i, (eid, n, a) in enumerate(
        zip(df.entity_id, df.n, df.a)
    ):
        if n:
            idx_n.setdefault(n, []).append(eid)

        if a:
            idx_a.setdefault(a, []).append(eid)

        if n and a:
            idx_na.setdefault(n + "||" + a, []).append(eid)

    return idx_n, idx_a, idx_na


n2, a2, na2 = make_index(s2)
n3, a3, na3 = make_index(s3)

print("Indexes ready.")
print("Generating matches...")


def retrieve(row, indexes):
    nidx, aidx, naidx = indexes

    n = row.n
    a = row.a

    candidates = set()

    if n and n in nidx:
        candidates.update(nidx[n])

    if a and a in aidx:
        candidates.update(aidx[a])

    if n and a:
        key = n + "||" + a
        if key in naidx:
            candidates.update(naidx[key])

    return candidates


results = []

for k, row in enumerate(s1.itertuples(index=False), 1):

    candidates = []

    c2 = retrieve(row, (n2, a2, na2))
    c3 = retrieve(row, (n3, a3, na3))

    # Exact normalized blocking is intentionally conservative.
    # If exact name/address retrieves candidates, retain them.
    candidates.extend(c2)
    candidates.extend(c3)

    # Remove duplicates while preserving source IDs.
    candidates = list(dict.fromkeys(candidates))

    # If nothing was retrieved, predict no match.
    if not candidates:
        matched = ""
    else:
        matched = ",".join(candidates)

    results.append((row.entity_id, matched))

    if k % 100000 == 0:
        print(f"Processed {k:,}/{len(s1):,}")

out = pd.DataFrame(
    results,
    columns=["source1_entity_id", "matched_entity_ids"]
)

out.to_csv(
    f"{OUT}/matching_results.tsv",
    sep="\t",
    index=False
)

print("DONE")
print(f"Output: {OUT}/matching_results.tsv")
print(f"Rows: {len(out):,}")