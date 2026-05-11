"""
sample_categories.py
--------------------
Reads the top-20 category Parquet file and samples exactly 2500 records
per category, producing a balanced 50,000-row dataset for ML training.

Since records can have multiple tags (e.g. "cs.LG cs.AI"), each record
is assigned to exactly ONE category using a priority order (rank 1 first).
This avoids counting the same record toward multiple category quotas.

Usage (after `uv add pandas pyarrow`):
  python sample_categories.py --input arxiv_top20.parquet --output arxiv_balanced.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SAMPLES_PER_CAT = 2500

# Ordered by rank from the top-20 output — used to assign each record to
# its single highest-priority category when it has multiple tags.
TOP_20_ORDERED = [
    "cs.LG", "hep-ph", "cs.CV", "hep-th", "quant-ph",
    "cs.AI", "gr-qc", "cond-mat.mtrl-sci", "cs.CL", "astro-ph",
    "cond-mat.mes-hall", "math-ph", "math.MP", "cond-mat.str-el",
    "cond-mat.stat-mech", "math.CO", "stat.ML", "astro-ph.GA",
    "astro-ph.CO", "math.AP",
]


# ──────────────────────────────────────────────
# 1. Assign each record a single primary category
# ──────────────────────────────────────────────

def assign_primary_category(df: pd.DataFrame, ordered_cats: list[str]) -> pd.DataFrame:
    """
    Adds a 'primary_category' column. For each record, the primary category
    is the highest-ranked tag it contains (rank = position in ordered_cats).
    Records with no matching tag are dropped (shouldn't happen in a
    pre-filtered file, but guards against edge cases).
    """
    rank_map = {cat: i for i, cat in enumerate(ordered_cats)}

    def best_tag(cell):
        tags = str(cell).split() if pd.notna(cell) else []
        ranked = [(rank_map[t], t) for t in tags if t in rank_map]
        return min(ranked)[1] if ranked else None

    df = df.copy()
    df["primary_category"] = df["categories"].apply(best_tag)
    before = len(df)
    df = df.dropna(subset=["primary_category"]).reset_index(drop=True)
    if len(df) < before:
        print(f"[WARN] Dropped {before - len(df)} rows with no recognised top-20 tag.")
    return df


# ──────────────────────────────────────────────
# 2. Sample exactly N records per category
# ──────────────────────────────────────────────

def sample_per_category(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Groups by primary_category and samples exactly n records from each.
    Uses a fixed random seed for reproducibility.
    Warns if any category has fewer than n records.
    """
    groups = []
    for cat, group in df.groupby("primary_category"):
        if len(group) < n:
            print(f"[WARN] '{cat}' has only {len(group):,} records (< {n}). Taking all.")
            groups.append(group)
        else:
            groups.append(group.sample(n=n, random_state=seed))

    return pd.concat(groups, ignore_index=True)


# ──────────────────────────────────────────────
# 3. Write to Parquet
# ──────────────────────────────────────────────

def write_parquet(df: pd.DataFrame, output_path: Path, compression: str = "zstd"):
    # Drop the helper column before saving
    df = df.drop(columns=["primary_category"])
    table = pa.Table.from_pandas(df, preserve_index=False)
    compression_level = 3 if compression == "zstd" else None
    pq.write_table(table, output_path, compression=compression,
                   compression_level=compression_level)


# ──────────────────────────────────────────────
# 4. Summary
# ──────────────────────────────────────────────

def print_summary(df_sampled: pd.DataFrame, output_path: Path):
    size_mb = output_path.stat().st_size / (1024 ** 2)

    print("\n── Sample Distribution ──────────────────")
    counts = df_sampled["primary_category"].value_counts().sort_index()
    for cat, count in counts.items():
        print(f"  {cat:<25} {count:>6,} records")
    print(f"\n  Total rows : {len(df_sampled):,}")
    print(f"  File size  : {size_mb:.2f} MB")
    print("─────────────────────────────────────────\n")


# ──────────────────────────────────────────────
# 5. CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample N records per category for a balanced ML dataset."
    )
    parser.add_argument("--input",  "-i", required=True, help="Path to top-20 .parquet file")
    parser.add_argument("--output", "-o", required=True, help="Path for output .parquet file")
    parser.add_argument("--samples", "-n", type=int, default=SAMPLES_PER_CAT,
                        help=f"Records per category (default: {SAMPLES_PER_CAT})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--compression", "-c", default="zstd",
                        choices=["snappy", "gzip", "brotli", "zstd", "none"],
                        help="Parquet compression codec (default: zstd level 3)")
    parser.add_argument("--summary", "-s", action="store_true",
                        help="Print per-category sample counts after writing")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/4] Reading  : {input_path}")
    df = pd.read_parquet(input_path)
    print(f"       Loaded {len(df):,} rows.")

    print(f"[2/4] Assigning primary category per record...")
    df = assign_primary_category(df, TOP_20_ORDERED)

    print(f"[3/4] Sampling {args.samples:,} records per category (seed={args.seed})...")
    df_sampled = sample_per_category(df, args.samples, seed=args.seed)

    compression = None if args.compression == "none" else args.compression
    print(f"[4/4] Writing  : {output_path}  (compression={compression})")
    write_parquet(df_sampled, output_path, compression=compression)

    print(f"\n✓ Done. {len(df_sampled):,} records saved to: {output_path}")

    if args.summary:
        print_summary(df_sampled, output_path)


if __name__ == "__main__":
    main()