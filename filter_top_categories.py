"""
filter_top_categories.py
------------------------
Reads an ArXiv Parquet file, finds the top 20 categories by record count,
and writes a filtered Parquet file containing only those records.

Note: the 'categories' column may contain multiple space-separated tags
per record (e.g. "cs.LG cs.AI"). Each record is counted once per tag it
contains, so ranking is by how many records mention that tag.

Usage (after `uv add pandas pyarrow`):
  python filter_top_categories.py --input arxiv.parquet --output arxiv_top20.parquet
  python filter_top_categories.py --input arxiv.parquet --output arxiv_top20.parquet --schema
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


TOP_N = 10


# ──────────────────────────────────────────────
# 1. Find top N categories by record count
# ──────────────────────────────────────────────

def get_top_categories(df: pd.DataFrame, n: int) -> list[str]:
    """
    Explodes the space-separated 'categories' column so each tag gets its
    own row, counts occurrences, and returns the top N tag strings.

    Example: "cs.LG cs.AI" → two separate entries: "cs.LG", "cs.AI"
    """
    tag_counts = (
        df["categories"]
        .dropna()
        .str.split()           # split "cs.LG cs.AI" → ["cs.LG", "cs.AI"]
        .explode()             # one tag per row
        .str.strip()
        .value_counts()        # count how many records mention each tag
    )

    top = tag_counts.head(n).index.tolist()

    print(f"\n── Top {n} Categories ────────────────────")
    for i, (tag, count) in enumerate(tag_counts.head(n).items(), 1):
        print(f"  {i:>2}. {tag:<20} {count:>10,} records")
    print("─────────────────────────────────────────\n")

    return top


# ──────────────────────────────────────────────
# 2. Filter: keep rows that contain any top tag
# ──────────────────────────────────────────────

def filter_by_categories(df: pd.DataFrame, top_cats: list[str]) -> pd.DataFrame:
    """
    Keeps a row if ANY of its categories appear in top_cats.
    A record tagged "cs.LG cs.AI" is kept if either tag is in the top list.
    """
    top_set = set(top_cats)

    mask = df["categories"].apply(
        lambda cell: bool(top_set & set(str(cell).split()))
        if pd.notna(cell) else False
    )

    return df[mask].reset_index(drop=True)


# ──────────────────────────────────────────────
# 3. Write filtered data to Parquet
# ──────────────────────────────────────────────

def write_parquet(df: pd.DataFrame, output_path: Path, compression: str = "zstd"):
    table = pa.Table.from_pandas(df, preserve_index=False)
    compression_level = 3 if compression == "zstd" else None
    pq.write_table(table, output_path, compression=compression,
                   compression_level=compression_level)


# ──────────────────────────────────────────────
# 4. Schema summary
# ──────────────────────────────────────────────

def print_schema(output_path: Path):
    meta    = pq.read_metadata(output_path)
    size_mb = output_path.stat().st_size / (1024 ** 2)

    print("── Output Parquet ───────────────────────")
    print(f"  Path       : {output_path}")
    print(f"  Total rows : {meta.num_rows:,}")
    print(f"  Row groups : {meta.num_row_groups}")
    print(f"  Size       : {size_mb:.2f} MB")
    print("─────────────────────────────────────────\n")


# ──────────────────────────────────────────────
# 5. CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter ArXiv Parquet to top N categories by record count."
    )
    parser.add_argument("--input",  "-i", required=True, help="Path to input .parquet file")
    parser.add_argument("--output", "-o", required=True, help="Path for output .parquet file")
    parser.add_argument("--top",    "-t", type=int, default=TOP_N,
                        help=f"Number of top categories to keep (default: {TOP_N})")
    parser.add_argument("--compression", "-c", default="zstd",
                        choices=["snappy", "gzip", "brotli", "zstd", "none"],
                        help="Parquet compression codec (default: zstd level 3)")
    parser.add_argument("--schema", "-s", action="store_true",
                        help="Print output file stats after writing")
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
    print(f"       Loaded {len(df):,} rows, {len(df.columns)} columns.")

    print(f"[2/4] Ranking categories...")
    top_cats = get_top_categories(df, args.top)

    print(f"[3/4] Filtering rows that contain at least one top-{args.top} category...")
    df_filtered = filter_by_categories(df, top_cats)
    dropped = len(df) - len(df_filtered)
    print(f"       Kept {len(df_filtered):,} rows  |  dropped {dropped:,} rows.")

    compression = None if args.compression == "none" else args.compression
    print(f"[4/4] Writing  : {output_path}  (compression={compression})")
    write_parquet(df_filtered, output_path, compression=compression)

    print(f"\n✓ Done. {len(df_filtered):,} records saved to: {output_path}")

    if args.schema:
        print_schema(output_path)


if __name__ == "__main__":
    main()