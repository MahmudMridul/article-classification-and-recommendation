"""
json_to_parquet.py

Keeps   : id, title, abstract, categories
Drops   : submitter, authors, comments, journal-ref, doi, versions

Usage (after `uv add pandas pyarrow`):
  python json_to_parquet.py --input arxiv.json --output arxiv.parquet
  python json_to_parquet.py --input arxiv.jsonl --output arxiv.parquet --schema
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


KEEP_COLUMNS = ["id", "title", "abstract", "categories"]


# ──────────────────────────────────────────────
# 1. Load JSON into a DataFrame
# ──────────────────────────────────────────────

def iter_chunks(path: Path, chunk_size: int):
    """
    Yields DataFrames of `chunk_size` rows at a time without loading
    the entire file into memory.

    For JSONL (one object per line): reads line by line.
    For a JSON array: falls back to full load (only safe for small files).
    """
    with path.open("r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        # JSONL path: stream line by line
        if first_char == "{" or path.suffix == ".jsonl":
            chunk = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk.append(json.loads(line))
                if len(chunk) == chunk_size:
                    yield pd.DataFrame(chunk)
                    chunk = []
            if chunk:
                yield pd.DataFrame(chunk)

        # JSON array path: load fully (only safe for small files)
        else:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            for i in range(0, len(data), chunk_size):
                yield pd.DataFrame(data[i : i + chunk_size])


# ──────────────────────────────────────────────
# 2. Select and clean relevant columns
# ──────────────────────────────────────────────

def select_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only ML-relevant columns and cleans their values.
    - Drops all columns not in KEEP_COLUMNS
    - Strips whitespace and normalises internal whitespace in text fields
    - Ensures 'categories' is a clean string (space-separated tags)
    """
    # Drop columns not in KEEP_COLUMNS (ignore if already missing)
    existing = [col for col in KEEP_COLUMNS if col in df.columns]
    missing  = [col for col in KEEP_COLUMNS if col not in df.columns]

    if missing:
        print(f"[WARN] Expected columns not found in input: {missing}")

    df = df[existing].copy()

    # Clean text columns: strip outer whitespace, collapse internal whitespace
    for col in ["title", "abstract"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

    # Clean categories: it may be a string like "cs.LG cs.AI" — keep as-is,
    # just strip whitespace. If it's a list, join to space-separated string.
    if "categories" in df.columns:
        df["categories"] = df["categories"].apply(
            lambda x: " ".join(x).strip() if isinstance(x, list)
            else str(x).strip() if pd.notna(x) else ""
        )

    # id: ensure string, strip whitespace
    if "id" in df.columns:
        df["id"] = df["id"].astype(str).str.strip()

    # Drop rows where both title and abstract are empty (junk rows)
    if "title" in df.columns and "abstract" in df.columns:
        before = len(df)
        df = df[~((df["title"] == "") & (df["abstract"] == ""))]
        dropped = before - len(df)
        if dropped:
            print(f"[WARN] Dropped {dropped} rows with empty title and abstract.")

    return df.reset_index(drop=True)


# ──────────────────────────────────────────────
# 3. Write to Parquet
# ──────────────────────────────────────────────

def write_parquet_chunked(output_path: Path, chunks, compression: str = "zstd"):
    """
    Writes an iterable of (cleaned) DataFrames to a single Parquet file
    incrementally using ParquetWriter. Each chunk becomes one row group.
    Memory at any point: one chunk, not the full dataset.
    """
    compression_level = 3 if compression == "zstd" else None
    writer = None
    total_rows = 0

    for df in chunks:
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(
                output_path, table.schema,
                compression=compression,
                compression_level=compression_level,
            )
        writer.write_table(table)
        total_rows += len(df)
        print(f"       Written {total_rows:,} rows...", end="\r")

    if writer:
        writer.close()

    print()  # newline after the \r progress line
    return total_rows


# ──────────────────────────────────────────────
# 4. Print schema summary (optional)
# ──────────────────────────────────────────────

def print_schema(sample_df: pd.DataFrame, parquet_path: Path):
    meta     = pq.read_metadata(parquet_path)
    size_mb  = parquet_path.stat().st_size / (1024 ** 2)

    print("\n── Parquet File ─────────────────────────")
    print(f"  Path       : {parquet_path}")
    print(f"  Total rows : {meta.num_rows:,}")
    print(f"  Row groups : {meta.num_row_groups}")
    print(f"  Size       : {size_mb:.2f} MB")

    print("\n── Column Info (sample from row 0) ──────")
    for col in sample_df.columns:
        val = sample_df[col].iloc[0] if not sample_df.empty else "N/A"
        val_str = str(val)[:60] + "..." if len(str(val)) > 60 else str(val)
        print(f"  {col:<15} sample: {val_str}")
    print("─────────────────────────────────────────\n")


# ──────────────────────────────────────────────
# 5. CLI entry point
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ArXiv JSON/JSONL to Parquet, keeping ML-relevant columns."
    )
    parser.add_argument("--input",  "-i", required=True, help="Path to input JSON or JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Path for output .parquet file")
    parser.add_argument("--compression", "-c", default="zstd",
                        choices=["snappy", "gzip", "brotli", "zstd", "none"],
                        help="Parquet compression codec (default: zstd level 3)")
    parser.add_argument("--chunk-size", "-n", type=int, default=10_000,
                        help="Records per chunk (default: 10000). Lower if RAM is tight.")
    parser.add_argument("--schema", "-s", action="store_true",
                        help="Print column info and file stats after conversion")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    compression = None if args.compression == "none" else args.compression
    print(f"[1/3] Reading  : {input_path}  (chunk_size={args.chunk_size:,})")
    print(f"[2/3] Cleaning : selecting columns {KEEP_COLUMNS}")
    print(f"[3/3] Writing  : {output_path}  (compression={compression})")

    # Pipeline: read a chunk -> clean it -> write it, never holding full dataset in RAM
    chunks = (
        select_and_clean(chunk)
        for chunk in iter_chunks(input_path, args.chunk_size)
    )
    total = write_parquet_chunked(output_path, chunks, compression=compression)

    print(f"\n✓ Done. {total:,} records saved to: {output_path}")

    if args.schema:
        sample_df = pd.read_parquet(output_path).head(1)
        print_schema(sample_df, output_path)


if __name__ == "__main__":
    main()