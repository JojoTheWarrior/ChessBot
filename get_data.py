# pip install "kagglehub[pandas-datasets]" pandas
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

def to_int_or_none(x):
    if pd.isna(x):
        return None
    try:
        return int(x)
    except Exception:
        return None

def row_flat(rec):
    return {
        "fen": rec.fen,
        "depth": to_int_or_none(rec.depth),
        "knodes": to_int_or_none(getattr(rec, "knodes", None)),
        "cp": to_int_or_none(getattr(rec, "cp", None)),
        "mate": to_int_or_none(getattr(rec, "mate", None)),
        "line": getattr(rec, "line", None),
    }

def row_raw_style(rec):
    # Build minimal raw-lichess-like: one eval with one PV
    pv = {"line": getattr(rec, "line", None) or ""}
    mate = getattr(rec, "mate", None)
    cp = getattr(rec, "cp", None)
    if mate is not None and not pd.isna(mate):
        pv["mate"] = to_int_or_none(mate)
    elif cp is not None and not pd.isna(cp):
        pv["cp"] = to_int_or_none(cp)
    obj = {
        "fen": rec.fen,
        "evals": [{
            "pvs": [pv],
            "knodes": to_int_or_none(getattr(rec, "knodes", None)),
            "depth": to_int_or_none(getattr(rec, "depth", None)),
        }]
    }
    return obj

def load_parquet(parquet_file):
    """Load a single parquet file and return as DataFrame."""
    df = pd.read_parquet(parquet_file)
    
    # Select only known columns if present
    wanted = [c for c in ["fen", "cp", "mate", "depth", "knodes", "line"] if c in df.columns]
    if wanted:
        df = df[wanted]
    
    return df

def format_rows(df, min_depth=None, only_mate=False, only_nonmate=False, limit=None):
    """Format DataFrame rows according to the desired schema."""
    # Optional filters
    if min_depth is not None and "depth" in df.columns:
        df = df[df["depth"].apply(lambda d: pd.notna(d) and int(d) >= min_depth)]
    if only_mate and "mate" in df.columns:
        df = df[df["mate"].notna()]
    if only_nonmate and "mate" in df.columns:
        df = df[df["mate"].isna()]
    
    n = 0
    for rec in df.itertuples(index=False):
        yield row_flat(rec)
        n += 1
        if limit is not None and n >= limit:
            break

def load_local_parquets(directory, min_depth=None, only_mate=False, only_nonmate=False, limit=None):
    """Load all parquet files from a local directory and yield formatted rows."""
    path = Path(directory)
    parquet_files = sorted(path.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {directory}", file=sys.stderr)
        return
    
    n = 0
    for parquet_file in parquet_files:
        df = load_parquet(parquet_file)
        for row in format_rows(df, min_depth, only_mate, only_nonmate, None):
            yield row
            n += 1
            if limit is not None and n >= limit:
                return

def main():
    ap = argparse.ArgumentParser(description="Load chess evaluation data from local parquets or kagglehub.")
    ap.add_argument("--local", help="Load from local directory with parquet files (e.g., data/lichess/).")
    ap.add_argument("--handle", default="lichess/chess-evaluations", help="Kaggle dataset handle.")
    ap.add_argument("--file-path", help="Exact file path inside the Kaggle dataset (e.g., shard .csv/.parquet).")
    ap.add_argument("-o", "--output", default="-", help="Output JSONL path ('-' = stdout).")
    ap.add_argument("--raw-style", action="store_true", help="Emit raw-evals-like objects with evals→pvs.")
    ap.add_argument("--min-depth", type=int, default=None, help="Keep rows with depth >= this.")
    ap.add_argument("--only-mate", action="store_true", help="Keep only rows where mate is present.")
    ap.add_argument("--only-nonmate", action="store_true", help="Keep only rows where mate is null.")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N rows.")
    args = ap.parse_args()

    if args.only_mate and args.only_nonmate:
        print("--only-mate and --only-nonmate are mutually exclusive.", file=sys.stderr)
        sys.exit(2)

    # Load data from local or Kaggle
    if args.local:
        rows = load_local_parquets(args.local, args.min_depth, args.only_mate, args.only_nonmate, args.limit)
    else:
        if not args.file_path:
            print("--file-path is required when not using --local", file=sys.stderr)
            sys.exit(2)
        
        # Load one file from the dataset into a DataFrame
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            args.handle,
            args.file_path,
            pandas_kwargs={"dtype": {"fen": "string", "line": "string"}}
        )

        # Try to keep just known columns if present
        wanted = [c for c in ["fen", "cp", "mate", "depth", "knodes", "line"] if c in df.columns]
        if wanted:
            df = df[wanted]

        # Optional filters
        if args.min_depth is not None and "depth" in df.columns:
            df = df[df["depth"].apply(lambda d: pd.notna(d) and int(d) >= args.min_depth)]
        if args.only_mate and "mate" in df.columns:
            df = df[df["mate"].notna()]
        if args.only_nonmate and "mate" in df.columns:
            df = df[df["mate"].isna()]

        def row_generator():
            n = 0
            conv = row_raw_style if args.raw_style else row_flat
            for rec in df.itertuples(index=False):
                yield conv(rec)
                n += 1
                if args.limit is not None and n >= args.limit:
                    break
        
        rows = row_generator()

    # Write JSONL
    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    try:
        conv = row_raw_style if args.raw_style else row_flat
        for obj in rows:
            if args.raw_style or args.local:
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        if out is not sys.stdout:
            out.close()

if __name__ == "__main__":
    main()
