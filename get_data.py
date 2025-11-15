import pandas as pd

def to_int_or_none(x):
    # Normalize numbers that may be strings or NA to Python int or None
    if pd.isna(x):
        return None
    try:
        return int(x)
    except Exception:
        return None

def get_field(rec, name, default=None):
    """
    Robust field accessor that works for:
    - pandas namedtuples from df.itertuples()
    - pandas Series (row from df.iloc[i])
    - dict-like rows
    """
    if hasattr(rec, name):
        return getattr(rec, name)
    try:
        return rec[name]
    except Exception:
        return default

def format_eval_row(rec):
    """
    Convert one dataset row into:
    {
        "fen": str,
        "depth": int|None,
        "knodes": int|None,
        "cp": int|None,
        "mate": int|None,
        "line": str|None
    }
    Assumes the dataset row includes the fields described by the Lichess evaluations schema. 
    """
    fen = get_field(rec, "fen")
    depth = to_int_or_none(get_field(rec, "depth"))
    knodes = to_int_or_none(get_field(rec, "knodes"))
    cp = to_int_or_none(get_field(rec, "cp"))
    mate = to_int_or_none(get_field(rec, "mate"))
    line = get_field(rec, "line")

    return {
        "fen": fen,
        "depth": depth,
        "knodes": knodes,
        "cp": cp,
        "mate": mate,
        "line": line,
    }

def iter_formatted_parquet_rows(filepath, limit=None):
    """
    Read a Parquet file and yield formatted dicts, one per row.
    Requires a Parquet engine such as pyarrow or fastparquet to be installed.
    """
    # Read only the needed columns to reduce I/O and memory
    cols = ["fen", "cp", "mate", "depth", "knodes", "line"]
    df = pd.read_parquet(filepath, columns=cols)  # needs pyarrow or fastparquet [web:153]
    n = 0
    for rec in df.itertuples(index=False):
        yield format_eval_row(rec)
        n += 1
        if limit is not None and n >= limit:
            break