
# inputs look like:
# https://www.kaggle.com/datasets/lichess/chess-evaluations?select=train-00000-of-00016.parquet
import re

def get_index(url: str) -> str:
    """
    Extracts the 5-digit index from filenames like:
    train-00000-of-00016.parquet
    """
    m = re.search(r"-(\d{5})-of-\d{5}\.parquet$", url)
    if not m:
        raise ValueError("Index not found in URL.")
    return m.group(1)
