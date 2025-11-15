# this file is for making altercations to files and playing with things
import pandas as pd
import time

# process.py

import torch
import pandas as pd
from chess_game import fen_to_tensor
from tqdm import tqdm
import numpy as np

def preprocess_shard(parquet_path, output_path):
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    
    Xs = []
    Ys = df["cp"].values.astype(np.float32) / 1000.0

    for fen in tqdm(df["fen"].values, desc="Converting FEN → tensor"):
        Xs.append(fen_to_tensor(fen))   # expensive step → done once

    Xs = torch.stack(Xs)
    Ys = torch.tensor(Ys)

    print(f"Saving to {output_path}...")
    torch.save((Xs, Ys), output_path)
    print("Done.")


if __name__ == "__main__":
    preprocess_shard(
        "datas/train-00000-of-00016.parquet",
        "datas/train-00000.pt"
    )


"""
start_time = time.time()
df = pd.read_parquet("./datas/train-00000-of-00016.parquet", engine="pyarrow")
end_time = time.time()

print(f"Loading Shard {1} took {end_time - start_time} seconds")

subset = df.head(100)

subset.to_parquet("./datas/subset_100.parquet")
"""