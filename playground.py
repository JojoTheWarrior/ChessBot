import numpy as np
from multiprocessing import Pool
from chess_game import fen_to_tensor
import pandas as pd
import torch
from tqdm import tqdm
import math
import os

def fen_to_tensor_np(fen):
    return fen_to_tensor(fen).numpy()

def preprocess_chunk(df_chunk, output_path):
    fens = df_chunk["fen"].values

    with Pool() as pool:
        tensors_np = list(tqdm(pool.imap(fen_to_tensor_np, fens, chunksize=100),
                               total=len(fens),
                               desc=f"Converting FEN → tensor ({os.path.basename(output_path)})"))

    Xs = torch.from_numpy(np.array(tensors_np))
    Ys = torch.tensor(df_chunk["cp"].values.astype(np.float32) / 1000.0)

    torch.save((Xs, Ys), output_path)

def preprocess_shard_in_chunks(parquet_path, output_dir, num_chunks=100):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    total_rows = len(df)
    chunk_size = math.ceil(total_rows / num_chunks)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i+1) * chunk_size, total_rows)
        df_chunk = df.iloc[start_idx:end_idx]

        if len(df_chunk) == 0:
            break

        output_path = os.path.join(output_dir, f"train_chunk_{i:03d}.pt")
        print(f"Processing chunk {i+1}/{num_chunks} rows {start_idx} to {end_idx} → {output_path}")
        preprocess_chunk(df_chunk, output_path)

def combine_pt_chunks(pt_files, output_path):
    X_list = []
    Y_list = []
    for pt_file in pt_files:
        Xs, Ys = torch.load(pt_file)
        X_list.append(Xs)
        Y_list.append(Ys)
    Xs_all = torch.cat(X_list, dim=0)
    Ys_all = torch.cat(Y_list, dim=0)
    torch.save((Xs_all, Ys_all), output_path)
    print(f"Saved combined dataset to {output_path}")

if __name__ == "__main__":
    parquet_path = "./datas/train-00000-of-00016.parquet"
    output_dir = "./datas/chunks"
    num_chunks = 100

    preprocess_shard_in_chunks(parquet_path, output_dir, num_chunks)

    # Later, when you want to combine:
    # import glob
    # pt_files = sorted(glob.glob(f"{output_dir}/train_chunk_*.pt"))
    # combine_pt_chunks(pt_files, "./datas/train_combined.pt")
