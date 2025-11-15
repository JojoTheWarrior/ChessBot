import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import chess
import numpy as np
import random
import pandas as pd
import os
from get_data import load_local_parquets

# this just gets the index from the kaggle URL that we pass in
from get_shard_number import get_index

# is able to take in pandas dataframes downloaded from kaggle and stream it 
from dataset import ChessEvalDataset

# imports the neural network architecture - can edit within model.py to change hyperparameters
from model import EvalNet, ResBlock

# imports fen to tensor function
from chess_game import fen_to_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# for loading a previously saved model, or by default, the last finished one
def load_model(path="checkpoint_epoch_24.pt"):
    net = EvalNet().to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net

# returns a score in pawn units (-3 to +3)
# also i dont like how this is loading the model each time - can we try to load it once then run a while loop? for the actual game of course. or send up an endpoint with flask maybe
def evaluate_position(fen, net=None):
    if net is None:
        net = load_model()

    t = fen_to_tensor(fen).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        score = net(t).item()
    return score

# training loop for ONE shard
def train_model_one_shard(net, opt, loss_fn, shard_path, epochs=1, batch_size=256):
    shard_index = get_index(shard_path)
    print(f"\nLoading shard: {shard_index}")
    df = pd.read_parquet(shard_path)

    dataset = ChessEvalDataset(df)

    # on laptop, always use num_workers=0; on paperspace, use 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"Loaded {len(dataset)} positions from Shard {shard_index}")
    
    # training loop
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            opt.zero_grad()
            pred = net(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Shard {shard_index}, Epoch {epoch}, Loss {avg_loss:.5f}")

        # save checkpoint per epoch
        torch.save(net.state_dict(), f"checkpoint_{shard_index}_epoch{epoch}.pt")

# data directory
data_dir = "./datas/"


# training loop for ALL shards
def train_model():
    net = EvalNet().to(DEVICE)

    # shouldn't we be using SGD instead of Adam here?
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    # our loss function is the mse (mean squared error) between what our model outputted and the eval bar from andrew's dataset
    loss_fn = nn.MSELoss()

    # loading our shards
    shard_paths = [
        data_dir + f"train-{i:05d}-of-00016.parquet"
        for i in range(16)
    ]

    # training over all 16 shards
    for shard in shard_paths:
        if not os.path.exists(shard):
            print(f"Missing shard {shard}, skipping.")
            continue

        train_model_one_shard(net, opt, loss_fn, shard_path=shard, epochs=1)

        # OPTIONAL: delete shard to save space
        os.remove(shard)
        print(f"Deleted shard {shard} to save space")

    # done message
    print("Training complete.")
    torch.save(net.state_dict(), "final_model.pt")


# just main code for testing
if __name__ == "__main__":
    train_model()

    """
    # andrew's fen
    fen = "2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - -"
    tensor = fen_to_tensor(fen)
    print("Shape:", tensor.shape)
    print(tensor)
    """