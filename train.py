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
from tqdm import tqdm
import time
import multiprocessing  # To get the number of CPU cores

# imports the neural network architecture - can edit within model.py to change hyperparameters
from model import EvalNet, ResBlock

# imports fen to tensor function
from chess_game import fen_to_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset to load data from single unified .pt file
class ChessUnifiedDataset(Dataset):
    def __init__(self, data_file):
        """
        Loads all training data from a single .pt file.
        """
        self.Xs, self.Ys = torch.load(data_file)
        print(f"Loaded {len(self.Ys)} samples from {data_file}")

    def __len__(self):
        return len(self.Ys)

    def __getitem__(self, idx):
        return self.Xs[idx], self.Ys[idx]

# for loading a previously saved model, or by default, the last finished one
def load_model(path="checkpoint_epoch_24.pt"):
    net = EvalNet().to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net

# returns a score in pawn units (-3 to +3)
def evaluate_position(fen, net=None):
    if net is None:
        net = load_model()

    t = fen_to_tensor(fen).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        score = net(t).item()
    return score

# Training loop for the unified data file
def train_model_unified(net, opt, loss_fn, data_file, epochs=1, batch_size=256, num_workers=4):
    print(f"\nLoading training data from: {data_file}")

    dataset = ChessUnifiedDataset(data_file)

    # DataLoader with parallelism (use num_workers > 0 for parallel loading)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    print(f"Loaded {len(dataset)} positions from {data_file}")
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        # Progress bar for training loop
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)

        for x, y in pbar:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            opt.zero_grad()
            pred = net(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

            # Update progress bar with current loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}, Loss {avg_loss:.5f}")

        # Save checkpoint per epoch
        torch.save(net.state_dict(), f"checkpoint_epoch{epoch}.pt")

# Data directory
data_dir = "./datas/"

# Training loop for all data from a unified .pt file
def train_model():
    # Confirm if we're using GPU or CPU
    print(f"Training on {DEVICE}.")

    # Dynamically set the number of workers to the number of available CPU cores
    num_workers = multiprocessing.cpu_count()

    # Print the number of workers
    print(f"Using {num_workers} CPU workers for data loading.")

    net = EvalNet().to(DEVICE)

    # Using Adam optimizer, but you could use SGD if needed
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    # MSE Loss function (mean squared error)
    loss_fn = nn.MSELoss()

    # Load the unified .pt file with all the training data
    data_file = os.path.join(data_dir, "chunks/train_chunk_000.pt")

    # Training over all epochs
    train_model_unified(net, opt, loss_fn, data_file=data_file, epochs=3, batch_size=256, num_workers=num_workers)

    # Final message after training completes
    print("Training complete.")
    torch.save(net.state_dict(), "final_model.pt")

# Main entry point
if __name__ == "__main__":
    train_model()

    """
    # andrew's fen
    fen = "2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - -"
    tensor = fen_to_tensor(fen)
    print("Shape:", tensor.shape)
    print(tensor)
    """