import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import chess
import numpy as np
import random

# imports the neural network architecture - can edit within model.py to change hyperparameters
from model import EvalNet, ResBlock

# imports fen to tensor function
from chess_game import fen_to_tensor



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# training loop
def train_model():
    print("Loading dataset...")
    # dataset = ChessEvalDataset("training_data.jsonl")
    # check what num_workers does
    # loader = DataLoader(dataset, batch_size = 256, shuffle=True, num_workers=4)

    # print(f"We have loaded ${len(dataset)} positions")

    net = EvalNet().to(DEVICE)
    # shouldn't we be using SGD instead of Adam here?
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# just main code for testing
if __name__ == "__main__":
    # andrew's fen
    fen = "2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - -"
    tensor = fen_to_tensor(fen)
    print("Shape:", tensor.shape)
    print(tensor)