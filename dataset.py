import numpy as np
from chess_game import fen_to_tensor
from torch.utils.data import Dataset

# this is going to take in a pandas dataframe (variable is df) and return a dataset with many (fen, eval) pairs
# the data will be downloaded from https://www.kaggle.com/datasets/lichess/chess-evaluations 
class ChessEvalDataset(Dataset):
    def __init__(self, df):
        self.fens = df["fen"].values
        # this normalizes centipawns to [-10, +10]
        self.cps = df["cp"].values.astype(np.float32) / 1000.0 

    # like how many entries
    def __len__(self):
        return len(self.fens)
    
    # access ONE index of this dataset
    def __getitem__(self, idx):
        fen = self.fens[idx]
        cp = self.cps[idx]

        board_tensor = fen_to_tensor(fen)
        return board_tensor, cp