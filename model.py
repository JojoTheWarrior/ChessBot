import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import chess
import numpy as np
import random

# this uses a "skip connection" to add the input to the output
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)

# this is my neural network
class EvalNet(nn.Module):
    def __init__(self, channels=96, blocks=10):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(14, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.res = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1)           # outputs centipawn eval normalized
        )

    def forward(self, x):
        x = self.input(x)
        x = self.res(x)
        x = self.head(x)
        return x.squeeze(-1)