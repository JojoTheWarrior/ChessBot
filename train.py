import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import chess
import numpy as np
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

