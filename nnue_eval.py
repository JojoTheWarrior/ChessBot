# nnue_eval.py
# Minimal standalone NNUE evaluator for Python-chess boards.
# Supports Stockfish-format .nnue files.

import struct
import numpy as np
import chess

# Layer definitions -------------------------------------------------------------

def read_int32(f):
    return struct.unpack("<i", f.read(4))[0]

def read_float32(f):
    return struct.unpack("<f", f.read(4))[0]

def read_vector_f32(f, n):
    return np.frombuffer(f.read(4 * n), dtype=np.float32)

def read_vector_i16(f, n):
    return np.frombuffer(f.read(2 * n), dtype=np.int16)

class Linear:
    def __init__(self, w, b):
        self.w = w
        self.b = b
    def __call__(self, x):
        return x @ self.w + self.b

class Relu:
    def __call__(self, x):
        return np.maximum(x, 0)

# NNUE evaluator ---------------------------------------------------------------

class NNUE:
    def __init__(self, path):
        with open(path, "rb") as f:
            self._load(f)

    def _load(self, f):
        # Header (magic/version/etc.)
        magic = f.read(4)
        if magic not in [b"NNUE", b"NNUB"]:
            raise ValueError("Not a valid NNUE file")

        version = read_int32(f)
        net_hash = read_int32(f)
        f.seek(12, 1)  # skip 3 unused ints

        # Input layer sizes (Stockfish format)
        self.input_size = read_int32(f)
        self.feature_dims = read_int32(f)
        self.hidden_size = read_int32(f)
        
        # Load input weights (i16)
        self.in_weights = read_vector_i16(f, self.input_size * self.hidden_size).reshape(self.input_size, self.hidden_size)
        self.in_bias = read_vector_i16(f, self.hidden_size).astype(np.float32)

        # Hidden → hidden
        h2_w = read_vector_f32(f, self.hidden_size * self.hidden_size).reshape(self.hidden_size, self.hidden_size)
        h2_b = read_vector_f32(f, self.hidden_size)
        self.hidden2 = Linear(h2_w, h2_b)

        # Hidden → output
        out_w = read_vector_f32(f, self.hidden_size)
        out_b = read_float32(f)
        self.out_w = out_w
        self.out_b = out_b

    # Input encoding ------------------------------------------------------------

    def _square_features(self, board):
        """Return list of active features for this board."""
        feats = []
        for sq, piece in board.piece_map().items():
            pc = piece.piece_type - 1
            color = 0 if piece.color == chess.WHITE else 1
            feats.append(pc * 2 * 64 + color * 64 + sq)
        return feats

    # Main eval ----------------------------------------------------------------

    def evaluate(self, board: chess.Board) -> float:
        feats = self._square_features(board)

        # Input accumulation
        x = np.copy(self.in_bias)
        for idx in feats:
            x += self.in_weights[idx]

        # Convert i16 → float32 and apply ReLU
        x = x.astype(np.float32)
        x = np.maximum(x, 0)

        # Hidden layer
        x = self.hidden2(x)
        x = np.maximum(x, 0)

        # Output
        score = x @ self.out_w + self.out_b
        return float(score)  # centipawns
