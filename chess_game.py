# chess_game.py
import chess
import torch
import numpy as np

# takes a fen and returns a 14x8x8 pytorch tensor
def fen_to_tensor(fen):
    board = chess.Board(fen)
    planes = np.zeros((14, 8, 8), dtype=np.float32)

    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    # White = 0–5, Black = 6–11
    for sq, pc in board.piece_map().items():
        p = piece_map[pc.piece_type] + (0 if pc.color else 6)
        r, c = divmod(sq, 8)
        planes[p, 7 - r, c] = 1

    # Side to move plane
    planes[12] = 1 if board.turn == chess.WHITE else 0

    # Castling rights plane
    planes[13] = 0
    if board.has_kingside_castling_rights(True):  planes[13, 0, 7] = 1
    if board.has_queenside_castling_rights(True): planes[13, 0, 0] = 1
    if board.has_kingside_castling_rights(False): planes[13, 7, 7] = 1
    if board.has_queenside_castling_rights(False):planes[13, 7, 0] = 1

    return torch.tensor(planes)


def board_to_tensor(board: chess.Board):
    """
    Converts a python-chess board to a (14,8,8) tensor.
    """

    planes = np.zeros((14, 8, 8), dtype=np.float32)

    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    # White pieces 0–5, black pieces 6–11
    for square, piece in board.piece_map().items():
        plane = piece_map[piece.piece_type] + (0 if piece.color else 6)
        r, c = divmod(square, 8)
        planes[plane, 7 - r, c] = 1

    # Side to move plane
    planes[12] = np.ones((8, 8)) if board.turn == chess.WHITE else np.zeros((8, 8))

    # Repetition or castling rights plane (optional)
    planes[13] = np.zeros((8, 8))
    if board.has_kingside_castling_rights(True): planes[13][0][7] = 1
    if board.has_queenside_castling_rights(True): planes[13][0][0] = 1
    if board.has_kingside_castling_rights(False): planes[13][7][7] = 1
    if board.has_queenside_castling_rights(False): planes[13][7][0] = 1

    return torch.tensor(planes)


ALL_MOVES = list(range(4672))  # fixed move encoding


def move_to_index(move: chess.Move):
    """
    Converts a move to an index 0..4671
    Format: from_square * 73 + move_type
    You can pick any encoding as long as it's consistent.
    """
    return move.from_square * 64 + move.to_square  # 4096 moves + promotions etc.


def index_to_move(idx: int, board: chess.Board):
    """
    Reverse of move_to_index.
    """
    from_sq = idx // 64
    to_sq = idx % 64
    return chess.Move(from_sq, to_sq)

# just running this file
if __name__ == "__main__":
    # andrew's fen
    fen = "2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - -"
    tensor = fen_to_tensor(fen)
    print("Shape:", tensor.shape)
    print(tensor)