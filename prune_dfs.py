# next_move_pruning.py
# Requires: pip install chess
import chess
import time
from typing import List, Optional, Tuple
from stockfish_eval import evaluate_fen

# survivors from each layer = [a1, a2, ..., a20]
# fully branch eacy survivor
# find best move
# dp and dfs

# -----------------------
# Hard-coded branching factors a1..a20
# product <= 20,000 (approx)
# -----------------------
BRANCHING_DFS: List[int] = [
    5,  # a1
    4,  # a2
    3,  # a3
    3,  # a4
    2,  # a5
    2,  # a6
    2,  # a7
    1,  # a8
    1,  # a9
    1,  # a10
    1,  # a11
    1,  # a12
    1,  # a13
    1,  # a14
    1,  # a15
    1,  # a16
    1,  # a17
    1,  # a18
    1,  # a19
    1,  # a20
]
BRANCHING_BFS: List[int] = [
    5,  # a1
    10,  # a2
    15,  # a3
    20,  # a4
    25,  # a5
    30,  # a6
    35,  # a7
    40,  # a8
    40,  # a9
    40,  # a10
    40,  # a11
    40,  # a12
    40,  # a13
    40,  # a14
    40,  # a15
    40,  # a16
    40,  # a17
    40,  # a18
    40,  # a19
    40,  # a20
]
STRAIGHT_BRANCHING = [1 for i in range(0, 20)]
MAX_DEPTH = 20

# -----------------------
# Time-control checkpoints (per-move caps depending on remaining clock)
# -----------------------
def _per_move_time_cap(time_left: float) -> float:
    if time_left >= 300.0:
        return 2.0
    if time_left >= 60.0:
        return 1.5
    if time_left >= 30.0:
        return 1.0
    if time_left >= 10.0:
        return 0.5
    if time_left >= 2.0:
        return 0.2
    return 0.05

# +1 means white, -1 means black; me is just my evaluation (already calculated)
def dfs(board: chess.Board, me, depth, color):
    if depth == MAX_DEPTH:
        return me
    
    srt = []
    cnt = 0
    for mv in board.legal_moves:
        board.push(mv)
        gb = evaluate(board)
        srt.append((mv, gb*color))
        board.pop()

        if gb*color > me*color: # for every position that returns something better than me already
            cnt += 1
        if cnt >= BRANCHING_DFS[depth-1]: # this means you have enough good positions
            break

    srt.sort(key=lambda x: x[1], reverse=True)
    rem = srt[:BRANCHING_DFS[depth-1]]
    print(rem, color)

    best_ret = -1e9
    for mv, next_me in rem:
        board.push(mv)
        gb = dfs(board, next_me, depth+1, -color)
        board.pop()
        best_ret = max(best_ret, color * gb)
    
    return best_ret * color

def bfs(rem, depth, color):
    if depth == MAX_DEPTH:
        best_ret = -1e9
        for _, _, bd in rem[-1]:
            for mv in bd.legal_moves:
                bd.push(mv)
                gb = evaluate(bd)
                bd.pop()
                best_ret = max(best_ret, color * gb)
        for l in rem:
            for item in l:
                print(item[0], end=" ")
            print()
        return best_ret * color
    
    srt = []
    for _, _, bd in rem[-1]:
        for mv in bd.legal_moves:
            bd.push(mv)
            gb = evaluate(bd)
            srt.append((mv, gb*color, bd.copy()))
            bd.pop()
    srt.sort(key=lambda x: x[1], reverse=True)
    rem.append(srt[:BRANCHING_BFS[depth-1]])
    return bfs(rem, depth+1, -color)

# color is +1 for white, -1 for black
def next_move(board: chess.Board, time_left: float, color) -> Optional[chess.Move]:
    per_move_cap = _per_move_time_cap(time_left)
    time_budget = min(2.0, per_move_cap)
    start_time = time.perf_counter()
    node_limit = 20000
    nodes = {"count": 0}

    moves = list(board.legal_moves)

    best_move, best_ret = "", -1e9
    for mv in moves:
        board.push(mv)
        # gb = dfs(board, evaluate(board), 18, -1*color)
        srt = []
        for mv2 in board.legal_moves:
            board.push(mv2)
            gb = evaluate(board)
            srt.append((mv, gb*color, board.copy()))
            board.pop()
        srt.sort(key=lambda x: x[1], reverse=True)
        gb = bfs([srt[:BRANCHING_BFS[0]]], 2, -1*color)
        if gb*color > best_ret:
            best_ret = gb*color
            best_move = mv
        board.pop()
    
    return best_move


def evaluate(board: chess.Board) -> float:
    # Piece values in centipawns
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # Usually king is not counted in material eval
    }

    score = 0
    for piece_type in values:
        # Count number of pieces of this type for white and black
        white_pieces = len(board.pieces(piece_type, chess.WHITE))
        black_pieces = len(board.pieces(piece_type, chess.BLACK))
        score += values[piece_type] * (white_pieces - black_pieces)

    # print(f"evaluate (material) is {score}")
    return float(score)



if __name__ == "__main__":
    b = chess.Board("2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - -")
    mv = next_move(b, time_left=120.0, color=1)
    print("Selected move:", mv)
