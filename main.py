# play.py
import chess
import torch
from mcts import MCTS
from model import ChessNet

DEVICE = "cuda"

def choose_move(board, checkpoint="data/checkpoints/iter_150.pt"):
    net = ChessNet().to(DEVICE)
    net.load_state_dict(torch.load(checkpoint))
    net.eval()

    mcts = MCTS(net, simulations=400, device=DEVICE)
    best_move = mcts.run(board)
    return best_move

# Example usage:
if __name__ == "__main__":
    board = chess.Board()
    move = choose_move(board)
    print(move)
