# self_play.py
import chess
import torch
from mcts import MCTS
from chess_game import board_to_tensor, move_to_index

def self_play_game(net, simulations=200, device="cuda"):
    mcts = MCTS(net, simulations=simulations, device=device)

    board = chess.Board()
    examples = []  # (state, policy, value)

    while not board.is_game_over():
        # Run MCTS
        move = mcts.run(board)

        # Collect training example
        state = board_to_tensor(board)
        policy = torch.zeros(4672)
        for m, n in mcts.root.N.items():
            idx = move_to_index(m)
            policy[idx] = n
        policy /= policy.sum()

        examples.append((state, policy))

        board.push(move)

    # Final result
    outcome = board.outcome().winner
    value = 1 if outcome == chess.WHITE else -1 if outcome == chess.BLACK else 0

    return [(s, p, value) for (s, p) in examples]
