# mcts.py
import math
import random
import chess
import torch
from chess_game import board_to_tensor, move_to_index, index_to_move

class Node:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = {}
        self.N = {}   # visit counts
        self.W = {}   # total value
        self.P = {}   # policy priors

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    def __init__(self, net, simulations=200, c_puct=1.2, device="cuda"):
        self.net = net
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device

    def run(self, root_board):
        root = Node(root_board)

        # Evaluate root with the network
        self._evaluate(root)

        for _ in range(self.simulations):
            node = root
            board = root_board.copy()

            # 1. Selection
            while not node.is_leaf() and not board.is_game_over():
                move = self._select(node)
                board.push(move)

                if move not in node.children:
                    node.children[move] = Node(board, parent=node)

                node = node.children[move]

            # 2. Expansion & evaluation
            if not board.is_game_over():
                self._evaluate(node)

            # 3. Backpropagation
            v = self._value_of(board)
            self._backpropagate(node, v)

        # Return move with highest visit count
        best_move = max(root.N, key=lambda m: root.N[m])
        return best_move

    # -------------------------

    def _evaluate(self, node):
        board = node.board
        x = board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.net(x)
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

        for move in board.legal_moves:
            idx = move_to_index(move)
            node.P[move] = policy[idx]
            node.N[move] = 0
            node.W[move] = 0

    def _select(self, node):
        # PUCB formula
        total_visits = sum(node.N[m] for m in node.N)
        best_score = -1
        best_move = None

        for move in node.N:
            if node.N[move] == 0:
                u = float("inf")
            else:
                q = node.W[move] / node.N[move]
                u = q + self.c_puct * node.P[move] * math.sqrt(total_visits) / (1 + node.N[move])

            if u > best_score:
                best_score = u
                best_move = move

        return best_move

    def _value_of(self, board):
        if board.is_game_over():
            result = board.outcome().winner
            return 1 if result == board.turn else -1
        return 0

    def _backpropagate(self, node, v):
        while node is not None:
            parent = node.parent
            if parent is not None:
                move = None
                for m, child in parent.children.items():
                    if child is node:
                        move = m
                        break
                parent.N[move] += 1
                parent.W[move] += v
            node = parent
