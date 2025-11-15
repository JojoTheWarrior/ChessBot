# mcts_with_evaluate.py
# Requires: pip install chess
import chess
import math
import time
import random
from typing import Callable, Optional

# -------------------------
# Helper: parse Stockfish-like evaluate() outputs
# -------------------------
def parse_stockfish_eval(raw):
    """
    Accepts evaluate(board) output and returns a float evaluation (white-perspective).
    Supported input types/formats:
      - int/float: treated as centipawns (e.g., 34 => 34 cp)
      - strings like 'M3', '#3', 'mate 3', '+M3', '-M2'
      - python-chess/Popen-like dicts are NOT assumed; adapt if needed.
    Returns a tuple (is_mate, cp_or_mate_signed)
      - if is_mate True: second value is mate distance (positive for White-winning mate, negative for losing mate)
      - if is_mate False: second value is centipawns (float), White-perspective
    """
    if raw is None:
        return False, 0.0
    # numeric: centipawns
    if isinstance(raw, (int, float)):
        return False, float(raw)
    s = str(raw).strip().lower()
    # look for mate patterns
    if s.startswith('m') or s.startswith('#') or s.startswith('mate') or 'mate' in s:
        # extract integer after letters
        import re
        m = re.search(r'(-?\d+)', s)
        if m:
            n = int(m.group(1))
            return True, int(n)
        else:
            # unknown mate format -> treat as big cp
            return False, 100000.0
    # try to parse as float fallback
    try:
        return False, float(s)
    except Exception:
        # unknown format: return 0
        return False, 0.0

# -------------------------
# MCTS node and main class
# -------------------------
class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, move: Optional[chess.Move]=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = {}  # move -> node
        self.N = 0  # visits
        self.W = 0.0  # total value (sum of values from White-perspective)
        self.P = 0.0  # prior probability assigned by parent
        self.is_expanded = False
        # store evaluator-derived node value for convenience (white-perspective)
        self.node_value = 0.0

    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0

class GameClock:
    def __init__(self, total_seconds: float = 60.0):
        self.total = total_seconds
        self.used = 0.0

    def remaining(self):
        r = max(0.0, self.total - self.used)
        return r

    def spend(self, seconds: float):
        self.used += seconds
        if self.used > self.total:
            self.used = self.total

class MCTS:
    def __init__(self,
                 evaluate_fn: Callable[[chess.Board], object],
                 total_game_time: float = 60.0,
                 c_puct: float = 1.5,
                 eval_scale_cp: float = 400.0,
                 use_transposition: bool = True,
                 min_alloc_per_move: float = 0.2):
        """
        evaluate_fn(board) -> Stockfish-like output (centipawns and/or mate).
        total_game_time: seconds available for the entire game.
        c_puct: exploration constant.
        eval_scale_cp: scale (centipawns) mapped to [-1,1] with tanh; adjust to your evaluator.
        use_transposition: whether to reuse nodes by transposition key.
        min_alloc_per_move: minimum seconds to allocate to a single move (to avoid starving).
        """
        self.evaluate_fn = evaluate_fn
        self.c_puct = c_puct
        self.eval_scale_cp = eval_scale_cp
        self.use_transposition = use_transposition
        self.tt = {}  # transposition table: key -> MCTSNode
        self.clock = GameClock(total_game_time)
        self.min_alloc = min_alloc_per_move

    def _eval_to_value(self, raw_eval):
        """
        Convert evaluate() output to a bounded value in (-1,1),
        using centipawn scaling and special handling for mate scores.
        """
        is_mate, val = parse_stockfish_eval(raw_eval)
        if is_mate:
            mate_dist = int(val)  # positive => White can mate in mate_dist; negative => White will be mated
            # map to near +/-1, closer for shorter mates
            if mate_dist == 0:
                return 0.0
            sign = 1.0 if mate_dist > 0 else -1.0
            dist = abs(mate_dist)
            # value in (0.8, 1.0) for mate distances, sharper for small dist
            v = 1.0 - 1.0 / (dist + 1.0)
            return sign * v
        else:
            # val is centipawns (white-perspective)
            cp = float(val)
            # scale by eval_scale_cp and tanh to put in (-1,1)
            return math.tanh(cp / self.eval_scale_cp)

    def _get_key(self, board: chess.Board):
        return board.transposition_key() if self.use_transposition else None

    def _get_or_create_node(self, board: chess.Board, parent=None, move=None):
        key = self._get_key(board)
        if key is not None and key in self.tt:
            node = self.tt[key]
            # update parent/move if missing (we reuse stats across transpositions)
            if node.parent is None and parent is not None:
                node.parent = parent
                node.move = move
            return node
        node = MCTSNode(board, parent=parent, move=move)
        if key is not None:
            self.tt[key] = node
        return node

    def _expand_and_set_priors(self, node: MCTSNode):
        if node.is_expanded:
            return
        legal = list(node.board.legal_moves)
        # terminal?
        if not legal:
            node.is_expanded = True
            # set terminal node value using evaluate or game result
            if node.board.is_game_over():
                res = node.board.result()
                if res == '1-0':
                    node.node_value = 1.0
                elif res == '0-1':
                    node.node_value = -1.0
                else:
                    node.node_value = 0.0
            else:
                node.node_value = 0.0
            return

        # Move ordering: captures and promotions first
        def move_score(mv):
            s = 0
            if node.board.is_capture(mv):
                s += 10
            if mv.promotion is not None:
                s += 8
            # prefer checks (temporary)
            node.board.push(mv)
            if node.board.is_check():
                s += 3
            node.board.pop()
            return -s

        legal.sort(key=move_score)

        # compute raw evals for each child to produce priors
        raw_evals = []
        for mv in legal:
            node.board.push(mv)
            raw = self.evaluate_fn(node.board)  # white-perspective
            raw_evals.append(raw)
            node.board.pop()

        # softmax of child evals -> priors
        # convert raw evals (centipawns + mate) to cp-like numbers for softmax scale:
        # for mate, push a large number proportional to mate closeness
        cp_values = []
        for r in raw_evals:
            is_m, v = parse_stockfish_eval(r)
            if is_m:
                # map mate distance to large cp: closer mates become larger cp magnitude
                if v == 0:
                    cp_values.append(0.0)
                else:
                    sign = 1.0 if v > 0 else -1.0
                    cp_values.append(sign * (20000.0 / (abs(v))))  # big magnitude for mate
            else:
                cp_values.append(float(v))

        # numeric stability softmax
        max_cp = max(cp_values) if cp_values else 0.0
        exps = [math.exp((x - max_cp) / max(1.0, self.eval_scale_cp / 4.0)) for x in cp_values]
        s = sum(exps) if sum(exps) != 0 else 1.0
        priors = [e / s for e in exps]

        for mv, p, raw in zip(legal, priors, raw_evals):
            node.board.push(mv)
            child = self._get_or_create_node(node.board, parent=node, move=mv)
            node.board.pop()
            child.P = p
            node.children[mv] = child

        # evaluate node itself (for value at expansion)
        node.node_value = self._eval_to_value(self.evaluate_fn(node.board))
        node.is_expanded = True

    def _select(self, root: MCTSNode):
        """
        Select until a non-expanded node (leaf) is found.
        Selection uses PUCT:
           U = c_puct * P * sqrt(sum_N_parent) / (1 + N_child)
           score = Q + U
        Returns leaf node and path (list of nodes from root to leaf).
        """
        node = root
        path = [node]
        while True:
            if not node.is_expanded:
                return node, path
            if not node.children:
                return node, path
            parent_N = max(1, node.N)
            best = None
            best_score = -1e9
            for mv, child in node.children.items():
                Q = child.Q
                U = self.c_puct * child.P * math.sqrt(parent_N) / (1 + child.N)
                score = Q + U
                # We keep Q as white-perspective; score comparison consistent
                if score > best_score:
                    best_score = score
                    best = child
            node = best
            path.append(node)

    def _backup(self, path, leaf_value_white: float):
        """
        Backpropagate leaf_value_white (white-perspective) up the path.
        We flip sign at each parent to account for side-to-move alternation.
        """
        v = leaf_value_white
        for node in reversed(path):
            node.N += 1
            node.W += v
            v = -v  # invert for parent

    def _allocate_time_for_move(self, board: chess.Board):
        """
        Simple adaptive allocation:
        - estimate moves left: roughly 40 - fullmove_number (minimum 10)
        - allocate remaining_time / estimated_moves_left, but not less than min_alloc
        """
        rem = self.clock.remaining()
        # estimate moves left conservatively
        est_moves_left = max(8, 40 - board.fullmove_number)  # tuneable
        alloc = rem / max(1.0, est_moves_left)
        # ensure at least min_alloc, and at most remaining
        alloc = max(self.min_alloc, alloc)
        alloc = min(alloc, rem)
        return alloc

    def search_move(self, board: chess.Board, verbose: bool=False) -> Optional[chess.Move]:
        """
        Run MCTS using remaining game time allocation for this move.
        Returns the chosen chess.Move.
        """
        # Create/clear transposition table? we keep stats across moves to reuse knowledge.
        root = self._get_or_create_node(board, parent=None, move=None)
        # Expand root
        self._expand_and_set_priors(root)

        time_for_move = self._allocate_time_for_move(board)
        if verbose:
            print(f"[MCTS] Allocating {time_for_move:.3f}s for this move. Remaining before search: {self.clock.remaining():.3f}s")
        start = time.time()
        iterations = 0
        # Iterative search until time_for_move used up or no time left
        while time.time() - start < time_for_move and self.clock.remaining() > 0.0001:
            iterations += 1
            leaf, path = self._select(root)
            # terminal?
            if leaf.board.is_game_over():
                # terminal value
                res = leaf.board.result()
                if res == '1-0':
                    v = 1.0
                elif res == '0-1':
                    v = -1.0
                else:
                    v = 0.0
                self._backup(path, v)
                continue
            # expand
            self._expand_and_set_priors(leaf)
            # evaluate leaf via evaluate_fn and convert
            raw = self.evaluate_fn(leaf.board)
            val = self._eval_to_value(raw)
            # backup
            self._backup(path, val)

        elapsed = time.time() - start
        self.clock.spend(elapsed)
        if verbose:
            print(f"[MCTS] Done. Iterations: {iterations}, elapsed: {elapsed:.3f}s, remaining_time: {self.clock.remaining():.3f}s")
        # pick best child by visit count
        if not root.children:
            return None
        best_move, best_node = max(root.children.items(), key=lambda kv: kv[1].N)
        if verbose:
            # show top-3 moves
            sorted_children = sorted(root.children.items(), key=lambda kv: kv[1].N, reverse=True)
            print("[MCTS] Top moves (move, visits, Q, P):")
            for mv, nd in sorted_children[:5]:
                print(f"  {mv} visits={nd.N} Q={nd.Q:.3f} P={nd.P:.3f}")
        return best_move

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Placeholder evaluate() to demonstrate format:
    # Replace with your actual evaluate(board) function that returns Stockfish-like eval
    def evaluate(board: chess.Board):
        # Very simple: material in centipawns (white-perspective)
        vals = {chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330, chess.ROOK:500, chess.QUEEN:900, chess.KING:20000}
        s = 0
        for t,v in vals.items():
            s += v * (len(board.pieces(t, chess.WHITE)) - len(board.pieces(t, chess.BLACK)))
        return float(s)  # centipawns

    # Example play: MCTS plays White until game over; human plays Black via UCI/SAN
    board = chess.Board()
    mcts = MCTS(evaluate_fn=evaluate, total_game_time=60.0, c_puct=1.5, eval_scale_cp=400.0)
    while not board.is_game_over():
        print(board)
        if board.turn == chess.WHITE:
            mv = mcts.search_move(board, verbose=True)
            if mv is None:
                print("No legal move found.")
                break
            print("MCTS plays:", mv)
            board.push(mv)
        else:
            user = input("Your move (UCI or SAN): ")
            try:
                mv = board.parse_san(user)
            except Exception:
                try:
                    mv = chess.Move.from_uci(user)
                except Exception:
                    print("Invalid format.")
                    continue
            if mv not in board.legal_moves:
                print("Illegal move.")
                continue
            board.push(mv)
    print("Result:", board.result())
