import subprocess
import sys
import re

STOCKFISH_PATH = "./stockfish/stockfish-windows-x86-64-avx2.exe"   # adjust this

def evaluate_fen(fen: str, depth: int = 12) -> int:
    engine = subprocess.Popen(
        [STOCKFISH_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # Init UCI
    engine.stdin.write("uci\n")
    engine.stdin.flush()

    for line in engine.stdout:
        if "uciok" in line:
            break

    # Send position + go
    engine.stdin.write(f"position fen {fen}\n")
    engine.stdin.write(f"go depth {depth}\n")
    engine.stdin.flush()

    eval_cp = None
    eval_mate = None

    for line in engine.stdout:
        if line.startswith("bestmove"):
            break

        if "score cp" in line:
            m = re.search(r"score cp (-?\d+)", line)
            if m:
                eval_cp = int(m.group(1))

        if "score mate" in line:
            m = re.search(r"score mate (-?\d+)", line)
            if m:
                eval_mate = int(m.group(1))

    # ---- SAFE QUIT ----
    try:
        if engine.poll() is None:  # engine is still running
            engine.stdin.write("quit\n")
            engine.stdin.flush()
    except OSError:
        pass   # stdin is already closed

    # Always wait() to avoid zombies
    try:
        engine.wait(timeout=1)
    except:
        pass

    # ---- RETURN SCORE ----
    if eval_mate is not None:
        return 99999 if eval_mate > 0 else -99999

    return eval_cp if eval_cp is not None else 0



if __name__ == "__main__":
    fen = "2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - -"
    start = "rnbqkbnr/pppppppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
    mi3 = "6k1/5ppp/8/8/8/3Q4/5PPP/6K1 w - - 0 1"
    draw = "8/7p/8/8/8/6B1/7P/7K w - - 0 1"

    print(evaluate_fen(fen, 2))
    print(evaluate_fen(start, 2))
    print(evaluate_fen(mi3, 2))
    print(evaluate_fen(draw, 2))
