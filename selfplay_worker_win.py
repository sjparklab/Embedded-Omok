# selfplay_worker_win.py
# Windows-friendly self-play worker.
# Each worker creates its own inference server (via MCTS use_infer_server=True)
import argparse
import os
import pickle
from tqdm import trange
import numpy as np
import torch
from alphazero_model import AlphaZeroNet
from mcts import MCTS, is_five

def augment_example(state_planes, pi):
    H = state_planes.shape[1]
    examples = []
    p_board = pi.reshape((H,H))
    for k in range(4):
        sp = np.rot90(state_planes, k, axes=(1,2))
        pp = np.rot90(p_board, k)
        examples.append((sp.copy(), pp.copy().flatten()))
        spf = np.flip(sp, axis=2)
        ppf = np.flip(pp, axis=1)
        examples.append((spf.copy(), ppf.copy().flatten()))
    return examples

def play_game(net, board_size=19, n_sim=200, temp_threshold=30, device='cuda:0'):
    net.eval()
    # NOTE: enable use_infer_server -> MCTS will spawn a local inference server (child process)
    mcts = MCTS(net, board_size=board_size, c_puct=1.0, n_sim=n_sim, device=device,
                max_candidates=200, use_infer_server=True, infer_batch_size=64, infer_device=device)
    board = np.zeros((board_size, board_size), dtype=int)
    to_move = 1
    history = []
    for move in range(board_size * board_size):
        temp = 1.0 if move < temp_threshold else 1e-3
        probs = mcts.get_action_probs(board.copy(), to_move, temp=temp)
        my = (board == to_move).astype(np.float32)
        opp = (board == -to_move).astype(np.float32)
        state_planes = np.stack([my, opp], axis=0)
        history.append((state_planes, probs.copy(), to_move))
        if temp > 0.1:
            a = np.random.choice(len(probs), p=probs)
        else:
            a = int(np.argmax(probs))
        x = a // board_size; y = a % board_size
        board[x,y] = to_move
        if is_five(board, x, y, to_move, board_size):
            winner = to_move
            examples = []
            for (s, pi, player) in history:
                z = 1.0 if player == winner else -1.0
                for (as_p, api) in augment_example(s, pi):
                    examples.append((as_p.astype(np.float32), api.astype(np.float32), float(z)))
            # clean up MCTS (stop infer server)
            try:
                mcts.close()
            except Exception:
                pass
            return examples
        to_move = -to_move
    # draw
    examples = []
    for (s, pi, player) in history:
        for (as_p, api) in augment_example(s, pi):
            examples.append((as_p.astype(np.float32), api.astype(np.float32), 0.0))
    try:
        mcts.close()
    except Exception:
        pass
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="examples")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--n_sim", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    net = AlphaZeroNet(board_size=19, num_filters=128)
    if args.model and os.path.exists(args.model):
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
    net.to(args.device)

    for i in trange(args.games, desc="Self-play games"):
        ex = play_game(net, board_size=19, n_sim=args.n_sim, device=args.device)
        outp = os.path.join(args.out_dir, f"examples_worker_{os.getpid()}_{i}.pkl")
        with open(outp, "wb") as f:
            pickle.dump(ex, f)

if __name__ == "__main__":
    main()