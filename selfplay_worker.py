# selfplay_worker.py
# 단일 프로세스 self-play worker: MCTS 사용, examples 파일로 저장
import argparse
import os
import pickle
import torch
from tqdm import trange
import numpy as np
from alphazero_model import AlphaZeroNet
from mcts import MCTS, is_five
from selfplay import augment_example  # reuse augment_example

def play_game(net, board_size=19, n_sim=200, temp_threshold=30, device='cuda'):
    net.eval()
    mcts = MCTS(net, board_size=board_size, c_puct=1.0, n_sim=n_sim, device=device, max_candidates=400)
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
            return examples
        to_move = -to_move
    examples = []
    for (s, pi, player) in history:
        for (as_p, api) in augment_example(s, pi):
            examples.append((as_p.astype(np.float32), api.astype(np.float32), 0.0))
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="examples")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--n_sim", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    net = AlphaZeroNet(board_size=19, num_filters=128)
    if args.model:
        net.load_state_dict(torch.load(args.model, map_location=args.device))
    net.to(args.device)

    for i in trange(args.games):
        ex = play_game(net, board_size=19, n_sim=args.n_sim, device=args.device)
        outp = os.path.join(args.out_dir, f"examples_worker_{os.getpid()}_{i}.pkl")
        with open(outp, "wb") as f:
            pickle.dump(ex, f)

if __name__ == "__main__":
    main()