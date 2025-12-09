#!/usr/bin/env python3
# generate_ab_games.py
# Generate expert games using the repository's iot12345_student (minimax/alpha-beta) engine.
# Output: pickle file containing list of (state_planes, pi_onehot) tuples for supervised pretraining.

import argparse
import pickle
import numpy as np
import os
from tqdm import trange

# import engine and stone class from repo
from iot12345_student import iot12345_student
from stone import stone

def is_five_at(board, x, y, player, board_size):
    dirs = [(1,0),(0,1),(1,1),(1,-1)]
    for dx, dy in dirs:
        cnt = 1
        nx, ny = x+dx, y+dy
        while 0 <= nx < board_size and 0 <= ny < board_size and board[nx,ny] == player:
            cnt += 1; nx += dx; ny += dy
        nx, ny = x-dx, y-dy
        while 0 <= nx < board_size and 0 <= ny < board_size and board[nx,ny] == player:
            cnt += 1; nx -= dx; ny -= dy
        if cnt >= 5:
            return True
    return False

def canonical_planes(board, to_move):
    # board: np.array (H,W) with values {0,1,-1}
    my = (board == to_move).astype(np.float32)
    opp = (board == -to_move).astype(np.float32)
    return np.stack([my, opp], axis=0)  # shape (2,H,W)

def safe_get_move(player, board_list, size):
    # Call player's next(); if it returns invalid move, pick a random empty
    try:
        stn = player.next(board_list, size)
        if stn is None:
            raise Exception("player.next returned None")
        x = int(stn.getX()); y = int(stn.getY())
        return x, y
    except Exception as e:
        # fallback: pick random empty
        empties = [(i,j) for i in range(size) for j in range(size) if board_list[i][j]==0]
        if not empties:
            return None
        x,y = empties[np.random.randint(len(empties))]
        return x,y

def generate_games(num_games=1000, board_size=19, out="examples_ab.pkl", max_moves=None, verbose=True):
    # instantiate two expert players (white=1, black=-1)
    p_white = iot12345_student(1)
    p_black = iot12345_student(-1)

    all_examples = []
    max_moves = max_moves or (board_size * board_size)
    for g in trange(num_games, desc="Generating games"):
        board = np.zeros((board_size, board_size), dtype=int)
        to_move = 1  # white starts per repo convention
        history = []
        for mv in range(max_moves):
            board_list = board.tolist()
            # record state before move
            planes = canonical_planes(board, to_move)
            # get move from the appropriate expert
            player = p_white if to_move == 1 else p_black
            x_y = safe_get_move(player, board_list, board_size)
            if x_y is None:
                break
            x, y = x_y
            # validate; if invalid, choose random empty
            if x < 0 or x >= board_size or y < 0 or y >= board_size or board[x,y] != 0:
                empties = [(i,j) for i in range(board_size) for j in range(board_size) if board[i,j]==0]
                if not empties:
                    break
                x,y = empties[np.random.randint(len(empties))]
            # create one-hot policy vector
            pi = np.zeros(board_size * board_size, dtype=np.float32)
            idx = x * board_size + y
            pi[idx] = 1.0
            # append (canonical state, pi)
            all_examples.append((planes.astype(np.float32), pi, float(0.0)))  # z placeholder 0.0 for supervised pretrain
            # play move
            board[x,y] = to_move
            # terminal?
            if is_five_at(board, x, y, to_move, board_size):
                break
            to_move = -to_move
        # optional: shuffle per-game or periodically flush to disk to save memory
        if verbose and (g+1) % 1000 == 0:
            print(f"Generated {g+1} games, samples so far: {len(all_examples)}")

    # save
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(all_examples, f)
    print(f"Saved {len(all_examples)} samples to {out}")
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--board_size", type=int, default=19)
    parser.add_argument("--out", type=str, default="examples_ab.pkl")
    parser.add_argument("--max_moves", type=int, default=None)
    args = parser.parse_args()
    generate_games(num_games=args.games, board_size=args.board_size, out=args.out, max_moves=args.max_moves)