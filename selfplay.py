# 자기대국으로 state, pi, z 수집
import numpy as np
import pickle
from tqdm import trange
import torch
from alphazero_model import AlphaZeroNet
from mcts import MCTS, is_five

def play_game(net, board_size=19, n_sim=200, temp_threshold=30, device='cpu'):
    """
    Play one self-play game using MCTS guided by net.
    Returns list of training examples: (state_planes, pi, z)
    state_planes: (2, H, W) np.float32 (canonicalized w.r.t. player to move)
    pi: flat prob vector length H*W (np.float32)
    z: final outcome from current player's perspective (float: 1/-1/0)
    """
    net.eval()
    mcts = MCTS(net, board_size=board_size, c_puct=1.0, n_sim=n_sim, device=device)
    board = np.zeros((board_size, board_size), dtype=int)
    to_move = 1  # 선(先)수 결정
    history = []

    for move in range(board_size * board_size):
        # temperature schedule: high temp for first temp_threshold moves
        temp = 1.0 if move < temp_threshold else 1e-3
        # run sims and get probs
        probs = mcts.get_action_probs(board.copy(), to_move, temp=temp)
        # store training example (canonicalized)
        my = (board == to_move).astype(np.float32)
        opp = (board == -to_move).astype(np.float32)
        state_planes = np.stack([my, opp], axis=0)
        history.append((state_planes, probs.copy(), to_move))
        # sample action according to probs (stochastic for exploration)
        if temp > 0.1:
            a = np.random.choice(len(probs), p=probs)
        else:
            a = int(np.argmax(probs))
        x = a // board_size; y = a % board_size
        board[x,y] = to_move
        # terminal?
        if is_five(board, x, y, to_move, board_size):
            winner = to_move
            # assign z labels
            examples = []
            for (s, pi, player) in history:
                z = 1.0 if player == winner else -1.0
                examples.append((s.astype(np.float32), pi.astype(np.float32), float(z)))
            return examples
        # next
        to_move = -to_move

    # if full board and no winner -> draw (z=0)
    examples = []
    for (s, pi, player) in history:
        examples.append((s.astype(np.float32), pi.astype(np.float32), 0.0))
    return examples

def generate_selfplay_data(net, num_games=20, out_path="examples.pkl", board_size=19, n_sim=200, device='cpu'):
    all_examples = []
    for i in trange(num_games, desc="Self-play games"):
        ex = play_game(net, board_size=board_size, n_sim=n_sim, device=device)
        all_examples.extend(ex)
    with open(out_path, "wb") as f:
        pickle.dump(all_examples, f)
    print(f"Saved {len(all_examples)} examples to {out_path}")
    return out_path

if __name__ == "__main__":
    # 간단 테스트 → 랜덤 초기화된 네트워크로 소수 게임 생성
    board_size = 19
    net = AlphaZeroNet(board_size=board_size)
    generate_selfplay_data(net, num_games=10, out_path="examples.pkl", board_size=board_size, n_sim=100)