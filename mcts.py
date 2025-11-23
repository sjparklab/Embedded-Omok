# PUCT 기반 MCTS (네트워크의 prior와 value 사용)
import math
import numpy as np
from collections import defaultdict
import torch

def is_five(board, x, y, player, board_size):
    """19x19 보드에서 5목 확인 (금수 규칙 없음)."""
    dirs = [(1,0),(0,1),(1,1),(1,-1)]
    for dx, dy in dirs:
        cnt = 1
        nx, ny = x+dx, y+dy
        while 0 <= nx < board_size and 0 <= ny < board_size and board[nx,ny] == player:
            cnt += 1
            nx += dx; ny += dy
        nx, ny = x-dx, y-dy
        while 0 <= nx < board_size and 0 <= ny < board_size and board[nx,ny] == player:
            cnt += 1
            nx -= dx; ny -= dy
        if cnt >= 5:
            return True
    return False

class MCTS:
    def __init__(self, net, board_size=19, c_puct=1.0, n_sim=200, device='cpu'):
        self.net = net
        self.board_size = board_size
        self.c_puct = c_puct
        self.n_sim = n_sim
        self.device = device

        # tree statistics
        self.P = {}   # state_key -> prior probs (flat length board_size*board_size)
        self.N = defaultdict(int)   # (state_key, action) -> visit count
        self.W = defaultdict(float) # (state_key, action) -> total value
        self.Q = defaultdict(float) # (state_key, action) -> mean value
        self.children = {}  # state_key -> list of legal actions

    def state_key(self, board, player):
        # key: (bytes(flattened board), player)
        return (board.tobytes(), int(player))

    def canonical_board(self, board, player):
        # planes: plane0 = 플레이어, plane1 = 상대방
        my = (board == player).astype(np.float32)
        opp = (board == -player).astype(np.float32)
        planes = np.stack([my, opp], axis=0)
        return planes

    def add_dirichlet_noise(self, probs, eps=0.25, alpha=0.3):
        """Dirichlet noise for root prior exploration."""
        if len(probs) == 0:
            return probs
        noise = np.random.dirichlet([alpha] * len(probs))
        return probs * (1 - eps) + noise * eps

    def get_action_probs(self, board, player, temp=1.0, add_root_noise=True):
        """
        Run n_sim simulations and return action probability vector (size board_size*board_size).
        temp: temperature for selecting distribution (>=0). If temp==0, return deterministic best.
        """
        root_key = self.state_key(board, player)
        # run sims
        for i in range(self.n_sim):
            self.simulate(board.copy(), player)
        counts = np.array([self.N[(root_key, a)] for a in range(self.board_size*self.board_size)], dtype=np.float64)
        if temp <= 0:
            best = int(np.argmax(counts))
            probs = np.zeros_like(counts)
            probs[best] = 1.0
            return probs
        # temperature softmax on counts
        #counts = counts ** (1.0 / (temp + 1e-9))
        #s = counts.sum()
        #if s == 0:
            
        # temperature-based selection, numerically stable version
        # 거듭제곱으로 인한 오버플로우를 방지하기 위해 log-space에서 계산
        log_counts = np.log(counts + 1e-9) # counts가 0일 수 있으므로 작은 값 더하기
        logits = log_counts / (temp + 1e-9)
        
        # 안정적인 softmax 계산
        logits = logits - np.max(logits) # 가장 큰 값을 빼서 오버플로우 방지
        exp_logits = np.exp(logits)
        
        probs = exp_logits / (exp_logits.sum() + 1e-9)
        if probs.sum() == 0:
            # fallback: uniform on legal moves
            legal = (board.flatten() == 0).astype(np.float64)
            #legal_sum = legal.sum()
            #if legal_sum == 0:
                #return np.zeros_like(counts)
            #return legal / legal_sum
        #probs = counts / s
            return legal / (legal.sum() + 1e-9)
        return probs

    def simulate(self, board, player):
        """Single MCTS simulation from (board, player)."""
        path = []
        key = self.state_key(board, player)
        last_move = None

        # selection
        while key in self.P:
            legal = self.children.get(key, [])
            if not legal:
                break
            best_a = None
            best_ucb = -1e9
            total_N = sum(self.N[(key, a)] for a in legal)
            for a in legal:
                n = self.N[(key, a)]
                q = self.Q.get((key, a), 0.0)
                p = self.P[key][a]
                u = q + self.c_puct * p * math.sqrt(total_N + 1) / (1 + n)
                if u > best_ucb:
                    best_ucb = u
                    best_a = a
            path.append((key, best_a))
            # play
            x = best_a // self.board_size
            y = best_a % self.board_size
            board[x, y] = player
            last_move = (x, y, player)
            # terminal? check 5-in-a-row
            if is_five(board, x, y, player, self.board_size):
                # value from perspective of player who just moved is +1
                value = 1.0
                # backup (invert value each parent step)
                for (s_key, a) in reversed(path):
                    self.N[(s_key, a)] += 1
                    self.W[(s_key, a)] += value
                    self.Q[(s_key, a)] = self.W[(s_key, a)] / self.N[(s_key, a)]
                    value = -value
                return
            player = -player
            key = self.state_key(board, player)

        # expansion
        planes = self.canonical_board(board, player)
        tensor = torch.from_numpy(planes).unsqueeze(0).to(self.device)  # (1,2,H,W)
        with torch.no_grad():
            logits, value = self.net(tensor)
            logits = logits.squeeze(0).cpu().numpy()
            value = float(value.item())

        # mask illegal moves
        mask = (board.flatten() == 0).astype(np.float32)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits * mask
        if probs.sum() <= 0:
            probs = mask
        probs = probs / (probs.sum() + 1e-9)

        node_key = self.state_key(board, player)
        self.P[node_key] = probs
        legal_actions = np.where(mask > 0)[0].tolist()
        self.children[node_key] = legal_actions

        # backup
        for (s_key, a) in reversed(path):
            self.N[(s_key, a)] += 1
            self.W[(s_key, a)] += value
            self.Q[(s_key, a)] = self.W[(s_key, a)] / self.N[(s_key, a)]
            value = -value
        return