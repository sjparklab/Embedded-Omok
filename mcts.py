# mcts.py
# PUCT 기반 MCTS (inference server 통합, Manager-safe for Windows)
import math
import numpy as np
from collections import defaultdict
import torch
import tempfile
import os
import multiprocessing as mp
import time
import random
from alphazero_model import AlphaZeroNet
from inference_server import InferenceServer  # updated server

def is_five(board, x, y, player, board_size):
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

def get_bounding_box(board, board_size, margin=2):
    min_x, max_x = board_size, -1
    min_y, max_y = board_size, -1
    for x in range(board_size):
        for y in range(board_size):
            if board[x][y] != 0:
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
    if max_x == -1:
        return None
    sx = max(0, min_x - margin)
    ex = min(board_size - 1, max_x + margin)
    sy = max(0, min_y - margin)
    ey = min(board_size - 1, max_y + margin)
    return sx, ex, sy, ey

class MCTS:
    def __init__(self, net, board_size=19, c_puct=1.0, n_sim=200, device='cpu',
                 max_candidates=None, use_infer_server=False, infer_batch_size=64, infer_device='cuda:0', infer_num_filters=128):
        self.net = net
        self.board_size = board_size
        self.c_puct = c_puct
        self.n_sim = n_sim
        self.device = device
        self.max_candidates = max_candidates

        # inference server support using a Manager-based queue/dict (safe on Windows)
        self.infer_server = None
        self.manager = None
        self.req_q = None
        self.res_dict = None
        if use_infer_server:
            try:
                # create a manager and shared req queue + res dict
                self.manager = mp.Manager()
                self.req_q = self.manager.Queue()
                self.res_dict = self.manager.dict()
                # dump network state to temp file for server to load
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
                tmpf_name = tmpf.name
                tmpf.close()
                torch.save(self.net.state_dict(), tmpf_name)
                server = InferenceServer(tmpf_name, req_q=self.req_q, res_dict=self.res_dict,
                                         board_size=board_size, device=infer_device,
                                         batch_size=infer_batch_size, num_filters=infer_num_filters)
                server.start()
                self.infer_server = server
                self._tmp_state_path = tmpf_name
            except Exception as e:
                print("[MCTS] Failed to start inference server, falling back to local net. Error:", e)
                self.infer_server = None
                self.req_q = None
                self.res_dict = None

        # tree stats
        self.P = {}
        self.N = defaultdict(int)
        self.W = defaultdict(float)
        self.Q = defaultdict(float)
        self.children = {}

    def close(self):
        # stop server and cleanup temp files
        if self.infer_server is not None and self.req_q is not None:
            try:
                # send stop signal
                self.req_q.put(("__STOP__", None))
                # wait for server to exit
                self.infer_server.join(timeout=5.0)
            except Exception:
                pass
            self.infer_server = None
        if hasattr(self, '_tmp_state_path') and os.path.exists(self._tmp_state_path):
            try:
                os.remove(self._tmp_state_path)
            except Exception:
                pass
        if self.manager is not None:
            try:
                self.manager.shutdown()
            except Exception:
                pass
            self.manager = None

    def state_key(self, board, player):
        return (board.tobytes(), int(player))

    def canonical_board(self, board, player):
        my = (board == player).astype(np.float32)
        opp = (board == -player).astype(np.float32)
        return np.stack([my, opp], axis=0)

    def add_dirichlet_noise(self, probs, eps=0.25, alpha=0.3):
        if len(probs) == 0:
            return probs
        noise = np.random.dirichlet([alpha] * len(probs))
        return probs * (1 - eps) + noise * eps

    def generate_candidate_moves(self, board, max_candidates=80, margin=2):
        if not (board != 0).any():
            c = self.board_size // 2
            return [c * self.board_size + c]
        bbox = get_bounding_box(board, self.board_size, margin=margin)
        if bbox is None:
            c = self.board_size // 2
            return [c * self.board_size + c]
        sx, ex, sy, ey = bbox
        candidates = []
        for x in range(sx, ex+1):
            for y in range(sy, ey+1):
                if board[x,y] == 0:
                    candidates.append(x * self.board_size + y)
        if not candidates:
            empties = np.where(board.flatten() == 0)[0].tolist()
            return empties
        if self.max_candidates is not None and len(candidates) > self.max_candidates:
            return list(np.random.choice(candidates, size=self.max_candidates, replace=False))
        return candidates

    def infer_via_server(self, planes, timeout=10.0):
        if self.req_q is None or self.res_dict is None:
            return None, None
        req_id = f"{os.getpid()}_{time.time_ns()}_{random.randint(0,1<<30)}"
        try:
            self.req_q.put((req_id, planes))
        except Exception:
            return None, None
        start = time.time()
        while True:
            if req_id in self.res_dict:
                try:
                    res = self.res_dict.pop(req_id)
                    return res
                except Exception:
                    return None, None
            if (time.time() - start) > timeout:
                return None, None
            time.sleep(0.001)

    def local_infer(self, planes):
        t = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(t)
        return logits.squeeze(0).cpu().numpy(), float(value.item())

    def get_action_probs(self, board, player, temp=1.0, add_root_noise=True):
        """
        Robust computation of action probabilities from visit counts.
        Avoids overflow by:
          - deterministic argmax if temp is very small
          - log-space normalization otherwise
        """
        root_key = self.state_key(board, player)
        # ensure we have at least expanded root once
        self.simulate(board.copy(), player)
        if add_root_noise and root_key in self.P:
            self.P[root_key] = self.add_dirichlet_noise(self.P[root_key])
        for i in range(max(0, self.n_sim - 1)):
            self.simulate(board.copy(), player)

        counts = np.array([self.N[(root_key, a)] for a in range(self.board_size*self.board_size)], dtype=np.float64)

        # if all counts are zero: fallback uniform over legal moves
        if np.all(counts == 0):
            legal = (board.flatten() == 0).astype(np.float64)
            legal_sum = legal.sum()
            if legal_sum == 0:
                return np.zeros_like(counts)
            return legal / legal_sum

        # If temp extremely small, use deterministic argmax to avoid huge exponents
        if temp <= 1e-3:
            best = int(np.argmax(counts))
            probs = np.zeros_like(counts)
            probs[best] = 1.0
            return probs

        # Use log-space normalization: log(counts) / temp -> subtract max -> exp -> normalize
        eps = 1e-12
        # Only consider positive counts in log (zeros will produce -inf)
        positive_mask = counts > 0
        if not positive_mask.any():
            legal = (board.flatten() == 0).astype(np.float64)
            legal_sum = legal.sum()
            if legal_sum == 0:
                return np.zeros_like(counts)
            return legal / legal_sum

        log_counts = np.full_like(counts, -np.inf, dtype=np.float64)
        log_counts[positive_mask] = np.log(counts[positive_mask] + eps) / (temp + 1e-12)

        # numeric stabilization
        max_log = np.max(log_counts[positive_mask])
        exp_vals = np.exp(log_counts - max_log)
        # zero out entries that were not legal (counts==0)
        exp_vals[~positive_mask] = 0.0

        s = exp_vals.sum()
        if s == 0 or not np.isfinite(s):
            legal = (board.flatten() == 0).astype(np.float64)
            legal_sum = legal.sum()
            if legal_sum == 0:
                return np.zeros_like(counts)
            return legal / legal_sum

        probs = exp_vals / s
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        return probs

    def simulate(self, board, player):
        path = []
        key = self.state_key(board, player)

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
            if best_a is None:
                break
            path.append((key, best_a))
            x = best_a // self.board_size
            y = best_a % self.board_size
            board[x, y] = player
            if is_five(board, x, y, player, self.board_size):
                value = 1.0
                for (s_key, a) in reversed(path):
                    self.N[(s_key, a)] += 1
                    self.W[(s_key, a)] += value
                    self.Q[(s_key, a)] = self.W[(s_key, a)] / self.N[(s_key, a)]
                    value = -value
                return
            player = -player
            key = self.state_key(board, player)

        # expansion: get network logits/value via server if available
        planes = self.canonical_board(board, player)
        logits = None; value = None
        if self.req_q is not None and self.res_dict is not None:
            logits, value = self.infer_via_server(planes)
        if logits is None:
            try:
                logits, value = self.local_infer(planes)
            except Exception:
                logits = np.zeros(self.board_size*self.board_size, dtype=np.float32)
                value = 0.0

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