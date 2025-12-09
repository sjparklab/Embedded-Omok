# iot12345_az.py
# AlphaZero-style 플레이어 (player 인터페이스 호환)
# num_filters 파라미터를 받아 모델 아키텍처를 체크포인트와 맞출 수 있도록 수정함.

from player import *
from stone import *
import numpy as np
import torch
from alphazero_model import AlphaZeroNet
from mcts import MCTS
import random
import os

class iot12345_az(player):
    def __init__(self, clr, model_path='checkpoints_finetune\\checkpoint_final.pt', board_size=19, device='cpu', n_sim=200, num_filters=64):
        super().__init__(clr)
        self.board_size = board_size
        self.device = device
        self.num_filters = num_filters
        # create network with requested num_filters
        self.net = AlphaZeroNet(board_size=board_size, num_filters=num_filters).to(device)

        if model_path:
            # Robust load: handle checkpoint dicts {'model':..., ...} or raw state_dict
            self._load_model_checkpoint(self.net, model_path, device)

        self.net.eval()
        # enable candidate pruning (max_candidates) to focus search near existing stones
        self.mcts = MCTS(self.net, board_size=board_size, c_puct=1.0, n_sim=n_sim, device=device, max_candidates=200)

    def _load_model_checkpoint(self, net, model_path, device):
        if not os.path.exists(model_path):
            print(f"[iot12345_az] model_path not found: {model_path}")
            return
        try:
            ck = torch.load(model_path, map_location=device)
        except Exception as e:
            print(f"[iot12345_az] Failed to torch.load({model_path}): {e}")
            return

        # Determine the actual state_dict
        state_dict = None
        if isinstance(ck, dict):
            for key in ('model', 'state_dict', 'model_state_dict', 'state_dicts'):
                if key in ck:
                    state_dict = ck[key]
                    break
            if state_dict is None:
                sample_keys = list(ck.keys())[:10]
                if sample_keys and all(isinstance(k, str) and ('.' in k or 'conv' in k or 'fc' in k or 'bn' in k) for k in sample_keys):
                    state_dict = ck
        else:
            state_dict = None

        if state_dict is None:
            print(f"[iot12345_az] No model state_dict found inside checkpoint {model_path}. Skipping load.")
            return

        # Strip 'module.' prefix if present
        new_state = {}
        for k, v in state_dict.items():
            new_k = k[len('module.'):] if k.startswith('module.') else k
            new_state[new_k] = v

        # Remove keys that do not match in shape to avoid size mismatch errors.
        model_state = net.state_dict()
        filtered_state = {}
        mismatched = []
        for k, v in new_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                mismatched.append((k, v.shape if hasattr(v, 'shape') else None, model_state[k].shape if k in model_state else None))

        if filtered_state:
            try:
                net.load_state_dict(filtered_state, strict=False)
                print(f"[iot12345_az] Loaded {len(filtered_state)} matching params from {model_path} (strict=False).")
            except Exception as e:
                print(f"[iot12345_az] Error loading filtered state_dict: {e}")
        else:
            print(f"[iot12345_az] No matching parameter shapes found to load from checkpoint {model_path}.")

        if mismatched:
            print(f"[iot12345_az] Warning: {len(mismatched)} parameter(s) had mismatched shapes and were skipped. Examples:")
            for item in mismatched[:10]:
                print("  ", item)

    def has_any_stone(self, board, length):
        for i in range(length):
            for j in range(length):
                if board[i][j] != 0:
                    return True
        return False

    def next(self, board, length):  # override
        print(" **** White (AlphaZero) player : My Turns **** ")
        stn = stone(self._color, length)
        my_color = self._color

        # center if empty (works both when playing as black or white if first move)
        if not self.has_any_stone(board, length):
            c = length // 2
            stn.setX(c); stn.setY(c)
            print(" === AlphaZero player placed at center ===")
            return stn

        board_np = np.array(board, dtype=int)
        probs = self.mcts.get_action_probs(board_np, my_color, temp=0.5)
        if probs.sum() == 0:
            empties = [(i,j) for i in range(length) for j in range(length) if board_np[i,j]==0]
            if not empties:
                stn.setX(length//2); stn.setY(length//2)
                return stn
            x,y = random.choice(empties)
            stn.setX(x); stn.setY(y)
            return stn

        best = int(np.argmax(probs))
        bx = best // self.board_size
        by = best % self.board_size
        stn.setX(int(bx)); stn.setY(int(by))
        print(f" === AlphaZero player chose ({bx}, {by}) ===")
        return stn