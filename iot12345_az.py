from player import *
from stone import *
import numpy as np
import torch
from alphazero_model import AlphaZeroNet
from mcts import MCTS
import random

class iot12345_az(player):
    def __init__(self, clr, model_path=None, board_size=19, device='cpu', n_sim=200):
        super().__init__(clr)
        self.board_size = board_size
        self.device = device
        self.net = AlphaZeroNet(board_size=board_size).to(device)
        if model_path:
            # model_path이 None이 아니면 로드 시도
            self.net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        self.net.eval()
        self.mcts = MCTS(self.net, board_size=board_size, c_puct=1.0, n_sim=n_sim, device=device)

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

        # 중앙이 비어있으면
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