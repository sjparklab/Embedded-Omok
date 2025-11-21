# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:04:19 2020

Maker : bychoi@deu.ac.kr 

@author: Com
"""


from player import *
from stone import *
# ★ 우리가 만든 핵심 AI 파일 임포트
from team05_ai import GomokuEngine

class iot6789_student(player):
    def __init__(self, clr):
        # player._color 에 흑(-1) 값 들어옴
        super().__init__(clr)
        self.engine = GomokuEngine()

    def __del__(self):
        pass

    def next(self, board, length):
        # 1. 엔진에게 "이 판세에서 내 색깔(self.my_color)로 둘 수 내놔" 요청
        r, c = self.engine.get_best_move(board, self._color)

        # 2. 결과 포장해서 제출
        stn = stone(self._color, length)
        stn.setX(r)
        stn.setY(c)
        return stn