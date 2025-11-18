# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:04:19 2020

Maker : bychoi@deu.ac.kr 

@author: Com
"""

# 사용자(사람)가 두는 흑돌용 클래스
# player 클래스를 상속해서, next()에서 그대로 부모의 입력 로직을 사용한다.

from player import *
from stone import *


class iot6789_student(player):
    def __init__(self, clr):
        # player._color 에 흑(-1) 값 들어옴
        super().__init__(clr)

    def __del__(self):
        pass

    def next(self, board, length):
        """
        원래 player 클래스의 next()는
        - 콘솔에서 x, y 좌표를 입력받고
        - 범위 체크 후 stone 객체를 반환하는 로직임.

        여기서는 그걸 그대로 재사용해서,
        '학생용 클래스' 형태를 유지하면서도
        실제 동작은 사람이 직접 두도록 만든다.
        """
        return super().next(board, length)
