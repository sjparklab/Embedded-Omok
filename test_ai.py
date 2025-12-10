# -*- coding: utf-8 -*-
"""
KataGomo AI 플레이어에 대한 간단한 테스트
"""

from katagomo_ai_player import katagomo_ai_player
from stone import stone
import logging
logging.basicConfig(level=logging.DEBUG)

# AI 요청
print("Creating KataGomo AI player...")
ai = katagomo_ai_player(-1)  # Black

print("AI player created successfully!")

# 간단한 테스트용 보드 생성
board = [[0 for i in range(19)] for j in range(19)]

print("Generating first move...")
move = ai.next(board, 19)

print(f"AI generated move: x={move.getX()}, y={move.getY()}")

print("Test completed!")
