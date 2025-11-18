# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:37:40 2020
Maker : bychoi@deu.ac.kr 

@author: Com
"""

# sample player file which must be made by student 

from player import *
from stone import *
from random import choice

INF = 10**9


class iot12345_student(player):
    def __init__(self, clr):
        super().__init__(clr)  # call constructor of super class

    def __del__(self):  # destructor
        pass

    # ===================== Helper methods (공통 유틸) =====================

    def is_five(self, board, x, y, color, length):
        """(x, y)에 color가 놓여 있다고 보고 5목 여부 검사"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1

            # 정방향
            i, j = x + dx, y + dy
            while 0 <= i < length and 0 <= j < length and board[i][j] == color:
                count += 1
                i += dx
                j += dy

            # 역방향
            i, j = x - dx, y - dy
            while 0 <= i < length and 0 <= j < length and board[i][j] == color:
                count += 1
                i -= dx
                j -= dy

            if count >= 5:
                return True
        return False

    def score_direction(self, board, x, y, color, dx, dy, length):
        """
        한 방향(dx, dy)에 대해 (x, y)에 color를 뒀다고 가정했을 때
        연속 개수와 양쪽 열린 정도를 이용해 점수 계산
        """
        count = 1
        open_ends = 0

        # 정방향
        i, j = x + dx, y + dy
        while 0 <= i < length and 0 <= j < length and board[i][j] == color:
            count += 1
            i += dx
            j += dy
        if 0 <= i < length and 0 <= j < length and board[i][j] == 0:
            open_ends += 1

        # 역방향
        i, j = x - dx, y - dy
        while 0 <= i < length and 0 <= j < length and board[i][j] == color:
            count += 1
            i -= dx
            j -= dy
        if 0 <= i < length and 0 <= j < length and board[i][j] == 0:
            open_ends += 1

        # 패턴별 가중치 (간단 휴리스틱)
        if count >= 5:
            return 100000  # 승리 수
        if count == 4 and open_ends == 2:
            return 10000   # 열린 4
        if count == 4 and open_ends == 1:
            return 5000    # 막힌 4
        if count == 3 and open_ends == 2:
            return 1000    # 열린 3
        if count == 3 and open_ends == 1:
            return 200     # 막힌 3
        if count == 2 and open_ends == 2:
            return 100     # 열린 2
        if count == 2 and open_ends == 1:
            return 10      # 막힌 2

        return count

    def score_move(self, board, x, y, color, length):
        """(x, y)에 color를 둔다고 가정했을 때의 점수"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        score = 0
        for dx, dy in directions:
            score += self.score_direction(board, x, y, color, dx, dy, length)
        return score

    def evaluate_move(self, board, x, y, my_color, opp_color, length):
        """
        한 수에 대해:
        - 내 입장에서의 공격 점수
        - 상대 입장에서의 방어 점수
        를 합쳐서 평가
        """
        original = board[x][y]

        # 내 수
        board[x][y] = my_color
        my_score = self.score_move(board, x, y, my_color, length)
        board[x][y] = original

        # 상대가 둔다고 가정했을 때
        board[x][y] = opp_color
        opp_score = self.score_move(board, x, y, opp_color, length)
        board[x][y] = original

        return my_score * 1.5 + opp_score * 1.2

    def evaluate_board(self, board, my_color, opp_color, length):
        """현재 보드 전체를 my_color 관점에서 평가"""
        total = 0
        for x in range(length):
            for y in range(length):
                if board[x][y] == my_color:
                    total += self.score_move(board, x, y, my_color, length)
                elif board[x][y] == opp_color:
                    total -= self.score_move(board, x, y, opp_color, length)
        return total

    def has_any_stone(self, board, length):
        for i in range(length):
            for j in range(length):
                if board[i][j] != 0:
                    return True
        return False

    def get_bounding_box(self, board, length):
        """돌들이 놓여 있는 최소 bounding box (없으면 None)"""
        min_x, max_x = length, -1
        min_y, max_y = length, -1
        for x in range(length):
            for y in range(length):
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
        return (min_x, max_x, min_y, max_y)

    def generate_candidate_moves(self, board, length, color, my_color, opp_color, max_candidates=12):
        """
        후보 수를 제한해서 반환:
        - 돌이 하나도 없으면 중앙 한 점
        - 아니면 현재 돌 주변 2칸 범위 내의 빈칸들 중에서
          휴리스틱 점수 상위 max_candidates 개
        """
        # 돌이 하나도 없는 경우
        if not self.has_any_stone(board, length):
            c = length // 2
            return [(c, c)]

        bbox = self.get_bounding_box(board, length)
        if bbox is None:
            c = length // 2
            return [(c, c)]

        min_x, max_x, min_y, max_y = bbox
        margin = 2
        sx = max(0, min_x - margin)
        ex = min(length - 1, max_x + margin)
        sy = max(0, min_y - margin)
        ey = min(length - 1, max_y + margin)

        candidates = []
        for x in range(sx, ex + 1):
            for y in range(sy, ey + 1):
                if board[x][y] != 0:
                    continue
                # color 쪽 관점에서의 휴리스틱 (공격/수비 섞기)
                if color == my_color:
                    score = self.evaluate_move(board, x, y, my_color, opp_color, length)
                else:
                    score = self.evaluate_move(board, x, y, opp_color, my_color, length)
                candidates.append((score, x, y))

        if not candidates:
            # bounding box 안에 빈칸이 하나도 없으면 전체에서 아무 빈칸이나
            empty = [(i, j) for i in range(length) for j in range(length) if board[i][j] == 0]
            if not empty:
                return []
            return [choice(empty)]

        # 점수 순으로 정렬 후 상위 max_candidates만 사용
        candidates.sort(reverse=True, key=lambda t: t[0])
        top = candidates[:max_candidates]
        return [(x, y) for _, x, y in top]

    # ===================== Minimax + 알파–베타 =====================

    def minimax(self, board, depth, alpha, beta,
                maximizing, my_color, opp_color, length,
                last_move=None, last_color=None):
        """
        my_color 관점에서의 minimax + alpha-beta 가지치기
        """
        # 직전에 둔 수로 승/패가 결정됐는지 먼저 확인
        if last_move is not None and last_color is not None:
            lx, ly = last_move
            if self.is_five(board, lx, ly, last_color, length):
                if last_color == my_color:
                    return 1000000  # 내가 이김
                else:
                    return -1000000  # 내가 짐

        # 더 이상 내려가지 않으면 보드 평가
        if depth == 0:
            return self.evaluate_board(board, my_color, opp_color, length)

        # 둘 곳이 없으면 평가
        if all(board[i][j] != 0 for i in range(length) for j in range(length)):
            return self.evaluate_board(board, my_color, opp_color, length)

        if maximizing:
            value = -INF
            color_to_play = my_color
            moves = self.generate_candidate_moves(board, length, color_to_play,
                                                  my_color, opp_color)
            if not moves:
                return self.evaluate_board(board, my_color, opp_color, length)

            for x, y in moves:
                board[x][y] = color_to_play
                score = self.minimax(board, depth - 1, alpha, beta,
                                     False, my_color, opp_color, length,
                                     last_move=(x, y), last_color=color_to_play)
                board[x][y] = 0
                if score > value:
                    value = score
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break  # beta cut-off
            return value
        else:
            value = INF
            color_to_play = opp_color
            moves = self.generate_candidate_moves(board, length, color_to_play,
                                                  my_color, opp_color)
            if not moves:
                return self.evaluate_board(board, my_color, opp_color, length)

            for x, y in moves:
                board[x][y] = color_to_play
                score = self.minimax(board, depth - 1, alpha, beta,
                                     True, my_color, opp_color, length,
                                     last_move=(x, y), last_color=color_to_play)
                board[x][y] = 0
                if score < value:
                    value = score
                if value < beta:
                    beta = value
                if alpha >= beta:
                    break  # alpha cut-off
            return value

    # ===================== 메인 AI 함수 =====================

    def next(self, board, length):  # override
        print(" **** White player : My Turns **** ")

        stn = stone(self._color, length)
        my_color = self._color
        opp_color = -self._color

        # 돌이 아예 없으면 중앙
        if not self.has_any_stone(board, length):
            c = length // 2
            stn.setX(c)
            stn.setY(c)
            print(" === White player was completed ==== ")
            return stn

        depth = 2  # 내 수 → 상대 수까지 본다

        best_score = -INF
        best_moves = []

        # 루트에서의 후보 수
        root_moves = self.generate_candidate_moves(board, length, my_color,
                                                   my_color, opp_color)

        if not root_moves:
            # 비정상적 상황: 그냥 아무 빈칸이나 둔다
            empty = [(i, j) for i in range(length) for j in range(length) if board[i][j] == 0]
            if not empty:
                c = length // 2
                stn.setX(c)
                stn.setY(c)
            else:
                x, y = choice(empty)
                stn.setX(x)
                stn.setY(y)
            print(" === White player was completed ==== ")
            return stn

        for x, y in root_moves:
            board[x][y] = my_color
            score = self.minimax(board, depth - 1, -INF, INF,
                                 False, my_color, opp_color, length,
                                 last_move=(x, y), last_color=my_color)
            board[x][y] = 0

            if score > best_score:
                best_score = score
                best_moves = [(x, y)]
            elif score == best_score:
                best_moves.append((x, y))

        bx, by = choice(best_moves)
        stn.setX(bx)
        stn.setY(by)
        print(" === White player was completed ==== ")
        return stn
