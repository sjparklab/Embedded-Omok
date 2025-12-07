# -*- coding: utf-8 -*-
"""
team05_jdw.py - Minimax 기반 오목 AI 엔진
iot12345_student에서 import해서 사용
"""

from random import choice

INF = 10**9


class MinimaxEngine:
    """Minimax + Alpha-Beta 기반 오목 AI"""
    
    def __init__(self):
        pass
    
    # =========================================================================
    # 기본 유틸리티
    # =========================================================================
    
    def is_five(self, board, x, y, color, length=19):
        """5목 여부 검사"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            
            i, j = x + dx, y + dy
            while 0 <= i < length and 0 <= j < length and board[i][j] == color:
                count += 1
                i += dx
                j += dy
            
            i, j = x - dx, y - dy
            while 0 <= i < length and 0 <= j < length and board[i][j] == color:
                count += 1
                i -= dx
                j -= dy
            
            if count >= 5:
                return True
        return False
    
    def score_direction(self, board, x, y, color, dx, dy, length=19):
        """한 방향 패턴 점수"""
        count = 1
        open_ends = 0
        
        i, j = x + dx, y + dy
        while 0 <= i < length and 0 <= j < length and board[i][j] == color:
            count += 1
            i += dx
            j += dy
        if 0 <= i < length and 0 <= j < length and board[i][j] == 0:
            open_ends += 1
        
        i, j = x - dx, y - dy
        while 0 <= i < length and 0 <= j < length and board[i][j] == color:
            count += 1
            i -= dx
            j -= dy
        if 0 <= i < length and 0 <= j < length and board[i][j] == 0:
            open_ends += 1
        
        if count >= 5:
            return 100000
        if count == 4 and open_ends == 2:
            return 10000
        if count == 4 and open_ends == 1:
            return 5000
        if count == 3 and open_ends == 2:
            return 1000
        if count == 3 and open_ends == 1:
            return 200
        if count == 2 and open_ends == 2:
            return 100
        if count == 2 and open_ends == 1:
            return 10
        return count
    
    def score_move(self, board, x, y, color, length=19):
        """단일 수 점수"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        score = 0
        for dx, dy in directions:
            score += self.score_direction(board, x, y, color, dx, dy, length)
        return score
    
    def evaluate_move(self, board, x, y, my_color, opp_color, length=19):
        """공격 + 방어 점수"""
        original = board[x][y]
        
        board[x][y] = my_color
        my_score = self.score_move(board, x, y, my_color, length)
        board[x][y] = original
        
        board[x][y] = opp_color
        opp_score = self.score_move(board, x, y, opp_color, length)
        board[x][y] = original
        
        return my_score * 1.5 + opp_score * 1.2
    
    def evaluate_board(self, board, my_color, opp_color, length=19):
        """보드 전체 평가"""
        total = 0
        for x in range(length):
            for y in range(length):
                if board[x][y] == my_color:
                    total += self.score_move(board, x, y, my_color, length)
                elif board[x][y] == opp_color:
                    total -= self.score_move(board, x, y, opp_color, length)
        return total
    
    def has_any_stone(self, board, length=19):
        for i in range(length):
            for j in range(length):
                if board[i][j] != 0:
                    return True
        return False
    
    def get_bounding_box(self, board, length=19):
        min_x, max_x = length, -1
        min_y, max_y = length, -1
        for x in range(length):
            for y in range(length):
                if board[x][y] != 0:
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
        if max_x == -1:
            return None
        return (min_x, max_x, min_y, max_y)
    
    def generate_candidate_moves(self, board, my_color, opp_color, length=19, max_candidates=14):
        """휴리스틱 기반 후보 수 생성"""
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
                score = self.evaluate_move(board, x, y, my_color, opp_color, length)
                candidates.append((score, x, y))
        
        if not candidates:
            empty = [(i, j) for i in range(length) for j in range(length) if board[i][j] == 0]
            if not empty:
                return []
            return [choice(empty)]
        
        candidates.sort(reverse=True, key=lambda t: t[0])
        return [(x, y) for _, x, y in candidates[:max_candidates]]
    
    # =========================================================================
    # Minimax + Alpha-Beta
    # =========================================================================
    
    def minimax(self, board, depth, alpha, beta, maximizing, my_color, opp_color, length, last_move=None):
        if last_move is not None:
            lx, ly = last_move
            last_color = board[lx][ly]
            if self.is_five(board, lx, ly, last_color, length):
                return 1000000 if last_color == my_color else -1000000
        
        if depth == 0:
            return self.evaluate_board(board, my_color, opp_color, length)
        
        if maximizing:
            value = -INF
            moves = self.generate_candidate_moves(board, my_color, opp_color, length)
            if not moves:
                return self.evaluate_board(board, my_color, opp_color, length)
            
            for x, y in moves:
                board[x][y] = my_color
                score = self.minimax(board, depth - 1, alpha, beta, False, my_color, opp_color, length, (x, y))
                board[x][y] = 0
                value = max(value, score)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = INF
            moves = self.generate_candidate_moves(board, opp_color, my_color, length)
            if not moves:
                return self.evaluate_board(board, my_color, opp_color, length)
            
            for x, y in moves:
                board[x][y] = opp_color
                score = self.minimax(board, depth - 1, alpha, beta, True, my_color, opp_color, length, (x, y))
                board[x][y] = 0
                value = min(value, score)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value
    
    # =========================================================================
    # 메인 인터페이스
    # =========================================================================
    
    def get_best_move(self, board, my_color):
        """최선의 수 반환"""
        length = len(board)
        opp_color = -my_color
        
        if not self.has_any_stone(board, length):
            c = length // 2
            return c, c
        
        depth = 3
        best_score = -INF
        best_moves = []
        
        root_moves = self.generate_candidate_moves(board, my_color, opp_color, length)
        
        if not root_moves:
            empty = [(i, j) for i in range(length) for j in range(length) if board[i][j] == 0]
            if empty:
                return choice(empty)
            return length // 2, length // 2
        
        for x, y in root_moves:
            board[x][y] = my_color
            score = self.minimax(board, depth - 1, -INF, INF, False, my_color, opp_color, length, (x, y))
            board[x][y] = 0
            
            if score > best_score:
                best_score = score
                best_moves = [(x, y)]
            elif score == best_score:
                best_moves.append((x, y))
        
        return choice(best_moves)

