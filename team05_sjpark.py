# -*- coding: utf-8 -*-
"""
team05_ai.py - 개선된 오목 AI 엔진
주요 개선:
1. Transposition Table (중복 계산 방지)
2. 더 정교한 패턴 평가 (띈 3, 44, 33)
3. Killer Move Heuristic
4. Iterative Deepening
"""

import copy
from random import choice

INF = 10**9


class GomokuEngine:
    def __init__(self):
        # Transposition Table
        self.tt = {}
        self.tt_hits = 0
        
        # Killer Moves (좋았던 수 기억)
        self.killer_moves = {}
    
    def clear_cache(self):
        """캐시 초기화"""
        self.tt.clear()
        self.killer_moves.clear()
    
    # =========================================================================
    # 기본 유틸리티
    # =========================================================================
    
    def board_hash(self, board):
        """보드 상태를 해시로 변환 (Transposition Table용)"""
        return tuple(tuple(row) for row in board)
    
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
    
    def analyze_line(self, board, x, y, color, dx, dy, length=19):
        """
        한 방향 라인 분석 - 더 정교한 패턴 감지
        Returns: (연속 개수, 열린 끝 수, 빈칸 포함 패턴 길이)
        """
        count = 1
        open_ends = 0
        gaps = 0  # 띈 3 감지용
        pattern_length = 1
        
        # 정방향
        i, j = x + dx, y + dy
        gap_found = False
        while 0 <= i < length and 0 <= j < length:
            if board[i][j] == color:
                count += 1
                pattern_length += 1
            elif board[i][j] == 0 and not gap_found:
                # 한 칸 띈 패턴 체크
                ni, nj = i + dx, j + dy
                if 0 <= ni < length and 0 <= nj < length and board[ni][nj] == color:
                    gaps += 1
                    gap_found = True
                    pattern_length += 1
                    i, j = ni, nj
                    continue
                else:
                    open_ends += 1
                    break
            else:
                break
            i += dx
            j += dy
        
        # 역방향
        i, j = x - dx, y - dy
        gap_found = False
        while 0 <= i < length and 0 <= j < length:
            if board[i][j] == color:
                count += 1
                pattern_length += 1
            elif board[i][j] == 0 and not gap_found:
                ni, nj = i - dx, j - dy
                if 0 <= ni < length and 0 <= nj < length and board[ni][nj] == color:
                    gaps += 1
                    gap_found = True
                    pattern_length += 1
                    i, j = ni, nj
                    continue
                else:
                    open_ends += 1
                    break
            else:
                break
            i -= dx
            j -= dy
        
        return count, open_ends, gaps, pattern_length
    
    def score_direction(self, board, x, y, color, dx, dy, length=19):
        """개선된 방향별 점수 계산"""
        count, open_ends, gaps, pattern_len = self.analyze_line(board, x, y, color, dx, dy, length)
        
        # 5목
        if count >= 5:
            return 100000
        
        # 4목 패턴
        if count == 4:
            if open_ends == 2:
                return 50000   # 열린 4 = 승리
            elif open_ends == 1:
                return 5000    # 막힌 4
            else:
                return 500     # 양쪽 막힘
        
        # 띈 4 (X.XXX 또는 XX.XX)
        if count == 4 and gaps > 0:
            return 4000
        
        # 3목 패턴
        if count == 3:
            if open_ends == 2:
                return 3000    # 열린 3
            elif open_ends == 1:
                return 500     # 막힌 3
            else:
                return 50
        
        # 띈 3 (X.XX 또는 XX.X)
        if count == 3 and gaps > 0 and open_ends >= 1:
            return 1500  # 띈 열린 3
        
        # 2목 패턴
        if count == 2:
            if open_ends == 2:
                return 200     # 열린 2
            elif open_ends == 1:
                return 50      # 막힌 2
        
        return count * 5
    
    def score_move(self, board, x, y, color, length=19):
        """단일 수 점수"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        score = 0
        
        # 각 방향 점수 합산
        direction_scores = []
        for dx, dy in directions:
            ds = self.score_direction(board, x, y, color, dx, dy, length)
            direction_scores.append(ds)
            score += ds
        
        # 44 패턴 (두 개의 4가 교차) - 승리 확정
        fours = sum(1 for s in direction_scores if s >= 4000)
        if fours >= 2:
            score += 40000
        
        # 33 패턴 (두 개의 열린 3이 교차) - 매우 강력
        threes = sum(1 for s in direction_scores if 2500 <= s <= 3500)
        if threes >= 2:
            score += 8000
        
        # 43 패턴 (4와 열린 3)
        if fours >= 1 and threes >= 1:
            score += 15000
        
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
        
        # 공격을 더 중시 (선공 이점)
        return my_score * 1.3 + opp_score * 1.1
    
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
    
    # =========================================================================
    # 후보 수 생성
    # =========================================================================
    
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
    
    def generate_candidate_moves(self, board, my_color, opp_color, length=19, max_candidates=20):
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
                
                # Killer Move 보너스
                if (x, y) in self.killer_moves:
                    score += self.killer_moves[(x, y)] * 100
                
                candidates.append((score, x, y))
        
        if not candidates:
            empty = [(i, j) for i in range(length) for j in range(length) if board[i][j] == 0]
            if not empty:
                return []
            return [choice(empty)]
        
        candidates.sort(reverse=True, key=lambda t: t[0])
        return [(x, y) for _, x, y in candidates[:max_candidates]]
    
    # =========================================================================
    # 즉시 위협 탐지
    # =========================================================================
    
    def find_winning_move(self, board, color, length=19):
        """즉시 5목이 되는 수 찾기"""
        for x in range(length):
            for y in range(length):
                if board[x][y] == 0:
                    board[x][y] = color
                    if self.is_five(board, x, y, color, length):
                        board[x][y] = 0
                        return (x, y)
                    board[x][y] = 0
        return None
    
    def find_double_threat_move(self, board, color, opp_color, length=19):
        """44, 43, 33 같은 다중 위협 찾기"""
        for x in range(length):
            for y in range(length):
                if board[x][y] == 0:
                    board[x][y] = color
                    score = self.score_move(board, x, y, color, length)
                    board[x][y] = 0
                    
                    # 44 또는 43이면 승리 확정
                    if score >= 40000:
                        return (x, y)
        return None
    
    def count_winning_spots(self, board, color, length=19):
        """5목이 되는 빈칸 개수"""
        count = 0
        for x in range(length):
            for y in range(length):
                if board[x][y] == 0:
                    board[x][y] = color
                    if self.is_five(board, x, y, color, length):
                        count += 1
                    board[x][y] = 0
        return count
    
    # =========================================================================
    # Minimax + Alpha-Beta + Transposition Table
    # =========================================================================
    
    def minimax(self, board, depth, alpha, beta, maximizing, my_color, opp_color, length, last_move=None):
        # Transposition Table 체크
        board_key = (self.board_hash(board), depth, maximizing)
        if board_key in self.tt:
            self.tt_hits += 1
            return self.tt[board_key]
        
        # 승리 체크
        if last_move is not None:
            lx, ly = last_move
            last_color = board[lx][ly]
            if self.is_five(board, lx, ly, last_color, length):
                score = 1000000 + depth if last_color == my_color else -1000000 - depth
                return score
        
        # 깊이 제한
        if depth == 0:
            return self.evaluate_board(board, my_color, opp_color, length)
        
        current_color = my_color if maximizing else opp_color
        
        # 즉시 승리 체크
        win_move = self.find_winning_move(board, current_color, length)
        if win_move:
            return 1000000 + depth if maximizing else -1000000 - depth
        
        # 즉시 방어 필요 체크
        threat_color = opp_color if maximizing else my_color
        block_move = self.find_winning_move(board, threat_color, length)
        if block_move:
            x, y = block_move
            board[x][y] = current_color
            score = self.minimax(board, depth - 1, alpha, beta, not maximizing, my_color, opp_color, length, (x, y))
            board[x][y] = 0
            self.tt[board_key] = score
            return score
        
        if maximizing:
            value = -INF
            moves = self.generate_candidate_moves(board, my_color, opp_color, length, max_candidates=15)
            if not moves:
                return self.evaluate_board(board, my_color, opp_color, length)
            
            best_move = None
            for x, y in moves:
                board[x][y] = my_color
                score = self.minimax(board, depth - 1, alpha, beta, False, my_color, opp_color, length, (x, y))
                board[x][y] = 0
                
                if score > value:
                    value = score
                    best_move = (x, y)
                alpha = max(alpha, value)
                if alpha >= beta:
                    # Killer Move 기록
                    if best_move:
                        self.killer_moves[best_move] = self.killer_moves.get(best_move, 0) + 1
                    break
            
            self.tt[board_key] = value
            return value
        else:
            value = INF
            moves = self.generate_candidate_moves(board, opp_color, my_color, length, max_candidates=15)
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
            
            self.tt[board_key] = value
            return value
    
    # =========================================================================
    # Iterative Deepening
    # =========================================================================
    
    def iterative_deepening_search(self, board, my_color, opp_color, length, max_depth=4):
        """점진적 깊이 탐색"""
        best_move = None
        best_score = -INF
        
        root_moves = self.generate_candidate_moves(board, my_color, opp_color, length, max_candidates=20)
        
        if not root_moves:
            return length // 2, length // 2
        
        # 깊이 1부터 max_depth까지 점진적 탐색
        for depth in range(1, max_depth + 1):
            current_best_move = None
            current_best_score = -INF
            
            # 이전 탐색 결과로 Move Ordering
            scored_moves = []
            for x, y in root_moves:
                board[x][y] = my_color
                score = self.minimax(board, depth - 1, -INF, INF, False, my_color, opp_color, length, (x, y))
                board[x][y] = 0
                scored_moves.append((score, x, y))
                
                if score > current_best_score:
                    current_best_score = score
                    current_best_move = (x, y)
            
            # 다음 반복을 위해 정렬
            scored_moves.sort(reverse=True, key=lambda t: t[0])
            root_moves = [(x, y) for _, x, y in scored_moves]
            
            best_move = current_best_move
            best_score = current_best_score
            
            # 승리 확정이면 조기 종료
            if best_score >= 900000:
                break
        
        return best_move if best_move else (length // 2, length // 2)
    
    # =========================================================================
    # 메인 인터페이스
    # =========================================================================
    
    def get_best_move(self, board, my_color):
        """최선의 수 반환"""
        length = len(board)
        opp_color = -my_color
        
        # 캐시 제한 (메모리 관리)
        if len(self.tt) > 100000:
            self.tt.clear()
        
        # 1. 돌이 없으면 중앙
        if not self.has_any_stone(board, length):
            c = length // 2
            return c, c
        
        # 2. 즉시 승리 체크
        win_move = self.find_winning_move(board, my_color, length)
        if win_move:
            return win_move
        
        # 3. 즉시 방어 체크
        block_move = self.find_winning_move(board, opp_color, length)
        if block_move:
            return block_move
        
        # 4. 다중 위협 (44, 43) 체크
        double_threat = self.find_double_threat_move(board, my_color, opp_color, length)
        if double_threat:
            return double_threat
        
        # 5. 상대 다중 위협 방어
        opp_double_threat = self.find_double_threat_move(board, opp_color, my_color, length)
        if opp_double_threat:
            return opp_double_threat
        
        # 6. Iterative Deepening Search
        return self.iterative_deepening_search(board, my_color, opp_color, length, max_depth=4)
