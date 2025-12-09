# -*- coding: utf-8 -*-
"""
team05_ai.py - 고성능 오목 AI 엔진
주요 기법:
1. Transposition Table + Zobrist Hashing
2. 정교한 패턴 평가 (연속, 띈 패턴, 복합 위협)
3. Killer Move + History Heuristic
4. Iterative Deepening + Principal Variation Search
5. VCT (Victory by Continuous Threats) 탐색
6. 효율적인 후보 수 생성 (인접 셀 기반)
"""

from random import choice, randint
import time

INF = 10**9
TIME_LIMIT = 4.5  # 5초 제한에서 여유 0.5초

# Zobrist Hashing용 랜덤 테이블 (전역, 재사용)
_ZOBRIST_TABLE = None
_ZOBRIST_SIZE = 19


def _init_zobrist(size=19):
    """Zobrist 해시 테이블 초기화"""
    global _ZOBRIST_TABLE, _ZOBRIST_SIZE
    if _ZOBRIST_TABLE is not None and _ZOBRIST_SIZE == size:
        return _ZOBRIST_TABLE

    _ZOBRIST_SIZE = size
    _ZOBRIST_TABLE = {}
    for i in range(size):
        for j in range(size):
            for color in [-1, 1]:
                _ZOBRIST_TABLE[(i, j, color)] = randint(0, 2**63 - 1)
    return _ZOBRIST_TABLE


class GomokuEngine:
    def __init__(self):
        # Transposition Table
        self.tt = {}
        self.tt_hits = 0

        # Killer Moves (depth별)
        self.killer_moves = {}

        # History Heuristic
        self.history = {}

        # Zobrist Hash
        self.zobrist = None
        self.current_hash = 0

        # 방향 벡터 (상수화)
        self.DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]

        # 인접 오프셋 (거리 1, 2)
        self.ADJACENT_OFFSETS = []
        for d in range(1, 3):
            for dx in range(-d, d + 1):
                for dy in range(-d, d + 1):
                    if dx != 0 or dy != 0:
                        if (dx, dy) not in self.ADJACENT_OFFSETS:
                            self.ADJACENT_OFFSETS.append((dx, dy))

        # 시간 제한
        self.start_time = 0
        self.time_out = False

    def is_timeout(self):
        """시간 초과 체크"""
        if time.time() - self.start_time > TIME_LIMIT:
            self.time_out = True
            return True
        return False

    def clear_cache(self):
        """캐시 초기화"""
        self.tt.clear()
        self.killer_moves.clear()
        self.history.clear()
        self.tt_hits = 0

    # =========================================================================
    # Zobrist Hashing
    # =========================================================================

    def init_hash(self, board, length):
        """보드의 초기 해시값 계산"""
        self.zobrist = _init_zobrist(length)
        h = 0
        for i in range(length):
            for j in range(length):
                if board[i][j] != 0:
                    h ^= self.zobrist[(i, j, board[i][j])]
        self.current_hash = h
        return h

    def update_hash(self, x, y, color):
        """해시 증분 업데이트"""
        if self.zobrist and (x, y, color) in self.zobrist:
            self.current_hash ^= self.zobrist[(x, y, color)]

    # =========================================================================
    # 기본 유틸리티
    # =========================================================================

    def is_five(self, board, x, y, color, length=19):
        """5목 여부 검사 (최적화)"""
        for dx, dy in self.DIRECTIONS:
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

    def get_line_pattern(self, board, x, y, color, dx, dy, length=19):
        """
        라인 패턴 분석 - 더 정교한 버전
        Returns: (연속돌, 열린끝, 총패턴길이, 빈칸패턴)
        빈칸패턴: 띈 형태 감지 (X_XX, XX_X 등)
        """
        opp = -color

        # 중심에서 양방향으로 최대 5칸씩 분석
        line = []  # 중심 기준 -5 ~ +5
        for step in range(-5, 6):
            ni, nj = x + dx * step, y + dy * step
            if 0 <= ni < length and 0 <= nj < length:
                line.append(board[ni][nj])
            else:
                line.append(opp)  # 벽은 상대돌처럼 처리

        # 연속 카운트 (중심 포함)
        center = 5
        count = 1

        # 정방향 연속
        i = center + 1
        while i < 11 and line[i] == color:
            count += 1
            i += 1
        end_right = i

        # 역방향 연속
        i = center - 1
        while i >= 0 and line[i] == color:
            count += 1
            i -= 1
        end_left = i

        # 열린 끝 체크
        open_ends = 0
        if end_left >= 0 and line[end_left] == 0:
            open_ends += 1
        if end_right < 11 and line[end_right] == 0:
            open_ends += 1

        # 띈 패턴 감지 (한 칸 빈 곳 너머 같은 돌)
        gap_count = 0
        gap_total = count

        # 정방향 갭 체크: 0 뒤에 같은 색
        if end_right < 10 and line[end_right] == 0:
            if line[end_right + 1] == color:
                gap_count += 1
                gap_total += 1
                # 더 연속되는지 체크
                gi = end_right + 2
                while gi < 11 and line[gi] == color:
                    gap_total += 1
                    gi += 1

        # 역방향 갭 체크
        if end_left >= 1 and line[end_left] == 0:
            if line[end_left - 1] == color:
                gap_count += 1
                gap_total += 1
                gi = end_left - 2
                while gi >= 0 and line[gi] == color:
                    gap_total += 1
                    gi -= 1

        return count, open_ends, gap_total, gap_count

    def score_pattern(self, count, open_ends, gap_total, gap_count):
        """패턴을 점수로 변환"""
        # 5목 이상
        if count >= 5:
            return 100000

        # 갭 포함 5 이상 (띈 5)
        if gap_total >= 5 and gap_count > 0:
            return 90000

        # 열린 4 (XXXX_)
        if count == 4:
            if open_ends == 2:
                return 50000  # 양쪽 열린 4 = 거의 승리
            elif open_ends == 1:
                return 5000   # 한쪽 열린 4
            else:
                return 200    # 막힌 4

        # 띈 4 (X_XXX, XX_XX)
        if gap_total == 4 and gap_count > 0:
            if open_ends >= 1:
                return 4500
            return 400

        # 열린 3
        if count == 3:
            if open_ends == 2:
                return 3000   # 양쪽 열린 3 = 매우 강력
            elif open_ends == 1:
                return 400    # 한쪽 열린 3
            else:
                return 20     # 막힌 3

        # 띈 3 (X_XX, XX_X)
        if gap_total == 3 and gap_count > 0:
            if open_ends == 2:
                return 1800   # 양쪽 열린 띈 3
            elif open_ends == 1:
                return 300
            return 15

        # 열린 2
        if count == 2:
            if open_ends == 2:
                return 150
            elif open_ends == 1:
                return 30
            return 5

        # 띈 2
        if gap_total == 2 and gap_count > 0:
            if open_ends >= 1:
                return 50
            return 5

        return count * 2

    def score_position(self, board, x, y, color, length=19):
        """특정 위치의 돌 점수 (모든 방향)"""
        if board[x][y] != color:
            return 0

        total_score = 0
        direction_scores = []

        for dx, dy in self.DIRECTIONS:
            count, open_ends, gap_total, gap_count = self.get_line_pattern(
                board, x, y, color, dx, dy, length
            )
            ds = self.score_pattern(count, open_ends, gap_total, gap_count)
            direction_scores.append((ds, count, open_ends))
            total_score += ds

        # 복합 위협 보너스 (44, 43, 33)
        fours = sum(1 for ds, cnt, _ in direction_scores if ds >= 4000 or cnt >= 4)
        open_threes = sum(1 for ds, cnt, opens in direction_scores
                         if 2500 <= ds <= 3500 or (cnt == 3 and opens == 2))

        if fours >= 2:
            total_score += 45000  # 44 = 필승
        if fours >= 1 and open_threes >= 1:
            total_score += 20000  # 43 = 필승
        if open_threes >= 2:
            total_score += 10000  # 33 = 매우 강력

        return total_score

    def score_move_potential(self, board, x, y, color, length=19):
        """빈 칸에 돌을 놓았을 때의 잠재 점수"""
        board[x][y] = color
        score = self.score_position(board, x, y, color, length)
        board[x][y] = 0
        return score

    def evaluate_board(self, board, my_color, opp_color, length=19):
        """보드 전체 평가 (최적화)"""
        my_score = 0
        opp_score = 0

        for x in range(length):
            for y in range(length):
                if board[x][y] == my_color:
                    my_score += self.score_position(board, x, y, my_color, length)
                elif board[x][y] == opp_color:
                    opp_score += self.score_position(board, x, y, opp_color, length)

        return my_score - opp_score * 1.1  # 방어 우선 (상대 위협 차단)

    # =========================================================================
    # 효율적인 후보 수 생성
    # =========================================================================

    def has_neighbor(self, board, x, y, length, dist=2):
        """주변에 돌이 있는지 체크"""
        for dx in range(-dist, dist + 1):
            for dy in range(-dist, dist + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < length and 0 <= ny < length:
                    if board[nx][ny] != 0:
                        return True
        return False

    def get_empty_neighbors(self, board, length, dist=2):
        """돌 주변의 빈 칸들 반환"""
        neighbors = set()
        for x in range(length):
            for y in range(length):
                if board[x][y] != 0:
                    for dx in range(-dist, dist + 1):
                        for dy in range(-dist, dist + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < length and 0 <= ny < length:
                                if board[nx][ny] == 0:
                                    neighbors.add((nx, ny))
        return neighbors

    def generate_moves(self, board, my_color, opp_color, length=19, max_moves=20):
        """휴리스틱 기반 후보 수 생성"""
        # 빈 보드면 중앙
        empty_neighbors = self.get_empty_neighbors(board, length, dist=2)
        if not empty_neighbors:
            c = length // 2
            return [(c, c)]

        candidates = []

        for x, y in empty_neighbors:
            # 내 점수
            my_potential = self.score_move_potential(board, x, y, my_color, length)
            # 상대 점수 (방어 가치)
            opp_potential = self.score_move_potential(board, x, y, opp_color, length)

            # 방어 우선 (상대 위협 먼저 차단)
            score = my_potential + opp_potential * 1.15

            # History Heuristic 보너스
            if (x, y) in self.history:
                score += self.history[(x, y)] * 10

            # Killer Move 보너스
            for depth in self.killer_moves:
                if (x, y) in self.killer_moves[depth]:
                    score += 500

            candidates.append((score, x, y))

        # 점수순 정렬
        candidates.sort(reverse=True, key=lambda t: t[0])

        return [(x, y) for _, x, y in candidates[:max_moves]]

    # =========================================================================
    # 위협 탐지 (최적화)
    # =========================================================================

    def find_winning_move(self, board, color, length=19):
        """즉시 5목 수 찾기 - 모든 가능한 위치 체크"""
        # 4개 연속인 라인의 빈 끝을 찾아야 하므로 dist=2 필요
        neighbors = self.get_empty_neighbors(board, length, dist=2)
        for x, y in neighbors:
            board[x][y] = color
            if self.is_five(board, x, y, color, length):
                board[x][y] = 0
                return (x, y)
            board[x][y] = 0
        return None

    def find_forcing_move(self, board, color, opp_color, length=19):
        """44, 43 같은 필승 수 찾기"""
        neighbors = self.get_empty_neighbors(board, length, dist=2)
        for x, y in neighbors:
            board[x][y] = color
            score = self.score_position(board, x, y, color, length)
            board[x][y] = 0

            if score >= 40000:  # 44 또는 43
                return (x, y)
        return None

    def find_open_four(self, board, color, length=19):
        """열린 4를 만드는 수 찾기"""
        neighbors = self.get_empty_neighbors(board, length, dist=2)
        for x, y in neighbors:
            board[x][y] = color
            for dx, dy in self.DIRECTIONS:
                count, open_ends, gap_total, gap_count = self.get_line_pattern(
                    board, x, y, color, dx, dy, length
                )
                if count == 4 and open_ends == 2:
                    board[x][y] = 0
                    return (x, y)
            board[x][y] = 0
        return None

    def find_open_three(self, board, color, length=19):
        """열린 3을 만드는 수 찾기"""
        neighbors = self.get_empty_neighbors(board, length, dist=2)
        best_moves = []
        for x, y in neighbors:
            board[x][y] = color
            for dx, dy in self.DIRECTIONS:
                count, open_ends, gap_total, gap_count = self.get_line_pattern(
                    board, x, y, color, dx, dy, length
                )
                if count == 3 and open_ends == 2:
                    best_moves.append((x, y))
                    break
            board[x][y] = 0
        return best_moves[0] if best_moves else None

    def count_threats(self, board, color, length=19):
        """위협(4, 열린3) 개수 세기"""
        fours = 0
        open_threes = 0

        checked = set()
        for x in range(length):
            for y in range(length):
                if board[x][y] == color:
                    for dx, dy in self.DIRECTIONS:
                        line_key = (x, y, dx, dy)
                        if line_key in checked:
                            continue
                        checked.add(line_key)

                        count, open_ends, gap_total, gap_count = self.get_line_pattern(
                            board, x, y, color, dx, dy, length
                        )
                        if count >= 4 or gap_total >= 4:
                            fours += 1
                        elif count == 3 and open_ends == 2:
                            open_threes += 1

        return fours, open_threes

    # =========================================================================
    # VCT (Victory by Continuous Threats) - 위협 수순 탐색
    # =========================================================================

    def vct_attack(self, board, color, opp_color, length, depth, max_depth=6):
        """
        VCT 공격 탐색
        연속 위협으로 승리 가능한지 탐색
        """
        if depth >= max_depth or self.is_timeout():
            return None

        # 즉시 승리 체크
        win = self.find_winning_move(board, color, length)
        if win:
            return win

        # 열린 4 찾기 (상대가 막아야 함)
        open_four = self.find_open_four(board, color, length)
        if open_four:
            return open_four

        # 44/43 필승 체크
        forcing = self.find_forcing_move(board, color, opp_color, length)
        if forcing:
            return forcing

        # 4를 만드는 수들 탐색
        threat_moves = []
        neighbors = self.get_empty_neighbors(board, length, dist=2)

        for x, y in neighbors:
            board[x][y] = color
            for dx, dy in self.DIRECTIONS:
                count, open_ends, gap_total, gap_count = self.get_line_pattern(
                    board, x, y, color, dx, dy, length
                )
                # 막힌 4나 띈 4 (상대가 막아야 하는 위협)
                if (count == 4 and open_ends >= 1) or (gap_total == 4 and gap_count > 0):
                    threat_moves.append((x, y))
                    break
            board[x][y] = 0

        # 각 위협에 대해 상대 방어 후 재귀 탐색
        for tx, ty in threat_moves[:5]:  # 상위 5개만
            board[tx][ty] = color

            # 상대 방어 (5목 막기)
            defense = self.find_winning_move(board, color, length)
            if defense:
                # 내가 이미 5목 가능
                board[tx][ty] = 0
                return (tx, ty)

            defense = self.find_winning_move(board, opp_color, length)
            if defense is None:
                # 상대가 막을 곳이 없음 = 내 승리
                board[tx][ty] = 0
                return (tx, ty)

            # 상대 방어
            dx, dy = defense
            board[dx][dy] = opp_color

            # 재귀적으로 다음 위협 탐색
            next_threat = self.vct_attack(board, color, opp_color, length, depth + 1, max_depth)

            board[dx][dy] = 0
            board[tx][ty] = 0

            if next_threat:
                return (tx, ty)

        return None

    # =========================================================================
    # Minimax + Alpha-Beta + PVS
    # =========================================================================

    def minimax(self, board, depth, alpha, beta, maximizing, my_color, opp_color,
                length, last_move=None, is_pv=True):
        """Alpha-Beta with Principal Variation Search"""

        # 시간 초과 체크
        if self.time_out or self.is_timeout():
            return 0  # 시간 초과 시 중립 점수 반환

        # Transposition Table 체크 (my_color 포함하여 관점 구분)
        board_key = (self.current_hash, depth, maximizing, my_color)
        if board_key in self.tt:
            stored = self.tt[board_key]
            if stored['depth'] >= depth:
                self.tt_hits += 1
                if stored['flag'] == 'exact':
                    return stored['value']
                elif stored['flag'] == 'lower' and stored['value'] > alpha:
                    alpha = stored['value']
                elif stored['flag'] == 'upper' and stored['value'] < beta:
                    beta = stored['value']
                if alpha >= beta:
                    return stored['value']

        # 승리 체크
        if last_move:
            lx, ly = last_move
            last_color = board[lx][ly]
            if self.is_five(board, lx, ly, last_color, length):
                return 1000000 + depth if last_color == my_color else -1000000 - depth

        # 깊이 제한
        if depth <= 0:
            return self.evaluate_board(board, my_color, opp_color, length)

        current_color = my_color if maximizing else opp_color
        other_color = opp_color if maximizing else my_color

        # 즉시 승리/방어
        win_move = self.find_winning_move(board, current_color, length)
        if win_move:
            return 1000000 + depth if maximizing else -1000000 - depth

        threat_move = self.find_winning_move(board, other_color, length)
        if threat_move:
            # 반드시 막아야 함
            x, y = threat_move
            board[x][y] = current_color
            self.update_hash(x, y, current_color)

            score = self.minimax(board, depth - 1, alpha, beta, not maximizing,
                               my_color, opp_color, length, (x, y), is_pv)

            self.update_hash(x, y, current_color)
            board[x][y] = 0
            return score

        # 후보 수 생성
        moves = self.generate_moves(board, current_color, other_color,
                                   length, max_moves=12)

        if not moves:
            return self.evaluate_board(board, my_color, opp_color, length)

        best_move = None
        original_alpha = alpha

        if maximizing:
            value = -INF
            for i, (x, y) in enumerate(moves):
                board[x][y] = my_color
                self.update_hash(x, y, my_color)

                # PVS: 첫 번째 수는 full window, 나머지는 null window
                if i == 0 or not is_pv:
                    score = self.minimax(board, depth - 1, alpha, beta, False,
                                        my_color, opp_color, length, (x, y), is_pv)
                else:
                    # Null window search
                    score = self.minimax(board, depth - 1, alpha, alpha + 1, False,
                                        my_color, opp_color, length, (x, y), False)
                    if alpha < score < beta:
                        # Re-search with full window
                        score = self.minimax(board, depth - 1, alpha, beta, False,
                                            my_color, opp_color, length, (x, y), True)

                self.update_hash(x, y, my_color)
                board[x][y] = 0

                if score > value:
                    value = score
                    best_move = (x, y)
                alpha = max(alpha, value)

                if alpha >= beta:
                    # Killer move 기록
                    if depth not in self.killer_moves:
                        self.killer_moves[depth] = []
                    if (x, y) not in self.killer_moves[depth]:
                        self.killer_moves[depth].insert(0, (x, y))
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth].pop()
                    # History 기록
                    self.history[(x, y)] = self.history.get((x, y), 0) + depth * depth
                    break
        else:
            value = INF
            for i, (x, y) in enumerate(moves):
                board[x][y] = opp_color
                self.update_hash(x, y, opp_color)

                if i == 0 or not is_pv:
                    score = self.minimax(board, depth - 1, alpha, beta, True,
                                        my_color, opp_color, length, (x, y), is_pv)
                else:
                    score = self.minimax(board, depth - 1, beta - 1, beta, True,
                                        my_color, opp_color, length, (x, y), False)
                    if alpha < score < beta:
                        score = self.minimax(board, depth - 1, alpha, beta, True,
                                            my_color, opp_color, length, (x, y), True)

                self.update_hash(x, y, opp_color)
                board[x][y] = 0

                if score < value:
                    value = score
                    best_move = (x, y)
                beta = min(beta, value)

                if alpha >= beta:
                    if depth not in self.killer_moves:
                        self.killer_moves[depth] = []
                    if (x, y) not in self.killer_moves[depth]:
                        self.killer_moves[depth].insert(0, (x, y))
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth].pop()
                    self.history[(x, y)] = self.history.get((x, y), 0) + depth * depth
                    break

        # Transposition Table 저장
        flag = 'exact'
        if value <= original_alpha:
            flag = 'upper'
        elif value >= beta:
            flag = 'lower'

        self.tt[board_key] = {
            'value': value,
            'depth': depth,
            'flag': flag,
            'best_move': best_move
        }

        return value

    # =========================================================================
    # Iterative Deepening
    # =========================================================================

    def iterative_deepening(self, board, my_color, opp_color, length, max_depth=4):
        """Iterative Deepening with Move Ordering"""

        # Zobrist 해시 초기화
        self.init_hash(board, length)

        best_move = None
        best_score = -INF

        root_moves = self.generate_moves(board, my_color, opp_color, length, max_moves=20)
        if not root_moves:
            c = length // 2
            return c, c

        if len(root_moves) == 1:
            return root_moves[0]

        for depth in range(1, max_depth + 1):
            # 시간 초과 체크
            if self.is_timeout():
                break

            current_best = None
            current_score = -INF
            scored_moves = []

            alpha = -INF
            beta = INF

            for i, (x, y) in enumerate(root_moves):
                # 시간 초과 체크
                if self.is_timeout():
                    break

                board[x][y] = my_color
                self.update_hash(x, y, my_color)

                # PVS
                if i == 0:
                    score = self.minimax(board, depth - 1, alpha, beta, False,
                                        my_color, opp_color, length, (x, y), True)
                else:
                    score = self.minimax(board, depth - 1, alpha, alpha + 1, False,
                                        my_color, opp_color, length, (x, y), False)
                    if score > alpha and not self.time_out:
                        score = self.minimax(board, depth - 1, alpha, beta, False,
                                            my_color, opp_color, length, (x, y), True)

                self.update_hash(x, y, my_color)
                board[x][y] = 0

                scored_moves.append((score, x, y))

                if score > current_score:
                    current_score = score
                    current_best = (x, y)

                alpha = max(alpha, score)

            # 시간 초과면 이전 결과 사용
            if self.time_out:
                break

            # Move ordering for next iteration
            scored_moves.sort(reverse=True, key=lambda t: t[0])
            root_moves = [(x, y) for _, x, y in scored_moves]

            best_move = current_best
            best_score = current_score

            # 승리 확정 시 조기 종료
            if best_score >= 900000:
                break

        return best_move if best_move else root_moves[0]

    # =========================================================================
    # 메인 인터페이스
    # =========================================================================

    def get_best_move(self, board, my_color):
        """최선의 수 반환"""
        # 시간 측정 시작
        self.start_time = time.time()
        self.time_out = False

        length = len(board)
        opp_color = -my_color

        # 1. 빈 보드 -> 중앙 (새 게임 시작, 캐시 초기화)
        has_stone = False
        for i in range(length):
            for j in range(length):
                if board[i][j] != 0:
                    has_stone = True
                    break
            if has_stone:
                break

        if not has_stone:
            self.clear_cache()  # 새 게임 시작 시 캐시 초기화
            c = length // 2
            return c, c

        # 캐시 관리 (너무 커지면 정리)
        if len(self.tt) > 100000:
            self.tt.clear()
            self.history.clear()

        # 2. 즉시 승리
        win = self.find_winning_move(board, my_color, length)
        if win:
            return win

        # 3. 즉시 방어
        block = self.find_winning_move(board, opp_color, length)
        if block:
            return block

        # 4. 필승 패턴 (44, 43)
        forcing = self.find_forcing_move(board, my_color, opp_color, length)
        if forcing:
            return forcing

        # 5. 상대 필승 패턴 방어
        opp_forcing = self.find_forcing_move(board, opp_color, my_color, length)
        if opp_forcing:
            return opp_forcing

        # 6. 상대 열린 4 방어 (열린 4는 막을 수 없으므로 미리 방어)
        opp_open_four = self.find_open_four(board, opp_color, length)
        if opp_open_four:
            return opp_open_four

        # 7. 내 열린 4 만들기
        my_open_four = self.find_open_four(board, my_color, length)
        if my_open_four:
            return my_open_four

        # 8. VCT 탐색 (위협 수순)
        vct_move = self.vct_attack(board, my_color, opp_color, length, 0, max_depth=6)
        if vct_move:
            return vct_move

        # 9. Iterative Deepening Search
        return self.iterative_deepening(board, my_color, opp_color, length, max_depth=4)
