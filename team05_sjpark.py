import random

# Constants for board dimensions
BOARD_SIZE = 19
TOTAL_CELLS = BOARD_SIZE * BOARD_SIZE  # 361

# Precompute masks for preventing bitboard wrap-around
# These masks ensure that a 5-stone pattern does not cross board edges when shifted.

# HORIZONTAL_MASK: Only allows patterns starting in columns 0-14.
HORZ_MASK = 0
for r in range(BOARD_SIZE):
    for c in range(BOARD_SIZE - 4): # Columns 0 to 14
        HORZ_MASK |= (1 << (r * BOARD_SIZE + c))

# VERTICAL_MASK: Only allows patterns starting in rows 0-14.
VERT_MASK = 0
for r in range(BOARD_SIZE - 4): # Rows 0 to 14
    for c in range(BOARD_SIZE):
        VERT_MASK |= (1 << (r * BOARD_SIZE + c))

# DIAG1_MASK (\ diagonal): Only allows patterns starting in rows 0-14 AND columns 0-14.
DIAG1_MASK = 0
for r in range(BOARD_SIZE - 4): # Rows 0 to 14
    for c in range(BOARD_SIZE - 4): # Columns 0 to 14
        DIAG1_MASK |= (1 << (r * BOARD_SIZE + c))

# DIAG2_MASK (/ diagonal): Only allows patterns starting in rows 0-14 AND columns 4-18.
DIAG2_MASK = 0
for r in range(BOARD_SIZE - 4): # Rows 0 to 14
    for c in range(4, BOARD_SIZE): # Columns 4 to 18
        DIAG2_MASK |= (1 << (r * BOARD_SIZE + c))

# Combine with directions for easier iteration
DIRECTION_MASKS = {
    1: HORZ_MASK,    # Horizontal
    19: VERT_MASK,   # Vertical
    18: DIAG1_MASK,  # Diagonal \
    20: DIAG2_MASK   # Diagonal /
}

# Masks for single columns/rows to prevent wrap-around for occupied shifts
COL_0_MASK = 0 # All bits in column 0
COL_18_MASK = 0 # All bits in column 18
ROW_0_MASK = 0 # All bits in row 0
ROW_18_MASK = 0 # All bits in row 18

for r in range(BOARD_SIZE):
    COL_0_MASK |= (1 << (r * BOARD_SIZE + 0))
    COL_18_MASK |= (1 << (r * BOARD_SIZE + 18))
for c in range(BOARD_SIZE):
    ROW_0_MASK |= (1 << (0 * BOARD_SIZE + c))
    ROW_18_MASK |= (1 << (18 * BOARD_SIZE + c))


# =============================================================================
# GomokuEngine (패턴 매칭 기반 AI)
# =============================================================================
class GomokuEngine:
    def __init__(self):
        # 더 이상 가중치 파일 로딩이 필요 없습니다!
        pass

    def parse_board_to_bitboard(self, board, my_color):
        """2차원 배열을 비트보드로 변환"""
        my_bb = 0
        opp_bb = 0
        for r in range(19):
            for c in range(19):
                stone = board[r][c]
                if stone == 0: continue
                idx = r * 19 + c
                if stone == my_color:
                    my_bb |= (1 << idx)
                else:
                    opp_bb |= (1 << idx)
        return my_bb, opp_bb

    def is_win(self, board):
        """5목 승리 판정 (4방향 체크)"""
        
        # Horizontal (d=1)
        b_masked = board & DIRECTION_MASKS[1]
        two = b_masked & (b_masked >> 1)
        three = two & (b_masked >> 2)
        four = three & (b_masked >> 3)
        if four & (b_masked >> 4):
            return True

        # Vertical (d=19)
        b_masked = board & DIRECTION_MASKS[19]
        two = b_masked & (b_masked >> 19)
        three = two & (b_masked >> 38)
        four = three & (b_masked >> 57)
        if four & (b_masked >> 76):
            return True

        # Diagonal \ (d=18)
        b_masked = board & DIRECTION_MASKS[18]
        two = b_masked & (b_masked >> 18)
        three = two & (b_masked >> 36)
        four = three & (b_masked >> 54)
        if four & (b_masked >> 72):
            return True

        # Diagonal / (d=20)
        b_masked = board & DIRECTION_MASKS[20]
        two = b_masked & (b_masked >> 20)
        three = two & (b_masked >> 40)
        four = three & (b_masked >> 60)
        if four & (b_masked >> 80):
            return True

        return False

    def evaluate(self, my_board, opp_board):
        """
        패턴 매칭을 통한 형세 판단
        (NNUE 대신 우리가 정한 점수표를 사용합니다)
        """
        score = 0
        occupied = my_board | opp_board
        
        # [점수표]
        SCORES = {
            'open_4': 10000,  # 승리 확정
            'open_3': 3000,   # 강력한 공격
            'closed_4': 1000, # 강제수 (상대 방어 유도)
            'closed_3': 100   # 견제
        }
        
        directions = [1, 19, 18, 20]
        
        for d in directions:
            mask = DIRECTION_MASKS[d]
            # --- 1. 내 돌 패턴 찾기 ---
            b_masked = my_board & mask
            two = b_masked & (b_masked >> d)
            three = two & (b_masked >> (2 * d))
            four = three & (b_masked >> (3 * d))
            
            # A. 열린 4 / 닫힌 4 구분
            # four는 현재 4개가 연속된 상태.
            # Determine valid positions for checking empty spots for a 4-stone pattern
            # left_empty checks position `pos - d`
            # right_empty checks position `pos + 4d`

            # Calculate actual bits for left/right empty, respecting boundaries
            # Temporary occupied for checking 'left' empty (position `pos - d`)
            temp_occ_left_4 = occupied << d
            if d == 1: # Horizontal
                temp_occ_left_4 &= ~COL_0_MASK
            elif d == 18: # Diagonal \
                temp_occ_left_4 &= ~(COL_0_MASK | ROW_0_MASK)
            elif d == 19: # Vertical
                temp_occ_left_4 &= ~ROW_0_MASK
            elif d == 20: # Diagonal /
                temp_occ_left_4 &= ~(COL_18_MASK | ROW_0_MASK)

            # Temporary occupied for checking 'right' empty (position `pos + 4d`)
            temp_occ_right_4 = occupied >> (4 * d)
            if d == 1: # Horizontal
                temp_occ_right_4 &= ~COL_18_MASK
            elif d == 18: # Diagonal \
                temp_occ_right_4 &= ~(COL_18_MASK | ROW_18_MASK)
            elif d == 19: # Vertical
                temp_occ_right_4 &= ~ROW_18_MASK
            elif d == 20: # Diagonal /
                temp_occ_right_4 &= ~(COL_0_MASK | ROW_18_MASK)

            left_empty_for_four = (~temp_occ_left_4)
            right_empty_for_four = (~temp_occ_right_4)
            
            valid_open_4 = four & left_empty_for_four & right_empty_for_four
            valid_closed_4 = four & (left_empty_for_four | right_empty_for_four) # at least one side open
            valid_closed_4 &= (~valid_open_4) # Remove those that are fully open, to count only partially open

            score += bin(valid_open_4).count('1') * SCORES['open_4']
            score += bin(valid_closed_4).count('1') * SCORES['closed_4']
            
            # B. 열린 3 ( . ● ● ● . )
            # 3개 연속(three) 이면서 양쪽이 비어있어야 함
            # Determine valid positions for checking empty spots for a 3-stone pattern
            # left_empty checks position `pos - d`
            # right_empty checks position `pos + 3d`

            temp_occ_left_3 = occupied << d
            if d == 1: # Horizontal
                temp_occ_left_3 &= ~COL_0_MASK
            elif d == 18: # Diagonal \
                temp_occ_left_3 &= ~(COL_0_MASK | ROW_0_MASK)
            elif d == 19: # Vertical
                temp_occ_left_3 &= ~ROW_0_MASK
            elif d == 20: # Diagonal /
                temp_occ_left_3 &= ~(COL_18_MASK | ROW_0_MASK)

            temp_occ_right_3 = occupied >> (3 * d)
            if d == 1: # Horizontal
                temp_occ_right_3 &= ~COL_18_MASK
            elif d == 18: # Diagonal \
                temp_occ_right_3 &= ~(COL_18_MASK | ROW_18_MASK)
            elif d == 19: # Vertical
                temp_occ_right_3 &= ~ROW_18_MASK
            elif d == 20: # Diagonal /
                temp_occ_right_3 &= ~(COL_0_MASK | ROW_18_MASK)

            left_empty_for_three = (~temp_occ_left_3)
            right_empty_for_three = (~temp_occ_right_3)

            valid_open_3 = three & left_empty_for_three & right_empty_for_three
            score += bin(valid_open_3).count('1') * SCORES['open_3']

        return score

    def get_heuristic_score(self, my_board, opp_board):
        # 내 공격 점수 - 상대 공격 점수 (방어)
        return self.evaluate(my_board, opp_board) - self.evaluate(opp_board, my_board) * 1.5
        # *1.5를 하는 이유: 내 공격보다 수비(상대 공격 막기)를 더 우선시하도록!

    def generate_moves(self, my_board, opp_board):
        """후보 수 생성 (주변 1칸)"""
        occupied = my_board | opp_board
        if occupied == 0: return [180] # 중앙(9,9)
        
        # 8방향 확장 (Dilation)
        empty = ~occupied & ((1 << 361) - 1)
        dirs = [1, 19, 18, 20]
        neighbor_mask = 0
        for d in dirs:
            neighbor_mask |= (occupied << d)
            neighbor_mask |= (occupied >> d)
        neighbor_mask &= ((1 << 361) - 1)
        
        candidates_bb = neighbor_mask & empty
        
        moves = []
        while candidates_bb:
            lsb = candidates_bb & -candidates_bb
            moves.append(lsb.bit_length() - 1)
            candidates_bb ^= lsb
        return moves

    def alpha_beta(self, depth, alpha, beta, my_board, opp_board):
        # 1. 상대 승리 확인 (내가 짐)
        if self.is_win(opp_board): return -50000
            
        # 2. 깊이 제한 -> 패턴 매칭 점수 반환
        if depth == 0:
            return self.get_heuristic_score(my_board, opp_board)
            
        moves = self.generate_moves(my_board, opp_board)
        
        # 간단한 정렬 (중앙에 가까울수록 가점 주는 등)은 생략되었으나,
        # 여기서 evaluate 함수를 이용해 정렬하면 훨씬 빨라집니다.
        
        for move_idx in moves:
            move_mask = (1 << move_idx)
            new_my_board = my_board | move_mask
            
            # 재귀 호출
            score = -self.alpha_beta(depth - 1, -beta, -alpha, opp_board, new_my_board)
            
            if score >= beta: return beta
            if score > alpha: alpha = score
                
        return alpha

    def search_root(self, my_bb, opp_bb):
        moves = self.generate_moves(my_bb, opp_bb)
        
        best_move = -1
        alpha = -9999999
        beta = 9999999
        depth = 5 # 속도에 따라 3~4로 늘려보세요!
        
        # Move Ordering을 위해 점수 미리 계산해보기 (선택 사항)
        # moves.sort(key=lambda m: ...) 
        
        for move_idx in moves:
            move_mask = (1 << move_idx)
            new_my_bb = my_bb | move_mask
            
            score = -self.alpha_beta(depth - 1, -beta, -alpha, opp_bb, new_my_bb)
            
            if score > alpha:
                alpha = score
                best_move = move_idx
                
        return best_move

    def get_best_move(self, board, my_color):
        """외부(Player)에서 호출하는 인터페이스"""
        # 1. 입력 변환
        my_bb, opp_bb = self.parse_board_to_bitboard(board, my_color)
        
        # ---------------------------------------------------------------------
        # [★ 중요] VCF 승리 탐색 (필살기)
        # 15수 정도 깊이로 "연속 공격 승리"가 있는지 먼저 봅니다.
        # ---------------------------------------------------------------------
        vcf_move = self.solve_vcf(my_bb, opp_bb, depth=15)
        if vcf_move != -1:
            return vcf_move // 19, vcf_move % 19
            
        # ---------------------------------------------------------------------
        # [방어] 상대방의 VCF 공격 막기
        # (상대 입장에서 VCF가 되는지 보고, 된다면 그 첫 수를 내가 막아야 함)
        # ---------------------------------------------------------------------
        threat_move = self.solve_vcf(opp_bb, my_bb, depth=7) # 방어는 좀 얕게
        if threat_move != -1:
            return threat_move // 19, threat_move % 19

        # 2. 일반 탐색 (Alpha-Beta)
        # VCF가 없으면 원래대로 차분하게 탐색
        best_move_index = self.search_root(my_bb, opp_bb)
        
        # 3. 출력 변환
        if best_move_index == -1:
            return 9, 9
            
        return best_move_index // 19, best_move_index % 19
    
    # =========================================================================
    # [추가] VCF (Victory by Continuous Four) 로직
    # =========================================================================
    
    def get_winning_spot(self, board):
        """
        현재 board에 '4목'이 있어서, 다음 수에 5목이 되는 자리(급소)를 하나 찾아서 반환.
        (상대방이 4를 쳤을 때 막아야 할 자리를 찾을 때 사용)
        """
        # 모든 빈칸을 다 뒤지는 건 느리니, 돌 주변만 검사
        # (더 빠르게 하려면 비트 연산으로 4목 패턴을 찾아야 하지만, 
        #  코드 복잡도를 줄이기 위해 시뮬레이션 방식을 씁니다.)
        candidates = self.generate_moves(board, 0) # 상대 돌은 0으로 처리(내 돌 기준)
        
        for move in candidates:
            # 이 자리에 뒀을 때 5목이 되는가?
            if self.is_win(board | (1 << move)):
                return move
        return -1

    def solve_vcf(self, my_bb, opp_bb, depth, path=[]):
        """
        VCF 탐색: 연속으로 4를 둬서 이기는 길이 있는지 찾는다.
        :return: 이기는 첫 번째 수의 인덱스 (없으면 -1)
        """
        if depth == 0: return -1
        
        # 1. 나의 후보 수 생성 (주변 빈칸들)
        moves = self.generate_moves(my_bb, opp_bb)
        
        for move in moves:
            move_mask = (1 << move)
            
            # [시뮬레이션] 내가 둠
            next_my_bb = my_bb | move_mask
            
            # A. 바로 승리 (5목) -> 찾았다!
            if self.is_win(next_my_bb):
                return move
            
            # B. 4목(공격)인지 확인
            # 상대방 입장에서 봤을 때 "막아야 할 자리"가 생겼는지 확인
            # 내가 뒀으니(next_my_bb), 이 돌들이 만드는 위협(급소)을 찾음
            defense_point = self.get_winning_spot(next_my_bb)
            
            if defense_point != -1:
                # 상대는 무조건 defense_point를 막아야 함 (강제수)
                # 만약 defense_point가 이미 내 돌이나 상대 돌로 차있다면? (그럴 리 없지만)
                # 문제는, 4-3 같은 게 터져서 급소가 2개 이상이면? -> 그건 '열린 4'이므로 이미 승리!
                
                # 열린 4 체크: 급소가 2개 이상이면 무조건 승리
                # (간단히 하기 위해: 내가 뒀는데 급소가 생겼다면 공격 성공으로 간주하고 진행)
                
                # 상대가 막음 (강제 방어)
                next_opp_bb = opp_bb | (1 << defense_point)
                
                # [재귀 호출] 계속 공격!
                # 상대가 막은 상태에서 다시 VCF를 찾음
                res = self.solve_vcf(next_my_bb, next_opp_bb, depth - 1, path + [move])
                
                if res != -1:
                    # 재귀에서 승리 길을 찾았다면, 지금 둔 수(move)가 정답!
                    return move
                    
        return -1
