# -*- coding: utf-8 -*-
"""
오목 AI 승률 벤치마크 테스트
여러 게임을 병렬로 실행하여 승률을 측정합니다.

사용법:
  python benchmark.py [게임수] [모드] [seq|par]
  
모드:
  1 또는 normal : 흑=team05_ai, 백=minimax (기본값)
  2 또는 swap   : 흑=minimax, 백=team05_ai (AI 교체)
  3 또는 same1  : 흑=team05_ai, 백=team05_ai (같은 AI)
  4 또는 same2  : 흑=minimax, 백=minimax (같은 AI)
"""

import copy
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter

# AI 임포트
from iot6789_student import iot6789_student
from iot12345_student import iot12345_student

# 전역 설정 (멀티프로세싱에서 사용)
GAME_MODE = 1  # 1=normal, 2=swap, 3=same1, 4=same2

MODE_NAMES = {
    1: "Normal (Black=team05_ai, White=minimax)",
    2: "Swap (Black=minimax, White=team05_ai)",
    3: "Same AI (Black=team05_ai, White=team05_ai)",
    4: "Same AI (Black=minimax, White=minimax)",
}


class SilentOmokGame:
    """화면 출력 없이 빠르게 실행되는 오목 게임"""
    
    def __init__(self, size=19, mode=1):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]
        
        # 모드에 따라 AI 선택
        if mode == 1:  # Normal
            self.black = iot6789_student(-1)   # team05_ai
            self.white = iot12345_student(1)   # minimax
            self.black_name = "team05_ai"
            self.white_name = "minimax"
        elif mode == 2:  # Swap
            self.black = iot12345_student(-1)  # minimax
            self.white = iot6789_student(1)    # team05_ai
            self.black_name = "minimax"
            self.white_name = "team05_ai"
        elif mode == 3:  # Same (team05_ai)
            self.black = iot6789_student(-1)
            self.white = iot6789_student(1)
            self.black_name = "team05_ai"
            self.white_name = "team05_ai"
        elif mode == 4:  # Same (minimax)
            self.black = iot12345_student(-1)
            self.white = iot12345_student(1)
            self.black_name = "minimax"
            self.white_name = "minimax"
        else:
            self.black = iot6789_student(-1)
            self.white = iot12345_student(1)
            self.black_name = "team05_ai"
            self.white_name = "minimax"
        
        self.turns = 0
        self.next_player = -1  # 흑돌 선공
        self.winner = 0
        self.draw = False
    
    def play_game(self, max_turns=361, time_limit=5.0):
        """
        게임 한 판 실행
        :param max_turns: 최대 턴 수
        :param time_limit: 각 수에 대한 시간 제한 (초)
        :return: 승자 (-1: 흑, 1: 백, 0: 무승부)
        """
        while self.turns < max_turns:
            self.turns += 1
            
            # 현재 플레이어 결정
            if self.next_player == -1:
                player = self.black
            else:
                player = self.white
            
            # 수 두기 (시간 제한 없이 빠르게)
            try:
                board_copy = copy.deepcopy(self.board)
                stone = player.next(board_copy, self.size)
                x, y = stone.getX(), stone.getY()
                
                # 유효성 검사
                if self.is_valid_move(x, y):
                    self.board[x][y] = self.next_player
                else:
                    # 잘못된 수 - 턴 넘김 (3번까지 허용 후 패배 처리)
                    pass
                    
            except Exception as e:
                # AI 오류 시 턴 넘김
                pass
            
            # 승리 체크
            if self.check_win():
                self.winner = self.next_player
                return self.winner
            
            # 무승부 체크
            if self.check_draw():
                self.draw = True
                return 0
            
            # 턴 교체
            self.next_player *= -1
        
        # 최대 턴 초과 - 무승부
        return 0
    
    def is_valid_move(self, x, y):
        """유효한 수인지 확인"""
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        if self.board[x][y] != 0:
            return False
        return True
    
    def check_win(self):
        """5목 승리 체크"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    continue
                color = self.board[i][j]
                
                for dx, dy in directions:
                    count = 1
                    # 정방향
                    x, y = i + dx, j + dy
                    while 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == color:
                        count += 1
                        x += dx
                        y += dy
                    
                    if count >= 5:
                        return True
        return False
    
    def check_draw(self):
        """무승부 체크 (모든 칸이 찼는지)"""
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return False
        return True


def run_single_game(args):
    """단일 게임 실행 (멀티프로세싱용)"""
    if isinstance(args, tuple):
        game_id, mode = args
    else:
        game_id = args
        mode = GAME_MODE
    
    try:
        game = SilentOmokGame(19, mode=mode)
        winner = game.play_game()
        turns = game.turns
        return {
            'game_id': game_id,
            'winner': winner,
            'turns': turns,
            'error': None,
            'black_name': game.black_name,
            'white_name': game.white_name,
        }
    except Exception as e:
        return {
            'game_id': game_id,
            'winner': None,
            'turns': 0,
            'error': str(e),
            'black_name': 'unknown',
            'white_name': 'unknown',
        }


def run_benchmark(num_games=10, num_workers=None, mode=1):
    """
    벤치마크 실행
    :param num_games: 총 게임 수
    :param num_workers: 병렬 프로세스 수 (None이면 CPU 코어 수)
    :param mode: 게임 모드 (1=normal, 2=swap, 3=same1, 4=same2)
    """
    global GAME_MODE
    GAME_MODE = mode
    
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), num_games)
    
    # 모드에 따른 AI 이름
    if mode == 1:
        black_ai, white_ai = "team05_ai", "minimax"
    elif mode == 2:
        black_ai, white_ai = "minimax", "team05_ai"
    elif mode == 3:
        black_ai, white_ai = "team05_ai", "team05_ai"
    elif mode == 4:
        black_ai, white_ai = "minimax", "minimax"
    else:
        black_ai, white_ai = "team05_ai", "minimax"
    
    print("=" * 60)
    print("[OMOK AI Benchmark Test]")
    print("=" * 60)
    print("  Mode: %s" % MODE_NAMES.get(mode, "Unknown"))
    print("  Black (first): %s" % black_ai)
    print("  White (second): %s" % white_ai)
    print("  Total games: %d" % num_games)
    print("  Parallel workers: %d" % num_workers)
    print("=" * 60)
    print()
    
    results = []
    start_time = time.time()
    
    # 병렬 실행
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_single_game, (i, mode)): i for i in range(num_games)}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            # 진행 상황 출력
            winner_str = {-1: "Black", 1: "White", 0: "Draw", None: "Error"}.get(result['winner'], "?")
            print("  [%3d/%d] Game #%3d: %s wins (%d turns)" % (
                completed, num_games, result['game_id'], winner_str, result['turns']))
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # 결과 집계
    winners = [r['winner'] for r in results if r['winner'] is not None]
    counter = Counter(winners)
    
    black_wins = counter.get(-1, 0)
    white_wins = counter.get(1, 0)
    draws = counter.get(0, 0)
    errors = sum(1 for r in results if r['winner'] is None)
    
    total_valid = len(winners)
    avg_turns = sum(r['turns'] for r in results if r['winner'] is not None) / max(total_valid, 1)
    
    # 결과 출력
    print()
    print("=" * 60)
    print("[Result Summary]")
    print("=" * 60)
    print("  Total games: %d (valid: %d, errors: %d)" % (num_games, total_valid, errors))
    print("  Elapsed: %.1f sec (avg %.2f sec/game)" % (elapsed, elapsed/num_games))
    print("  Avg turns: %.1f" % avg_turns)
    print()
    print("  +------------------+--------+---------+")
    print("  |   Player (AI)    |  Wins  |  Rate   |")
    print("  +------------------+--------+---------+")
    print("  | Black %-10s | %4d   | %5.1f%%  |" % (black_ai, black_wins, black_wins/max(total_valid,1)*100))
    print("  | White %-10s | %4d   | %5.1f%%  |" % (white_ai, white_wins, white_wins/max(total_valid,1)*100))
    print("  | Draw             | %4d   | %5.1f%%  |" % (draws, draws/max(total_valid,1)*100))
    print("  +------------------+--------+---------+")
    print()
    
    # 승률 분석
    if total_valid > 0:
        if black_wins > white_wins:
            diff = (black_wins - white_wins) / total_valid * 100
            print("  >> Black (%s) is stronger by %.1f%%p!" % (black_ai, diff))
        elif white_wins > black_wins:
            diff = (white_wins - black_wins) / total_valid * 100
            print("  >> White (%s) is stronger by %.1f%%p!" % (white_ai, diff))
        else:
            print("  >> Both sides are balanced.")
    
    print("=" * 60)
    
    return {
        'black_wins': black_wins,
        'white_wins': white_wins,
        'draws': draws,
        'errors': errors,
        'avg_turns': avg_turns,
        'elapsed': elapsed
    }


def run_sequential_benchmark(num_games=10, mode=1):
    """
    순차 실행 벤치마크 (디버깅용)
    """
    global GAME_MODE
    GAME_MODE = mode
    
    # 모드에 따른 AI 이름
    if mode == 1:
        black_ai, white_ai = "team05_ai", "minimax"
    elif mode == 2:
        black_ai, white_ai = "minimax", "team05_ai"
    elif mode == 3:
        black_ai, white_ai = "team05_ai", "team05_ai"
    elif mode == 4:
        black_ai, white_ai = "minimax", "minimax"
    else:
        black_ai, white_ai = "team05_ai", "minimax"
    
    print("=" * 60)
    print("[OMOK AI Benchmark Test - Sequential]")
    print("=" * 60)
    print("  Mode: %s" % MODE_NAMES.get(mode, "Unknown"))
    print("  Black (first): %s" % black_ai)
    print("  White (second): %s" % white_ai)
    print("  Total games: %d" % num_games)
    print("=" * 60)
    print()
    
    results = []
    start_time = time.time()
    
    for i in range(num_games):
        result = run_single_game((i, mode))
        results.append(result)
        
        winner_str = {-1: "Black", 1: "White", 0: "Draw", None: "Error"}.get(result['winner'], "?")
        print("  [%3d/%d] Game #%3d: %s wins (%d turns)" % (
            i+1, num_games, i, winner_str, result['turns']))
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # 결과 집계
    winners = [r['winner'] for r in results if r['winner'] is not None]
    counter = Counter(winners)
    
    black_wins = counter.get(-1, 0)
    white_wins = counter.get(1, 0)
    draws = counter.get(0, 0)
    total_valid = len(winners)
    
    print()
    print("=" * 60)
    print("[Result Summary]")
    print("=" * 60)
    print("  Black (%s) wins: %d (%.1f%%)" % (black_ai, black_wins, black_wins/max(total_valid,1)*100))
    print("  White (%s) wins: %d (%.1f%%)" % (white_ai, white_wins, white_wins/max(total_valid,1)*100))
    print("  Draws: %d (%.1f%%)" % (draws, draws/max(total_valid,1)*100))
    print("  Elapsed: %.1f sec" % elapsed)
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import sys
    
    # 기본값
    num_games = 10
    mode = 1
    parallel = True
    
    # 명령줄 인자 처리
    args = sys.argv[1:]
    
    for arg in args:
        arg_lower = arg.lower()
        # 실행 모드
        if arg_lower == 'seq':
            parallel = False
        elif arg_lower == 'par':
            parallel = True
        # 게임 모드
        elif arg_lower in ['normal', '1']:
            mode = 1
        elif arg_lower in ['swap', '2']:
            mode = 2
        elif arg_lower in ['same1', '3']:
            mode = 3
        elif arg_lower in ['same2', '4']:
            mode = 4
        # 게임 수
        else:
            try:
                num_games = int(arg)
            except:
                pass
    
    print()
    print("=" * 60)
    print("Usage: python benchmark.py [num_games] [mode] [seq|par]")
    print("=" * 60)
    print("  num_games: Number of games (default: 10)")
    print()
    print("  mode:")
    print("    1 or normal : Black=team05_ai, White=minimax (default)")
    print("    2 or swap   : Black=minimax, White=team05_ai")
    print("    3 or same1  : Black=team05_ai, White=team05_ai")
    print("    4 or same2  : Black=minimax, White=minimax")
    print()
    print("  seq|par: Sequential or Parallel execution")
    print()
    print("Examples:")
    print("  python benchmark.py 20              # 20 games, normal mode")
    print("  python benchmark.py 10 swap         # 10 games, swap mode")
    print("  python benchmark.py 5 swap seq      # 5 games, swap, sequential")
    print("  python benchmark.py 10 same1        # same AI vs same AI")
    print("=" * 60)
    print()
    
    if parallel:
        run_benchmark(num_games, mode=mode)
    else:
        run_sequential_benchmark(num_games, mode=mode)
