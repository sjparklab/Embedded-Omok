# -*- coding: utf-8 -*-
from player import *
from stone import *
import subprocess
import os
import time
import threading


class katagomo_ai_player(player):
    # 코디네이트 변환용 상수
    _BOARD_SIZE = 19
    _GTP_COL_LETTERS = 'ABCDEFGHJKLMNOPQRST'  # GTP 좌표 문자
    _MODEL_LOAD_TIMEOUT = 30  # seconds
    _MODEL_LOAD_CHECK_INTERVAL = 1  # seconds
    
    def __init__(self, clr):
        super().__init__(clr)
        self._engine = None
        self._stderr_thread = None
        self._start_engine()
    
    def __del__(self):
        self._stop_engine()
    
    def _start_engine(self):
        """KataGomo 엔진을 GTP 프로토콜을 통해 시작"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            katago_path = os.path.join(script_dir, 'engine/gom20x_trt.exe')
            model_path = os.path.join(script_dir, 'tworule_20x_b18.bin')
            config_path = os.path.join(script_dir, 'gomoku_gtp.cfg')
            
            if not os.path.exists(katago_path):
                raise FileNotFoundError(f"KataGomo 실행 파일을 찾을 수 없습니다: {katago_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
            
            self._engine = subprocess.Popen(
                [katago_path, 'gtp', '-model', model_path, '-config', config_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            self._start_stderr_reader()
            
            self._wait_for_ready()
            # 엔진이 중간에 죽었는지 다시 한 번 확인
            if self._engine.poll() is not None:
                stderr_msg = ''
                try:
                    if self._engine.stderr:
                        stderr_msg = self._engine.stderr.read().strip()
                except Exception:
                    pass
                raise RuntimeError(f"엔진이 시작 후 종료되었습니다. STDERR: {stderr_msg}")

            # 보드 사이즈/코미 설정이 실패하면 엔진 사용을 포기하고 랜덤으로 동작
            komi_resp = self._send_command('komi 7.5')
            if komi_resp is None:
                raise RuntimeError("시작 후 코미 설정에 실패했습니다")

            init_resp = self._send_command('boardsize 19')
            if init_resp is None:
                raise RuntimeError("시작 후 보드 크기 설정에 실패했습니다")
            
            print(f" === AI Engine은 {'Black' if self._color == -1 else 'White'}으로 시작합니다=== ")
        except Exception as e:
            print(f" === 엔진 시작 중 오류 발생: {e} === ")
            self._engine = None
            raise
    
    def _stop_engine(self):
        """엔진 중지"""
        if self._engine:
            try:
                # 엔진 종료 시에는 GTP "quit" 명령을 굳이 보내지 않고
                # 프로세스를 직접 종료하여 파이프 관련 에러([Errno 22])를 방지한다.
                if self._engine.poll() is None:
                    try:
                        self._engine.terminate()
                    except Exception:
                        pass
                    try:
                        self._engine.wait(timeout=3)
                    except Exception:
                        pass
            except:
                self._engine.kill()
            self._engine = None
            self._stderr_thread = None
    
    def _wait_for_ready(self):
        """엔진이 준비될 때까지 대기"""
        print(" === 엔진이 모델을 로드하는 중... === ")
        start_time = time.time()
        time.sleep(3)
        
        while time.time() - start_time < self._MODEL_LOAD_TIMEOUT:
            try:
                if self._engine.poll() is not None:
                    raise RuntimeError("엔진이 시작 중에 종료되었습니다")
                time.sleep(self._MODEL_LOAD_CHECK_INTERVAL)
                if time.time() - start_time >= 10:
                    break
            except Exception as e:
                if time.time() - start_time >= self._MODEL_LOAD_TIMEOUT:
                    raise RuntimeError(f"엔진 대기 시간 초과: {e}")
                time.sleep(self._MODEL_LOAD_CHECK_INTERVAL)
        
        print(" === 엔진이 준비된 것으로 보입니다 === ")
    
    def _send_command(self, command):
        """엔진에 GTP 명령을 보내고 응답을 받음"""
        # 엔진 프로세스가 없거나 이미 종료되었으면 통신을 시도하지 않는다.
        if not self._engine or self._engine.poll() is not None:
            print(" === 엔진이 실행 중이 아니므로 명령을 건너뜁니다 === ")
            return None
        
        try:
            self._engine.stdin.write(command + '\n')
            self._engine.stdin.flush()
            print(f"명령 전송: {command}")
            
            response_lines = []
            while True:
                line = self._engine.stdout.readline()
                if line == '':  # EOF
                    print(" === 엔진이 stdout을 닫았습니다 === ")
                    break
                line = line.strip()
                print(f"수신된 라인: {line}")
                if not line:
                    continue
                if line.startswith('='):
                    if len(line) > 1:
                        response_lines.append(line[2:])
                    while True:
                        line = self._engine.stdout.readline().strip()
                        if not line:
                            break
                        response_lines.append(line)
                    break
                elif line.startswith('?'):
                    print(f" === 엔진 오류: {line} === ")
                    break
            
            return '\n'.join(response_lines).strip()
        except Exception as e:
            print(f" === 엔진과 통신 중 오류 발생: {e} === ")
            return None
    
    def _convert_coords_to_gtp(self, x, y):
        """보드 좌표(0-18)를 GTP 형식(예: 'J10')으로 변환"""
        if not (0 <= x < self._BOARD_SIZE and 0 <= y < self._BOARD_SIZE):
            raise ValueError(f"코디네이트 인자 범위 초과: ({x}, {y})")
        
        col = self._GTP_COL_LETTERS[y]
        row = x + 1
        return f"{col}{row}"
    
    def _convert_gtp_to_coords(self, gtp_move):
        """GTP 형식의 수(예: 'J10')를 보드 좌표로 변환"""
        if not gtp_move or len(gtp_move) < 2:
            return None, None
        
        col_char = gtp_move[0].upper()
        if col_char not in self._GTP_COL_LETTERS:
            return None, None
        
        try:
            y = self._GTP_COL_LETTERS.index(col_char)
            row = int(gtp_move[1:])
            if not (1 <= row <= self._BOARD_SIZE):
                return None, None
            x = row - 1
            if not (0 <= x < self._BOARD_SIZE and 0 <= y < self._BOARD_SIZE):
                return None, None
            return x, y
        except (ValueError, IndexError):
            return None, None
    
    def next(self, board, length):
        """Katagomo AI를 사용하여 다음 수를 생성"""
        player_name = "Black" if self._color == -1 else "White"
        print(f" **** {player_name} player (AI): My Turns **** ")
        
        stn = stone(self._color, length)
        
        if not self._engine:
            raise RuntimeError("엔진이 실행 중이 아닙니다")
        
        color_gtp = 'B' if self._color == -1 else 'W'
        while True:
            response = self._send_command(f'genmove_analyze {color_gtp} 10')
            if not response:
                raise RuntimeError("엔진이 수를 반환하지 않았습니다")

            move = self._parse_move_from_response(response)
            if not move:
                raise RuntimeError("엔진 응답에 수가 포함되어 있지 않습니다")

            if move == 'RESIGN':
                print(" === 엔진이 RESIGN을 반환했습니다; 수 생성 재시도 중 === ")
                continue
            if move == 'PASS':
                raise RuntimeError("엔진이 PASS를 반환했습니다; 오목에서는 예상치 못한 동작입니다")
            break

        x, y = self._convert_gtp_to_coords(move)
        if x is None or y is None:
            raise RuntimeError(f"잘못된 형식 '{move}'")
        if not (0 <= x < length and 0 <= y < length):
            raise RuntimeError(f"엔진 '{move}' 좌표가 범위를 벗어남")
        if board[x][y] != 0:
            raise RuntimeError(f"엔진 '{move}'이(가) 이미 점유된 위치를 가리킴")

        stn.setX(x)
        stn.setY(y)
        
        print(f" === {player_name} player (AI) completed ==== ")
        return stn
    
    def update_opponent_move(self, x, y):
        """상대방의 수를 엔진에 업데이트"""
        if not self._engine:
            return
        
        gtp_move = self._convert_coords_to_gtp(x, y)
        opponent_color = 'W' if self._color == -1 else 'B'
        self._send_command(f'play {opponent_color} {gtp_move}')

    def _parse_move_from_response(self, response):
        lines = response.splitlines()
        for line in reversed(lines):
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0].lower() == 'play' and len(tokens) >= 2:
                return tokens[1].upper()
            for token in reversed(tokens):
                candidate = token.upper()
                if candidate in ('PASS', 'RESIGN'):
                    return candidate
                if self._is_valid_gtp_coordinate(candidate):
                    return candidate
        return None

    def _is_valid_gtp_coordinate(self, token):
        if len(token) < 2:
            return False
        col_char = token[0]
        if col_char not in self._GTP_COL_LETTERS:
            return False
        try:
            row = int(token[1:])
        except ValueError:
            return False
        return 1 <= row <= self._BOARD_SIZE

    def _start_stderr_reader(self):
        if not self._engine or not self._engine.stderr:
            return

        def _drain():
            try:
                for line in self._engine.stderr:
                    if not line:
                        break
                    stripped = line.strip()
                    if stripped:
                        print(f"[STDERR] {stripped}")
            except Exception:
                pass

        self._stderr_thread = threading.Thread(target=_drain, daemon=True)
        self._stderr_thread.start()
