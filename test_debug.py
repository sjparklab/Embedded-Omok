# test_debug.py
# 주의: 이 파일은 iot6789.py, team05_ai.py, player.py, stone.py가
# 모두 같은 폴더에 있을 때 실행해야 합니다.

try:
    from iot6789_student import iot6789_student # 학생 파일 import
    print("✅ 모듈 import 성공!")
except ImportError as e:
    print(f"❌ Import 에러: {e}")
    exit()

# 1. 가짜 보드 만들기 (19x19 빈 판)
dummy_board = [[0 for _ in range(19)] for _ in range(19)]

# 2. 상황 설정: 중앙에 흑돌 하나가 놓여있다고 가정
dummy_board[9][9] = -1 

print("🤖 AI 생성 중...")
# 3. AI 생성 (백돌로 설정)
ai_player = iot6789_student(1) 

print("🧠 생각 중 (Alpha-Beta + NNUE)...")
# 4. 수 요청 (돌 개수는 중요하지 않으니 0으로 넘김)
try:
    stone = ai_player.next(dummy_board, 19)
    print(f"🎉 성공! AI가 착수한 위치: ({stone.getX()}, {stone.getY()})")
    
    # 5. 검증: 범위 안에 잘 뒀는지?
    if 0 <= stone.getX() < 19 and 0 <= stone.getY() < 19:
        print("✅ 좌표 범위 정상")
    else:
        print("⚠️ 좌표 범위 이상 (0~18 사이여야 함)")

except Exception as e:
    print(f"🔥 실행 중 에러 발생: {e}")
    import traceback
    traceback.print_exc()