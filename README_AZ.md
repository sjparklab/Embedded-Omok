```markdown
# 알파제로판 (19x19, 금수없음)

- alphazero_model.py : 경량 정책+가치 신경망 (PyTorch)
- mcts.py : PUCT 기반 MCTS (is_five 포함, 금수 규칙 없음)
- selfplay.py : MCTS-guided self-play으로 (state, pi, z) 수집
- train.py : 수집 데이터로 학습(간단 루프)
- iot12345_az.py : 기존 player 인터페이스(next)와 호환되는 AlphaZero 플레이어
- requirements.txt : 필수 라이브러리

훈련특수:
- self-play에서 게임 종료 후 z = +1(해당 플레이어 승), -1(패), 0(무승부)로 라벨링합니다.

사용법:
1) 의존성 설치:
   pip install -r requirements.txt

2) 자기대국 데이터 생성 (예: 20게임):
   python selfplay.py
   (net 랜덤 초기화 상태로 예제를 생성, 기존 학습 모델이 있을 경우 net을 로드하도록 코드 바꿔야함.)

3) 학습:
   python train.py

- 한번 돌려보고 gpu쓰게 쿠다 추가도 해볼 예정

바꿀것(gpt-5 피셜):
- 루트 노드에서 Dirichlet noise를 추가해 탐색 다양성 확보 (mcts.add_dirichlet_noise가 구현되어 있음, 필요 시 적용).
- MCTS 트리 재사용(성능 향상), 트랜스포지션 테이블(Zobrist) 추가.
- 더 많은 self-play 게임 + 반복적 학습 고리 (자기 대국 → 학습 → 모델 교체 → 반복).
- 병렬 시뮬레이션으로 속도 개선.
```