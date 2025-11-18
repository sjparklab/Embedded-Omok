# Embedded-Omok – White AI Player (`iot12345_student.py`)

## 1. 프로젝트 개요

이 프로젝트는 콘솔 기반 **오목 게임**이다.

- 보드 크기: `19 x 19`
- 실행 파일: `main.py`
- 게임 엔진: `omokgame.py`, `board.py`, `stone.py`, `player.py`
- **학생이 수정 가능한 파일 (과제 조건)**  
  - `iot6789_student.py` : 흑(Black) 플레이어용 클래스  
  - `iot12345_student.py` : 백(White) 플레이어용 클래스  

이번 구현에서의 목표:

1. **흑돌은 사람(사용자)이 직접 두기**
2. **백돌은 Minimax + 알파–베타 기반 AI로 자동으로 두기**
3. 외부 엔진 교체 없이 **두 학생 파일만 수정**해서 구현

---

## 2. 실행 및 플레이 방식

### 2.1 실행

```bash
python main.py
````

실행 시 `omokgame(19)`가 생성되고, 내부에서 다음과 같이 플레이어가 설정된다.

```python
self.__black = iot6789_student(-1)   # 흑
self.__white = iot12345_student(1)   # 백
```

### 2.2 게임 진행

* **흑(Black) – 사용자**

  `iot6789_student.next()` 가 호출될 때 내부에서 `player.next()` 를 그대로 사용한다.
  콘솔에 아래와 같은 입력창이 뜨고, 사용자가 좌표를 넣는다.

  ```text
  **** Black player : My Turns ****
  Input position x for new stone :
  Input position y for new stone :
  ```

* **백(White) – AI**

  `iot12345_student.next()` 안에서 **AI 로직이 실행**되고,
  사람이 아무 입력을 하지 않아도 백돌이 자동으로 최선의 수를 선택해 둔다.

---

## 3. 흑 플레이어 – `iot6789_student.py` (사용자 입력 래퍼)

원래 `iot6789_student.py`는 샘플 AI 코드였지만,
이번 구현에서는 **사람이 직접 흑돌을 두기 위한 래퍼 클래스**로 변경하였다.

```python
from player import *
from stone import *

class iot6789_student(player):
    def __init__(self, clr):
        super().__init__(clr)

    def __del__(self):
        pass

    def next(self, board, length):
        print(" **** Black player : My Turns **** ")
        # 부모 클래스(player)의 좌표 입력 로직을 그대로 사용
        return super().next(board, length)
```

* `player.next()` 에 이미

  * x, y 좌표를 입력받고,
  * 범위 체크 후,
  * `stone` 객체를 만들어 반환하는 콘솔 입력 로직이 구현돼 있다.
* 따라서 `iot6789_student` 는 **형식만 “학생용 클래스”일 뿐, 실제 동작은 사람 입력**이다.

> ※ 필요하면 축(가로/세로) 감각에 맞게 `stn.setX`, `stn.setY` 를 바꿔 끼우는 식으로
> 좌표계를 조정할 수 있다. (과제 조건 범위 안에서만 수정)

---

## 4. 백 플레이어 – `iot12345_student.py` (AI 오목)

`iot12345_student.py` 는 **실제 AI 오목 엔진**이 들어간 부분이다.
이번 구현은 다음 세 가지 아이디어를 기반으로 작성되었다.

1. **패턴 기반 평가 함수 (Pattern-based Evaluation)**
2. **후보 수 제한 (Move Candidate Pruning)**
3. **Minimax + Alpha-Beta Pruning (깊이 2)**

이 과정에서, 외부 오픈소스 프로젝트인
[five-in-a-row (StuartSul)](https://github.com/StuartSul/five-in-a-row) 를 **직접 포크하지 않고**,

* “줄 단위 패턴을 분석해서 점수화한다”
* “모든 칸을 탐색하지 않고 일부 유망한 후보만 본다”
* “Minimax/알파–베타로 게임 트리를 탐색한다”

라는 **알고리즘적 아이디어와 설계 방향만 참고**하였다.
실제 코드는 과제 조건에 맞게 `iot12345_student.py` 안에서 새로 작성하였다.

### 4.1. 패턴 기반 평가 함수

한 칸 `(x, y)`에 돌을 둔다고 가정했을 때:

* 가로 / 세로 / 두 대각선 방향 `(dx, dy)`에 대해
* 같은 색 돌이 얼마나 연속으로 이어져 있는지(`count`)
* 양쪽 끝이 비어 있는지(`open_ends`)를 계산한다.

```python
def score_direction(self, board, x, y, color, dx, dy, length):
    count = 1
    open_ends = 0
    ...
    if count >= 5:
        return 100000   # 승리(5목)
    if count == 4 and open_ends == 2:
        return 10000    # 열린 4
    if count == 4 and open_ends == 1:
        return 5000     # 막힌 4
    if count == 3 and open_ends == 2:
        return 1000     # 열린 3
    if count == 3 and open_ends == 1:
        return 200      # 막힌 3
    if count == 2 and open_ends == 2:
        return 100      # 열린 2
    if count == 2 and open_ends == 1:
        return 10       # 막힌 2
```

이렇게 해서 **공격/수비에서 중요한 패턴들(4, 열린 3, 열린 2 등)을 높은 점수로 평가**한다.
이 부분의 아이디어는 five-in-a-row에서 사용되는 “형태별 점수 매기기” 컨셉을 참고해 구현했다.

### 4.2. 한 수에 대한 공격 + 수비 평가

`evaluate_move()` 는 **해당 칸에 내가 두는 경우와 상대가 두는 경우를 모두 시뮬레이션**해서
공격과 수비를 동시에 고려한다.

```python
def evaluate_move(self, board, x, y, my_color, opp_color, length):
    original = board[x][y]

    # 내가 둘 때
    board[x][y] = my_color
    my_score = self.score_move(board, x, y, my_color, length)
    board[x][y] = original

    # 상대가 둘 때 (막아야 하는 위협)
    board[x][y] = opp_color
    opp_score = self.score_move(board, x, y, opp_color, length)
    board[x][y] = original

    return my_score * 1.5 + opp_score * 1.2
```

* `my_score`   : 내 돌을 두었을 때의 공격적 가치
* `opp_score`  : 상대가 두었을 때의 위협(수비 필요도)

five-in-a-row의 “공격/수비 모두 점수화해서 합산하는 방식”에서 아이디어를 가져와,
단순히 **내 형만 보는 AI가 아니라, 위협 수를 우선 차단할 수 있는 AI**가 되도록 했다.

### 4.3. 보드 전체 평가 함수

Minimax의 말단 노드(더 이상 깊이 안 내려갈 때)에서는
현재 보드를 한 번에 평가하는 함수가 필요하다.

```python
def evaluate_board(self, board, my_color, opp_color, length):
    total = 0
    for x in range(length):
        for y in range(length):
            if board[x][y] == my_color:
                total += self.score_move(board, x, y, my_color, length)
            elif board[x][y] == opp_color:
                total -= self.score_move(board, x, y, opp_color, length)
    return total
```

* 내 돌은 양수, 상대 돌은 음수로 누적해서
  `my_color` 기준 전체 스코어를 만든다.

---

## 5. 후보 수 줄이기 (Move Candidate Pruning)

19x19 전체 보드(361칸)를 매번 전부 탐색하면 연산량이 너무 커진다.
그래서 **현재 놓여 있는 돌 주변 일부만 “유망 후보”로 선정**하는 방식을 사용했다.

five-in-a-row 프로젝트에서도 비슷하게 “돌 주변 몇 칸만 후보로 본다”는 전략을 사용하고 있어,
그 아이디어를 참고하여 아래와 같이 구현했다.

```python
def generate_candidate_moves(self, board, length, color, my_color, opp_color, max_candidates=12):
    # 1) 돌이 하나도 없으면: 중앙 한 점
    # 2) 돌이 있는 최소/최대 x,y를 구해 bounding box 계산
    # 3) bounding box 주변 margin(2칸)을 확장한 영역만 탐색
    # 4) 빈 칸들에 대해 evaluate_move로 점수를 구하고,
    #    상위 max_candidates 개만 후보로 사용
```

* `max_candidates = 12` 로 설정하여
  **“좋아 보이는 10~12개의 수”만 Minimax 탐색 대상으로 사용**한다.
* 이렇게 하면:

  * **연산량은 크게 줄이면서도**,
  * “현재 돌 주변에서 의미 있는 수”는 대부분 고려할 수 있다.

---

## 6. Minimax + Alpha-Beta (깊이 2)

### 6.1. Minimax 구조

```python
def minimax(self, board, depth, alpha, beta,
            maximizing, my_color, opp_color, length,
            last_move=None, last_color=None):
    # 1. 직전 수(last_move)가 5목을 만들었는지 검사
    #    -> 이겼으면 +1_000_000, 졌으면 -1_000_000 반환

    # 2. depth == 0, 또는 둘 곳이 없으면 evaluate_board 반환

    # 3. maximizing 단계 (내 차례)
    #    - generate_candidate_moves()로 후보 생성
    #    - 각 후보에 대해 재귀 호출
    #    - value = max(value, child_score)
    #    - alpha / beta 갱신 및 가지치기

    # 4. minimizing 단계 (상대 차례)
    #    - 위와 동일하지만 min / beta 중심
```

* `depth = 2` 를 사용:

  * 루트(백 차례)에서

    * 백이 한 수 둔다 → (depth 1)
    * 흑이 응수한다 → (depth 0)
  * **“내 수 → 상대 수”까지 보는 2수 탐색** 구조

* **Alpha-Beta Pruning** 덕분에
  모든 후보를 끝까지 보지 않고도
  “더 볼 가치가 없는 가지”는 중간에 잘라낸다.

five-in-a-row 프로젝트도 Minimax/알파–베타를 사용하고 있으며,
이번 구현은 그 개념을 참고해서 **현재 과제 엔진 구조와 시간 제한(5초) 안에서 돌아가도록 단순화한 버전**이다.

### 6.2. `next()`에서의 실제 사용

`iot12345_student.next()` 의 흐름:

1. **초기 상태**

   * 보드에 돌이 하나도 없으면 → 중앙에 둔다.
2. **그 외 일반 상황**

   * `generate_candidate_moves(...)` 로 후보 좌표 리스트 생성
   * 각 후보 `(x, y)`에 대해:

     * 백이 `(x, y)`에 둔다고 가정하고 보드에 임시로 놓은 뒤
     * `minimax(..., depth=1, maximizing=False, last_move=(x, y))` 호출
   * Minimax 결과 점수 중 **최고 점수를 가진 후보들을 모두 모으고**,
     그 중 **랜덤으로 한 수를 최종 선택**한다.
3. 선택된 좌표를 `stone` 객체에 담아 반환:

   ```python
   stn.setX(bx)
   stn.setY(by)
   print(" === White player was completed ==== ")
   return stn
   ```

---

## 7. 난이도 조절 방법 (옵션)

AI 난이도를 바꾸고 싶을 때 수정하면 되는 포인트들:

1. **탐색 깊이**

   ```python
   depth = 2
   ```

   * `1`로 낮추면 → 단순 휴리스틱 그리디에 가까워져 **약해짐**
   * `3` 이상으로 올리면 → 강해지지만 Python 속도/5초 제한을 고려해야 함

2. **후보 수 개수**

   ```python
   max_candidates = 12
   ```

   * 6, 4 등으로 줄이면 → 탐색 폭이 좁아져 **실수도 많아지고 난이도 다운**

3. **공격/수비 가중치**

   ```python
   total = my_score * 1.5 + opp_score * 1.2
   ```

   * `my_score` 비율을 키우면 → 더 공격적인 AI
   * `opp_score` 비율을 키우면 → 위협 수를 더 적극적으로 막는 수비형 AI

---

## 8. 제한 사항 및 참고

* **3x3 금수, 장목 금지 규칙은 구현되어 있지 않다.**
  `omokgame.validCheck()`는 현재 “이미 돌이 있는 칸인지(겹쳐두기 여부)”만 검사한다.
* `omokgame`에서 한 턴당 `next()` 실행 시간이 **5초를 넘으면 해당 턴이 넘어가는 로직**이 있으므로,
  흑(사용자) 입력은 5초 이내에 하는 것이 안전하다.
* AI는 five-in-a-row 프로젝트에서 사용된 개념들(패턴 기반 평가, 후보 제한, Minimax/알파–베타)을 참고해
  **과제 엔진 구조와 파일 제한 안에서 새로 구현한 버전**이다.
  외부 코드를 직접 가져오거나, 엔진 전체를 교체하지 않았다.

---

```
::contentReference[oaicite:0]{index=0}
```
