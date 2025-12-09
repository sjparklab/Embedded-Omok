from player import *
from stone import *
from team05_jdw import MinimaxEngine


class iot12345_student(player):
    def __init__(self, clr):
        super().__init__(clr)
        self.engine = MinimaxEngine()

    def __del__(self):
        pass

    def next(self, board, length):
        print(" **** White player : My Turns **** ")
        
        r, c = self.engine.get_best_move(board, self._color)
        
        stn = stone(self._color, length)
        stn.setX(r)
        stn.setY(c)
        
        print(" === White player was completed ==== ")
        return stn
