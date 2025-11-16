# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:55:34 2020
Maker : bychoi@deu.ac.kr 
@author: Com
"""
from omokgame import *
import msvcrt


def main():
    #game =omokgame(15)
    game =omokgame(19)
    
    game.game_start()
    return 0

if __name__ =="__main__":
    main()
    #msvcrt.getch() # wait for key input 
    s = input() # wait for key input

