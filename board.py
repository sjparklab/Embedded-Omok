# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:04:16 2020
Maker : bychoi@deu.ac.kr 
@author: Com
"""
from stone import *
from os import system, name 
import copy  #깊은 복사를 하기위한 모듈 ( modified by DS Koo)

class board:
    def __init__(self, size=19):
        self.__size=size
        self.__game_board=[[0 for i in range(self.__size)] for j in range(self.__size)]
    def __del__(self):  # 소멸자 
        #print("board instance is destroyed")
        pass
    
    def update(self,st):
        x = st.getX()
        y = st.getY()
        stone = st.getStone()
        print(x,",", y,":",stone)
        self.__game_board[x][y]=stone
        self.display()
        
    def show(self) :
        temp = copy.deepcopy(self.__game_board)  # ( modified by DS Koo)
        return temp
    
    def get(self,x, y):
        return self.__game_board[x][y]
    
    def display(self) :
        # for windows 
        if name == 'nt':
            _ = system('cls') 
        # for mac and linux(here, os.name is 'posix') 
        else:
            _ = system('clear') 
        print("{0:^3}".format(" "), end="") # no newline 
        for i  in range(0, self.__size):
            print("{0:^3}".format(i), end="") # no newline
        print()  # new line 
        for i  in range(self.__size-1,-1, -1):
            print("{0:^3}".format(i), end="") # no newline
            for j  in range(0, self.__size):
               val=self.write_char(self.__game_board[i][j]) # no newline
               print("{0:^3}".format(val), end="") # no newline
            print()  # new line 
        
    def write_char(self,stn):
        if (stn == 1):
            return 'W'
        elif (stn == -1):
            return 'B'
        elif (stn == 0):
            return '.' # '+'
        else:
            return '.' # '+'
            
                
        
