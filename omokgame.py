# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:01:46 2020
Maker : bychoi@deu.ac.kr 

@author: Com
"""

import copy  #깊은 복사를 하기위한 모듈 ( modified by DS Koo)
from board import *
from player import *
from iot6789_student import *
from stone import *
from iot12345_student import *
#from iot6789_student import *
import time  # append by bychoi


class omokgame:
    def __init__(self, sz):
        self.__size=sz
        self.__bd = board(self.__size)
        #self.__black = player(-1)
        self.__black = iot6789_student(-1)
        self.__white = iot12345_student(1)
        self.__turns=0
        self.__next= -1
        self.__draw = 0
        self.__winner =0
        self.__bd.display()
    def __del__(self):
        #print("omok instanced is destroyed")
        pass
    
    def game_start(self):
        # do ~ while 
        while True:
            self.__turns +=1  # increment
            self.msg_display()
            if (self.__next == -1):
                print(" black Player: Turns = %5d" % self.__turns )
                time_b =0  # time ==> time_b
                time_delay =0;
                # do while 
                while True:
                    print(" black Player: time = %5d" % time_b )
                    start = time.time()
                    stn_b = self.__black.next(self.__bd.show(), self.__size)
                    end = time.time()
                    time_delay = end - start   # exec_time= next() method
                    time_b +=1 
                    if ((time_b >= 4) or (self.validCheck(stn_b) and (time_delay < 5)) ): # person: time_delay=100
                        break
                if (time_b < 4) :
                    self.__bd.update(stn_b)
                else:
                    print("Too many wrong input or long time, black's turn is over")
                self.__next = self.__next * (-1)
            elif (self.__next == 1) :
                print(" white Player: Turns = %5d" % self.__turns )
                time_w =0  # time ==> time_w
                # do while 
                while True:
                    print(" White Player: time = %5d" % time_w )
                    start1 = time.time()
                    stn_w = self.__white.next(self.__bd.show(), self.__size)
                    end1 = time.time()
                    time_delay1 = end1 - start1   # exec_time= next() method
                    time_w += 1
                    if ((time_w >= 4) or (self.validCheck(stn_w) and (time_delay1 < 5)) ): # person: time_delay=100
                        break
                if (time_w < 4):
                    self.__bd.update(stn_w)
                else:
                    print("Too many wrong input or long time, white's turn is over")
                self.__next = self.__next * (-1)
            if (self.endCheck()):  # do-while end
                break
        self.__winner = self.__next * (-1)
        self.msg_display()
        
    def msg_display(self):
        if (self.__turns !=0 and self.__winner ==0):
            print("Turn " , self.__turns, ", ", end="")
            if (self.__next == -1):
                print("Black")
            elif (self.__next == 1):
                print("White")
        if (self.__draw ==1):
            print()
            print("== No Winner : Game Result is draw ")
        elif (self.__winner !=0 ) :
            print()
            print("Congraturation!")
            print("The winner is ", end="") 
            if (self.__winner == -1):
                print("Black!!")
            elif (self.__winner == 1):
                print("White!!")
            
        
    def endCheck(self):
        # horizontal omok 
        for i  in range(0, self.__size):
            for j  in range(0, self.__size-4):
                if (self.__bd.get(i,j)!=0):
                    check=self.__bd.get(i,j)+self.__bd.get(i,j+1)+self.__bd.get(i,j+2)+self.__bd.get(i,j+3)+self.__bd.get(i,j+4)
                    if (check == (5 * self.__bd.get(i,j))):
                        return True
                    
        # vertical omok
        for i  in range(0, self.__size):
            for j  in range(0, self.__size-4):
                if (self.__bd.get(j,i)!=0):
                    check=self.__bd.get(j,i)+self.__bd.get(j+1,i)+self.__bd.get(j+2,i)+self.__bd.get(j+3,i)+self.__bd.get(j+4,i)
                    if (check == (5 * self.__bd.get(j,i))):
                        return True            
        
         
        #  diagonal 1        
        for i  in range(0, self.__size-4):
            for j  in range(0, self.__size-4):
                if (self.__bd.get(i,j)!=0):
                    check=self.__bd.get(i,j)+self.__bd.get(i+1,j+1)+self.__bd.get(i+2,j+2)+self.__bd.get(i+3,j+3)+self.__bd.get(i+4,j+4)
                    if (check == (5 * self.__bd.get(i,j))):
                        return True        
                
        #  diagonal 2          
        for i  in range(0, self.__size-4):
            for j  in range(4, self.__size):
                if (self.__bd.get(i,j)!=0):
                    check=self.__bd.get(i,j)+self.__bd.get(i+1,j-1)+self.__bd.get(i+2,j-2)+self.__bd.get(i+3,j-3)+self.__bd.get(i+4,j-4)
                    if (check == (5 * self.__bd.get(i,j))):
                        return True         
        # draw check : turns >= max 
        if (self.drawCheck()):
            return True
        # no match (no Omok)
            
        return False 

    def drawCheck(self):
        if (self.__turns >= self.__size * self.__size - 2):
            self.__draw=1
            return True
        else :
            self.__draw=0
            return False
        


    def validCheck(self, stn):
        # 3 by3 check =>  not implemented
        # overlapped check
        if (self.__bd.get(stn.getX(), stn.getY()) !=0):
            return False
        return True
    