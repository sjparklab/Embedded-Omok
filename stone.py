# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:14:57 2020
Maker : bychoi@deu.ac.kr 

@author: Com
"""

from board import *


class stone:
    def __init__(self):
        self.__size=19  # private
        self.__x = 9 
        self.__y = 9
        self.__bw=0
    
    
    def __init__(self, stn, sz=19):
        self.__size=sz  # private : variable with double underscore 
        self.__x = (self.__size-1)//2 # integer division
        self.__y = (self.__size-1)//2 # integer division
        self.__bw=stn
        
    def __del__(self):  # 소멸자
        #print(" stone 객체가 소멸합니다.") 
        pass

    def set(self, posX, posY, stn):
        self.__x = posX % self.__size
        self.__y = posX % self.__size
        self.__bw= stn
        
    def setStone(self, stn):
        self.__bw = stn
        
    def setX (self, posX):
        self.__x = posX % self.__size
        
    def setY (self, posY):
        self.__y = posY % self.__size
        
    def get(self):
        ret = stone()
        ret.set(self.__x, self.__y, self.__bw)
        return ret
    
    def getStone(self):
        return self.__bw
    
    def getX(self):
        return self.__x

    def getY(self):
        return self.__y
