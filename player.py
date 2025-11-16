# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:19:19 2020
Maker : bychoi@deu.ac.kr 

@author: Com
"""


from stone import *

class player:
    def __init__(self, clr):
        self._color = clr   # proteced variable with single underscore
        
    def __del__(self):
        pass
    
    def next(self, board, length):
        print (" **** Black player : My Turns **** ")
        stn = stone(self._color, length)
        pos =int(input("Input position x for new stone : "))
        while ((pos < 0) or (pos >= length)):
            pos = input("Wrong position, please input again : ")
            
        stn.setX(pos)
        pos =int(input("Input position y for new stone : "))
        while ((pos < 0) or (pos >= length)):
            pos = input("Wrong position, please input again : ")
            
        stn.setY(pos)
        print (" === Black player was completed ==== ")
        return stn