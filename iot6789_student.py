# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:04:19 2020

Maker : bychoi@deu.ac.kr 

@author: Com
"""

# sample player file which must be made by student 

from player import *
from stone import *
from random import *

class iot6789_student(player):
     def __init__(self, clr):
          super().__init__( clr)  # call constructor of super class, self 제거
       
     
        
     def __del__(self):  # destructor
         pass 
     
     def next(self, board, length):  # override
         print (" **** Black player : My Turns **** ")
         stn = stone(self._color)  # protected variable 
         while True:
             x=randint(0,length-1) % length
             y=randint(0,length-1) % length
             if (board[x][y] ==0):
                 break
         stn.setX(x)
         stn.setY(y)
         print (" === Black player was completed ==== ")
         return stn
        
