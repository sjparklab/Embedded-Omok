# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:37:40 2020
Maker : bychoi@deu.ac.kr 

@author: Com
"""

# sample player file which must be made by student 

from player import *
from stone import *
from random import *

class iot12345_student(player):
     def __init__(self, clr):
          super().__init__( clr)  # call constructor of super class
       
     
        
     def __del__(self):  # destructor
         pass 
     
     def next(self, board, length):  # override
         print (" **** White player : My Turns **** ")
         stn = stone(self._color)  # protected variable 
         while True:
             x=randint(0,length-1) % length
             y=randint(0,length-1) % length
             if (board[x][y] ==0):
                 break
         stn.setX(x)
         stn.setY(y)
         print (" === White player was completed ==== ")
         return stn
        
    
    
        
    
