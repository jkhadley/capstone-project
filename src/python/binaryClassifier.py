# -*- coding: utf-8 -*-
"""
Author(s): Joseph Hadley
Date Created : 2018-06-06
Date Modified: 2018-06-06
Description: 
"""
import os
import numpy as np

# Change working directory to data path
currentPath = os.getcwd()
os.chdir("../../data/maize")
dataPath = os.getcwd()

# Invesitgate the number of pictures in each category
typesOfClasses = os.listdir()
numOfPics = []

for i in typesOfClasses:
    os.chdir(i)
    numOfPics.append(len(os.listdir()))
    
    if i != "healthy":
        print("AAAAAAAAAAAAAAAAAAAA")
    else:
        pass #do nothing 
        
    os.chdir("..")


undersampleSize = np.ceil(numOfPics[2]/(len(numOfPics) - 1))

#healthy = "healthy/" + os.listdir()

# Randomly choose 