# Author(s): Joseph Hadley
# Date Created: 2018-07-01
# Date Modified: 2018-07-17
# Description: Script to develop and test misc features

from multiprocessing import Pool
from skimage import io 
import matplotlib.pyplot as plt 
import numpy as np 
import os
import keras  

def savePropOfGround(src,writename):
    os.chdir(src)
    pwd = os.getcwd()

    f = open(writename,"w")
    f.write("imageName,propGround\n")

    img = os.listdir()

    for i in img:
        image = io.imread(pwd + "/"  + i)
        l,w = image.shape
        image[image >= 1] = 1
        
        tot = np.sum(image) 
        propGround = 1-tot/(l*w)

        f.write(i + "," + str(propGround) + "\n")

def getPropOfGround(directory,name):
    oldDir = os.getcwd()
    os.chdir(directory)
    pwd = os.getcwd()

    img = os.listdir()
    propList = []

    for i in img:
        image = io.imread(pwd + "/" + i)
        
        l,w = image.shape
        image[image >= 1] = 1
        
        tot = np.sum(image) 
        propGround = tot/(l*w)
        propList.append(1 - propGround)

    os.chdir(oldDir)
    #writeList()

    return propList

def writeList(l,where):
    f = open(where,"w")
    for i in l:
        f.write(str(i))
        f.write("\n")
    f.close()

def testPropOfGround():
    l = getPropOfGround("../../../data/groundcover2016/mungbean/label","tmp")

    writeList(l,"tmp.csv")
    plt.hist(l)
    plt.show()



