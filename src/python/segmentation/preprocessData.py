# Author(s): Joseph Hadley
# Date Created : 2018-06-13
# Date Modified: 2018-06-13
# Description: Script to convert labels to black and white 
#       This seems to mkae the images larger so will just load in the labels in as needed layer in their current format
#----------------------------------------------------------------------------
import os
import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import savemat
from PIL import Image
import pickle
import cv2
import sys
#----------------------------------------------------------------------------
#                               Functions
#----------------------------------------------------------------------------
def splitDataImagesOnSize(path):

    # change into data directory
    os.chdir(path + "/data")

    sizeDict = {}

    images = os.listdir()
    
    for img in images:
        im = Image.open(img)
        size = str(im.size)
        sizeDict[img] = str(size)
        im.close()
    
    # get the unique sizes
    uniqueSizes = np.unique(list(sizeDict.values()))

    # make train and test directories to move images to
    makeTrainAndTestDirs(uniqueSizes)

    # create a map to put the images of different sizes in different lists
    sizeMap = {}
    indexMap = {}

    for i in range(len(uniqueSizes)):
        print(str(uniqueSizes[i]))
        sizeMap[uniqueSizes[i]] = i
        indexMap[i] = uniqueSizes[i]

    # put images in list based on their size
    sizeList = [[] for i in range(len(uniqueSizes))]

    for img in images:
        whichSize = sizeDict[img]
        whichList = sizeMap[str(whichSize)]
        sizeList[whichList].append(img)

    # move images to new directory
    for i in range(len(uniqueSizes)):
        directory = str(indexMap[i])
        l = sizeList[i]

        # get random training and test indices
        trainInd = np.random.randint(0,len(l),size = round(0.8*len(l)))
        print("Train Index Length: " + len(trainInd))
        
        for j in range(len(l)):
            if j in trainInd:
                os.rename("./" + l[j],"./../" + directory + "/train/data/" + l[j])
            else:
                os.rename("./" + l[j],"./../" + directory + "/test/data/" + l[j])
   
    os.chdir("./..")
    # remove data dirctory if empty
    if not os.listdir("data"):
        os.rmdir("data")


def renameAndMoveLabels(path):

    # rename the labels    
    os.chdir("./label")
    labels = os.listdir()

    for i in range(len(labels)):
        os.rename("./" + labels[i],"./" + labels[i].replace(".tif",".jpg").replace("CE_NADIR_",""))
        #os.rename("./" + labels[i],"./" + labels[i] + ".jpg")

    labels = os.listdir()

    os.chdir("./..")

    sizes = os.listdir()

    for i in sizes:
        if i != "label":
            trainData = os.listdir("./" + i + "/train/data")
            testData = os.listdir("./" + i + "/test/data")
            print(trainData)
            for l in labels:
                if l in trainData:
                    os.rename("./label/" + l,"./" + i + "/train/label/" + l)
                elif l in testData:
                    os.rename("./label/" + l,"./" + i + "/test/label/" + l)

    # remove label dirctory if empty
    if not os.listdir("label"):
        os.rmdir("label")


def makeTrainAndTestDirs(uniqueDirs):
    cwd = os.getcwd()
    os.chdir("./..")
    # mkdir if it doesn't exist
    for directory in uniqueDirs:
        directory = str(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)    
            os.makedirs(directory + "/train")
            os.makedirs(directory + "/test")
            os.makedirs(directory + "/train/data")
            os.makedirs(directory + "/train/label")
            os.makedirs(directory + "/test/data")
            os.makedirs(directory + "/test/label")
    os.chdir(cwd)

#----------------------------------------------------------------------------
#                                 Main
#----------------------------------------------------------------------------
def preprocessData():
    cwd = os.getcwd()
    # change directory to directory with images
    os.chdir("./../../data/groundcover2016/maize")
    splitDataImagesOnSize("./")
    renameAndMoveLabels("./")
    # change back to initial directory
    os.chdir(cwd)

#----------------------------------------------------------------------------
#                                 Run
#----------------------------------------------------------------------------
if __name__ == "__main__":
    preprocessData()

