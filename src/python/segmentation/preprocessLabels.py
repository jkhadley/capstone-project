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
import pickle
import cv2
import sys
#----------------------------------------------------------------------------
#                               Functions
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
#                                 Main
#----------------------------------------------------------------------------
def preprocessLabels():
    # change directory to directory with images
    os.chdir("./../../data/groundcover2016/")
    imageDir = os.listdir()
    # get directory to go back to 
    cwd = os.getcwd()

    i = imageDir[0] # going to be replaced with loop
    
    labelDir = "./" + i + "/label/"
    os.chdir(labelDir)
    
    saveDir = cwd + "/" + i + "/labeljpg/"

    # mkdir if it doesn't exist
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    # get list of the images
    labels = os.listdir()

    # Initialize dictionary to store the matrices
    labelDict = {}

    tmp = 0

    for j in labels:
        # load in image 
        img = cv2.imread(j,0)

        # get file name
        fname = j.replace(".tif","").replace("CE_NADIR_","")
        
        # convert to zeros and ones
        img[img > 0] = 1
        
        # put image in dictionary  
        # values inverted because as loaded in, 1 is ground and 0 is leaf
        img = np.invert(np.array(img,dtype = bool).flatten())

        labelDict[fname] = img 
        print(np.shape(img))        

        tmp+=1

    sys.exit()
    with open(saveDir + "/" + i + "Label.pkl" ,"wb") as f:
        pickle.dump(labelDict,f,protocol=pickle.HIGHEST_PROTOCOL)


    # go back to main image directory
    os.chdir(cwd)

#----------------------------------------------------------------------------
#                                 Run
#----------------------------------------------------------------------------
if __name__ == "__main__":
    preprocessLabels()

