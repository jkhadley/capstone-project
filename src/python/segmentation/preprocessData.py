# Author(s): Joseph Hadley
# Date Created : 2018-06-13
# Date Modified: 2018-07-11
# Description: Script to put images into train, test, and validate categories and also 
#           split the images themselves into small 256x256 chucks with the exception of 
#           the test data. These images are placed into subdirectories for each category 
#           to avoid having massive directories. Now uses multiprocessing to split images.
#----------------------------------------------------------------------------
import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import numpy as np 
import random
from shutil import copy
from skimage import io
from shutil import copy
from multiprocessing import Pool
#----------------------------------------------------------------------------
#                               Functions
#----------------------------------------------------------------------------
def renameLabels():
    oldDir = os.getcwd()
    os.chdir("../../../data/groundcover2016/")
    
    skipdir = ['train','test','validate','maize']

    pwd = os.getcwd()
    dirs = os.listdir()

    for i in dirs:
        if i in skipdir:
            pass
        else:
            d = pwd + "/" + i + "/label/"
            labels = os.listdir(d)
            for j in range(len(labels)):
                os.rename(d + labels[j],d + labels[j].replace(".tif",".jpg").replace("CE_NADIR_",""))
    os.chdir(oldDir)

def splitImagesIntoDirectories():
    oldDir = os.getcwd()
    os.chdir("../../../data/groundcover2016/")
    
    skipdir = ['train','test','validate']

    pwd = os.getcwd()
    dirs = os.listdir()

    for i in dirs:
        if i in skipdir:
            pass
        else:
            d = pwd + "/" + i 
            names = os.listdir(d + '/data/')
            trainInd,testInd,valInd = randomSplit(len(names))

            for j in range(len(names)):
                if j in trainInd:
                    copy(d + "/data/" + names[j],pwd + '/train/' + i + '/data/' + names[j])
                    copy(d + "/label/" + names[j],pwd + '/train/' + i + '/label/'+ names[j])
                elif j in valInd:
                    copy(d + "/data/" + names[j],pwd + '/validate/' + i + '/data/'+ names[j])
                    copy(d + "/label/" + names[j],pwd + '/validate/' + i + '/label/'+ names[j])
                elif j in testInd:
                    copy(d + "/data/" + names[j],pwd + '/test/' + i + '/data/'+ names[j])
                    copy(d + "/label/" + names[j],pwd + '/test/' + i + '/label/'+ names[j])
        print(i + "done")
    os.chdir(oldDir)

def randomSplit(l):
    ind = [i for i in range(l)]
    random.shuffle(ind)
    b1 = round(0.8*len(ind))
    b2 = round(0.9*len(ind))
    trainInd = ind[:b1]
    valInd = ind[b1:b2]
    testInd = ind[b2:]
    return trainInd,testInd,valInd

def makeSplitDirs():
    oldDir = os.getcwd()
    os.chdir("../../../data/groundcover2016/")
    pwd = os.getcwd()
    # mkdir if it doesn't exist
    l1 = ['train','test','validate']
    l2 = ['wheat','maize','maizevariety','mungbean']
    l3 = ['data','label']

    for i in l1:
        os.mkdir(pwd + "/" + i)
        for j in l2:
            os.mkdir(pwd + "/" + i + '/' + j)
            for k in l3:
                os.mkdir(pwd + "/" + i + '/' + j + "/" + k)
    os.chdir(oldDir)
        # go through train, test, and validate directories

def splitImageMP(params):
    shape = params['shape']

    os.chdir("../../../data/groundcover2016/")
    pwd = os.getcwd()

    d = pwd + '/' + params['baseDir'] + '/' + params['classDir']
    images = os.listdir(d +'/data')
    subDirCount = 0
    fileCnt = 0
    subdir = "sub0/"

    # make new directories
    os.mkdir(d + "/data/" + subdir)
    os.mkdir(d + "/label/" + subdir)
    for img in images:
        # load image and label
        data = io.imread(d + '/data/' + img)
        label = io.imread(d + '/label/' + img)
        # extract the name
        name = img.replace(".jpg","").replace(".jpeg","")

        # get dimensions of the image
        r1,c1,_ = data.shape
        r2,_ = label.shape

        if r1 == r2:
            # get full divisions of segment into shape
            r = r1//shape[0]
            c = c1//shape[1]

            rd = 0
            cd = 0
            if r1%shape[0] > 0:
                rd = 1
            if c1%shape[0] > 0:
                cd = 1

            # initialize variables
            x1 = 0
            window = 0
            
            if fileCnt > 5000:
                # reset file count & change subDir
                fileCnt = 0
                subDirCount += 1
                subdir = "sub" + str(subDirCount)
                # make new directories
                os.mkdir(d + "/data/" + subdir)
                os.mkdir(d + "/label/" + subdir)
                print(d + subdir)
                
            for n in range(r + rd):
                y1 = 0
                for m in range(c + cd):
                    # get upper bounds of window
                    x2 = (n+1)*shape[0] 
                    y2 = (m+1)*shape[1] 
                    # check if outer dimension is larger than image size and adjust
                    if x2 > r1:
                        x2 = r1
                        x1 = r1 - shape[0]

                    if y2 > c1:
                        y2 = c1
                        y1 = c1 - shape[1]
                    
                    # crop image
                    imgCrop = data[x1:x2,y1:y2,:]
                    labCrop = label[x1:x2,y1:y2]

                    # save image
                    io.imsave(d + "/data/" + subdir  + "/" + name + "_" + str(window) + ".jpg",imgCrop)
                    io.imsave(d + "/label/" + subdir + "/" + name + "_" + str(window) + ".jpg",labCrop)
                    
                    fileCnt += 1

                    y1 = y2
                    window += 1

                x1 = x2

            # delete the image to save space
            os.remove(d + '/data/' + img)
            os.remove(d + '/label/' + img)

#----------------------------------------------------------------------------
#                                 Main
#----------------------------------------------------------------------------
def preprocessData():
    #renameLabels()
    makeSplitDirs()
    splitImagesIntoDirectories()
    # create list of parameters to give to Pool
    paramList = []
    baseDirs = ['train','validate']
    classDirs = ['maize','maizevariety','wheat','mungbean']
    for i in baseDirs:
        # go through each class
        for j in classDirs:
            paramList.append(
                {'shape' : (256,256),
                'baseDir': i,
                'classDir': j
            })

    pool = Pool()
    pool.map(splitImageMP,paramList)

#----------------------------------------------------------------------------
#                                 Run
#----------------------------------------------------------------------------
if __name__ == "__main__":
    preprocessData()