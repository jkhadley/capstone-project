import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import numpy as np 
import random
from shutil import copy
from skimage import io
from multiprocessing import Pool

def renameLabels(dataPath,ignoreDirs,ignore,replace):
    """Goes through the data directories and renames labels so that 
    they match the image names.

    Parameters
    ----------
    dataPath : String
        Location to the base directory of the data
    ignoreDirs : List of Strings
        Directories in the base directory to ignore
    ignore : List of Strings
        List of patterns to ignore
    replace : Dictionary of Strings
        Patterns to replace and what to replace them with
    """
    oldDir = os.getcwd()
    os.chdir(dataPath)
    
    pwd = os.getcwd()
    dirs = os.listdir()

    replaceKeys = []
    replaceValues = []
    # get key,value pairs from dictionary
    for k, v in replace.items():
        replaceKeys.append(k)
        replaceValues.append(v)

    # go through directories and make replacements
    for i in dirs:
        if i in ignoreDirs:
            pass
        else:
            d = pwd + "/" + i + "/label/"
            labels = os.listdir(d)
            for j in range(len(labels)):
                l = labels[j].copy()
                # make replacements 
                for k in ignore:
                    l.replace(k,"")
                for k in range(len(replaceKeys)):
                    l.replace(replaceKeys[k],replaceValues[k])
                # rename file
                os.rename(d + labels[j],d + l)

    os.chdir(oldDir)

def splitImagesIntoDirectories(path,ignoreDirs,propTrain,propValidate):
    """Split image and label pairs into train, validate, and test sets.

    Parameters
    ----------
    path : String
        Location to the base directory of the data 
    ignoreDirs : List of Strings
        Directories in the base directory to ignore
    propTrain : float [0->1]
        proportion of data that should be training data
    propValidate : float [0->1]
        proportion of data that should be validation data
    """
    oldDir = os.getcwd()
    os.chdir(path)

    pwd = os.getcwd()
    dirs = os.listdir()

    for i in dirs:
        if i in ignoreDirs:
            pass
        else:
            d = pwd + "/" + i 
            
            names = os.listdir(d + '/data/')

            trainInd,valInd,testInd = randomSplit(len(names),
                                                propTrain,
                                                propValidate)
            # put data into appropriate directories
            for j in range(len(names)):
                if j in trainInd:
                    copy(d + "/data/" + names[j],
                        pwd + '/train/' + i + '/data/' + names[j])
                    copy(d + "/label/" + names[j],
                        pwd + '/train/' + i + '/label/'+ names[j])
                elif j in valInd:
                    copy(d + "/data/" + names[j],
                        pwd + '/validate/' + i + '/data/'+ names[j])
                    copy(d + "/label/" + names[j],
                        pwd + '/validate/' + i + '/label/'+ names[j])
                elif j in testInd:
                    copy(d + "/data/" + names[j],
                        pwd + '/test/' + i + '/data/'+ names[j])
                    copy(d + "/label/" + names[j],
                        pwd + '/test/' + i + '/label/'+ names[j])
        print(i + "done")
    os.chdir(oldDir)

def randomSplit(l,propTrain,propValidate):
    """Create list of indexes to split the data into training, 
    validation, and testing sets.

    Parameters
    ----------
    l : int
        length of the list to
    propTrain : float [0->1]
        proportion of data that should be training data
    propValidate : float [0->1]
        proportion of data that should be validation data
    
    Returns
    -------
    trainInd : List of int
        Indices to use for training data
    valInd : List of int
        Indices to use for validation data
    testInd : List of int
        Indices to use for testing data
    """
    # create list of indexes
    ind = [i for i in range(l)]
    random.shuffle(ind)
    b1 = round(propTrain*len(ind))
    b2 = round((propTrain+propValidate)*len(ind))
    trainInd = ind[:b1]
    valInd = ind[b1:b2]
    testInd = ind[b2:]
    return trainInd,valInd,testInd

def makeSplitDirs(path,classNames):
    """Make directories to put data into.

    Parameters
    ----------
    path : String
        Where to start putting the directories 
    classNames : List of Strings
        Class directories to put into each category
    """
    oldDir = os.getcwd()
    os.chdir("../../../data/groundcover2016/")
    pwd = os.getcwd()
    # mkdir if it doesn't exist
    l1 = ['train','test','validate']
    l2 = ['data','label']

    for i in l1:
        os.mkdir(pwd + "/" + i)
        for j in classNames:
            os.mkdir(pwd + "/" + i + '/' + j)
            for k in l2:
                os.mkdir(pwd + "/" + i + '/' + j + "/" + k)
    os.chdir(oldDir)


def splitImageMP(params):
    """Splits images into smaller pieces.

    Splits and renames the image label pairs of data into pieces of the
    specified size. Image and label pairs will be placed into 
    sub-directories of the specified size so that the directories do 
    not get too large to deal with. Removes the full size image from 
    the directory when it is done splitting it. 

    Parameters
    ----------
    path : String
        Location of data
    shape : tuple of ints
        Size that the images should be split into
    whichDir : String
        Which directory in the path directory to use
    whichClass : String
        Which class to use
    subdirSize : int
        Approximately how many images to put into each subdirectory
    """

    path = params['path']
    shape = params['shape']
    whichDir = params['whichDir']
    whichClass = params['whichClass']
    subdirSize = params['subdirSize']

    os.chdir(path)
    pwd = os.getcwd()

    d = pwd + '/' + whichDir + '/' + whichClass
    images = os.listdir(d +'/data')
    subDirCount = 0
    fileCnt = 0
    subdir = "sub0/"

    # make new directories
    os.mkdir(d + "/data/" + subdir)
    os.mkdir(d + "/label/" + subdir)

    # split images
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

            # if the shape doesn't cleanly divide by the shape,
            # indicate that there is a remainder
            if r1%shape[0] > 0:
                rd = 1
            if c1%shape[0] > 0:
                cd = 1

            # initialize variables
            x1 = 0
            window = 0
            
            # check if need to make new subdirectory
            if fileCnt > subdirSize:
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
def preprocessData(path,classNames,subdirSize,imageShape,trainProp,valProp):
    """Calls the other functions in the right order and feeds them the
    appropriate arguments. Assumes that renameLabels has already been 
    called or that it doesn't need to be called. 

    Parameters
    ----------
    path : String
        Location of the base directory where the data is located.
    classNames : List of Strings
        Names of classes in the that need preprocessing
    subdirSize : int
        Number of images and labels to put into each sub-directory
    imageShape : tuple of int
        Size to split images into
    trainProp : float [0->1]
        Proportion of data to put into training directory
    valProp : float [0->1]
        Proportion of data to put into validation directory
    """ 
    
    makeSplitDirs(path,classNames)

    splitImagesIntoDirectories(path,['train','test','validate'],trainProp,valProp)
    
    # create list of parameters to give to Pool
    paramList = []
    baseDirs = ['train','validate']
    for i in baseDirs:
        # go through each class
        for j in classNames:
            paramList.append(
                {'shape' : imageShape,
                'whichDir': i,
                'whichClass': j,
                'path' : path,
                'subdirSize' : subdirSize
            })
    pool = Pool()
    # split the images into the specified size
    pool.map(splitImageMP,paramList)

#----------------------------------------------------------------------------
#                                 Run
#----------------------------------------------------------------------------
if __name__ == "__main__":
    path = "../../../data/groundcover2016/"
    classNames = ['wheat','maize','maizevariety','mungbean']
    ignore = ["CE_NADIR_"]
    replace = {".tif" : ".jpg"}
    size = 5000
    shape = (256,256)

    renameLabels(path,['train','validate','test'],ignore,replace)

    preprocessData(path,classNames,size,shape,0.8,0.1)