# Author(s): Joseph Hadley
# Date Created : 2018-06-18
# Date Modified: 2018-07-19
# Description: Generators to feed the keras models with
#----------------------------------------------------------------------------------------------------------------
from keras.preprocessing.image import ImageDataGenerator
from notebookFunctions import dirSize
from skimage import io
import numpy as np
import random
import os
#----------------------------------------------------------------------------------------------------------------
#                                               Setup data generators
#----------------------------------------------------------------------------------------------------------------
def singleClassGenerator(path,className):
    """ 
    Generates pairs of Single class images.

    Generates images of a single class. It will start reiterating through the directories when it runs out of unique sets of images.

    Parameters
    ----------
    path : String
        location to base directory of class folders
    className : String
        Which class in the base directory to have images generated from it
    
    Yields
    ------
    image : np.array
        np.array containing the image
    label : np.array
        np.array containing the label
    """
    os.chdir(path)
    # get list of subdirectories
    dataPath = path + className + "/data/"
    labelPath = path + className + "/label/"
    subdirs = os.listdir(dataPath)
    numOfDirs = len(subdirs)
    random.shuffle(subdirs)
    # initialize variables
    ind = 0
    dirInd = 0
    whichSubDir = subdirs[dirInd]
    imageNames = os.listdir(dataPath + whichSubDir)
    dirLen = len(imageNames)
    random.shuffle(imageNames) 

    while True:
        # check if still have images in subdirectory
        if ind >= dirLen:
            dirInd += 1
            ind = 0
            # check if iterated through all subdirectories
            if dirInd >= numOfDirs:
                dirInd = 0
                random.shuffle(subdirs)
            # get a new subdirectory
            whichSubDir = subdirs[dirInd]
            imageNames = os.listdir(dataPath + whichSubDir)
            random.shuffle(imageNames)
            dirLen = len(imageNames)

        # get image
        image  = io.imread(dataPath + whichSubDir + "/" + imageNames[ind])

        # get label
        label = io.imread(labelPath + whichSubDir + "/" + imageNames[ind])

        ind += 1
        # normaize image and label and send out
        if(np.max(image) > 1):
            image = image/255
        if(np.max(label) > 1):
            label = label/np.max(label)
            label[label > 0.5] = 1
            label[label < 0.5] = 0
        
        yield (image,label)

def multiClassGenerator(path,classMap,labelShape):
    """ 
    Generates pairs of multiclass images.

    Generates images by cycling through each class. It will start reiterating through the directories when it runs out of unique sets of images.

    Parameters
    ----------
    path : String
        location to base directory of class folders
    classMap : dictionary
        Maps class to categorical output
    labelShape : tuple
        Size of the tuple

    Yields
    ------
    image : np.array
        np.array containing the image
    label : np.array
        np.array containing the label
    """
    os.chdir(path)
    cwd = os.getcwd()
    # get class directories
    classDirs = os.listdir()
    lenClassDirs = len(classDirs)
    dirLength = [0]*lenClassDirs
    imageNames = [0]*lenClassDirs
    subDirOrder = [0]*lenClassDirs
    whichSubDir = [0]*lenClassDirs
    subDirLen = [0]*lenClassDirs
    numOfSubDirs = [0]*lenClassDirs
    pastDirsLenForEpoch = [0]*lenClassDirs
    
    # initialize lists of subdirectories and first lists of image names 
    for c in range(len(classDirs)):
        subDirs =os.listdir(classDirs[c] + "/data/") 
        random.shuffle(subDirs)
        subDirOrder[c] = subDirs
        numOfSubDirs[c] = len(subDirs)
        tmpImages = os.listdir(cwd +'/'+ classDirs[c] +'/data/' + subDirs[0])
        # initialize subdirectories for later
        subDirLen[c] += len(tmpImages)
        imageNames[c] = tmpImages
        random.shuffle(imageNames[c])

    im = 0
    
    while True:
        listPos = im//lenClassDirs
        whichDir = im%lenClassDirs
        
        whichImInSubDir = listPos%dirLength[whichDir]-pastDirsLenForEpoch[whichDir]
        
        # check if need to change the subdir
        if whichImInSubDir >= subDirLen[whichDir] or whichImInSubDir < 0:
            # change the subdirectory to pull files from
            whichSubDir[whichDir]+=1
            
            # reset subdirectories if required
            if whichSubDir[whichDir] >= numOfSubDirs[whichDir]:
                whichSubDir[whichDir] = 0
                random.shuffle(subDirOrder[whichDir])
                whichImInSubDir = 0
                pastDirsLenForEpoch[whichDir] = 0
            else:
                pastDirsLenForEpoch[whichDir] += subDirLen[whichDir]
               
            # get list of images in the new subdirectory, length of new subdirectory, and shuffle them          
            imageNames[whichDir] = os.listdir(cwd + '/' + classDirs[whichDir] + "/data/" + subDirOrder[whichDir][whichSubDir[whichDir]])     
            subDirLen[whichDir] = len(imageNames[whichDir])
            random.shuffle(imageNames[whichDir])

        # paths to images
        imagePath = subDirOrder[whichDir][whichSubDir[whichDir]] + "/" + imageNames[whichDir][whichImInSubDir]
        beginDir = cwd + '/' + classDirs[whichDir]

        # load in regular image
        image = io.imread(beginDir + "/data/" + imagePath)
        # normalize image
        if(np.max(image) > 1):
            image = image/255
    
        label = np.zeros(shape = labelShape)
        tmpLab = io.imread(beginDir + "/label/" + imagePath)    

        # modify label
        label[: , : , 0] = (tmpLab >= 1).astype(int)
        label[: , : , classMap[classDirs[whichDir]]] = (tmpLab < 1).astype(int) 

        im += 1
        yield(image,label)


def batchGenerator(batch_size,generator,data_shape,label_shape):
    """ 
    Creates batches of specified size to feed the model.

    Generates 4-D numpy arrays of images.

    Parameters
    ----------
    batch_size : int
        Size of the batch to be returned each iteration
    generator : generator
        Generator that generates pairs of images        
    data_size : tuple
        shape of individual data images to be generated
    label_size : tuple
        shape of individual label images to be generated        
    
    Yields
    ------
    data : 4-D np.array
        4-D array of data images
    label : 4-D np.array
        4-D array of label images 

    """
    batchCnt = 0

    while True:
        batchCnt +=1
        # initialize arrays
        data = np.zeros(shape = data_shape)
        label = np.zeros(shape = label_shape)
        # use generator to populate matrices
        for i,batch in enumerate(generator):
            if i >= batch_size:
                break
            else:
                data[i,:,:,:] = batch[0]
                label[i,:,:,:] = batch[1]
        if batchCnt%5000 == 0:
            print(batchCnt)
        yield(data,label)


def getBatchGenerators(batch_size,path,shape,**kwargs):
    """ 
    Creates training and validation batch generators.

    Creates generator for single or multi-class data based on either the classMap or className that is passed to it via the kwargs.

    Parameters
    ----------
    batch_size : int
        Size of the batch to be returned each iteration
    path : String
        Location of the data up until the classes
    shape : tuple
        Specifies the shape of the image
    classMap : dictionary
        Maps class to categorical output. Default is None
    ClassName : String
        Name of the single class to use. Default is None 

    Returns
    ------
    trainGenerator : generator
        Batch Generator for training data
    validationGenerator : generator
        Batch Generator for validation data
    """
    # check if multiple classes
    if 'classMap' in kwargs:
        # number of classes is unique values + 1 for background
        n_classes = len(np.unique(list(kwargs.values()))) + 1
    else:
        # check if there is a class name
        if 'className' not in kwargs:
            raise NameError("Need either className or classMap")
        className = kwargs['className']
        n_classes = 1

    data_shape = (batch_size, shape[0], shape[1], 3)
    label_shape = (batch_size, shape[0], shape[1], n_classes)

    if n_classes > 1:
        gen = multiClassGenerator(path,kwargs['classMap'],(shape[0],shape[1],n_classes))
    else:
        gen = singleClassGenerator(path,className)
    
    trainGenerator = batchGenerator(batch_size,gen,data_shape,label_shape)
    validateGenerator = batchGenerator(batch_size,gen,data_shape,label_shape)

    return (trainGenerator,validateGenerator)