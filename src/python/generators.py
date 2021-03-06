# Author(s): Joseph Hadley
# Date Created : 2018-06-18
# Date Modified: 2018-08-04
# Description: Generators to feed the keras models with
#----------------------------------------------------------------------------------------------------------------
from keras.preprocessing.image import ImageDataGenerator
from misc import dirSize
from skimage import io
import numpy as np
import random 
import os
#----------------------------------------------------------------------------------------------------------------
#                                               Setup data generators
#----------------------------------------------------------------------------------------------------------------
def singleClassGenerator(path,classMap,regression):
    """Generates pairs images and labels for a single-class classifier.

    Generates images of a single class. It will start re-iterating through the directories when it runs out of unique sets of images.

    Parameters
    ----------
    path : String
        location to base directory of class folders.
    className : String
        The directory in the base directory to get the images and labels from.
    regression: boolean
        Indicates the type of model being trained to return the appropriate output.
    
    Yields
    ------
    image : np.array
        np.array containing the image
    label : np.array
        np.array containing the label
    """
    os.chdir(path)
    # get list of subdirectories
    className = list(classMap.keys())[0]
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
    s = np.shape(io.imread(dataPath + whichSubDir + "/" + imageNames[ind]))
    label = np.zeros((s[0],s[1],2))
    
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
        tmpLab = io.imread(labelPath + whichSubDir + "/" + imageNames[ind])
        
        ind += 1
        # normaize image and label and send out
        if(np.max(image) > 1):
            image = image/255

        image = image.astype(np.float16)

        label[:,:,1] = (tmpLab < 1)
        label[:,:,0] = (tmpLab >= 1)

        # if regression model convert label 
        if regression == True:
            labelShape = np.shape(label)
            labelOutput = (np.sum(np.sum(label,axis = 0),axis = 0)/(labelShape[0]*labelShape[1]))
        else:
            labelOutput = label

        yield (image,labelOutput)

def multiClassGenerator(path,classMap,labelShape,regression):
    """Generates image, label pairs for multi-class model.

    Generates images by cycling through each class. It will start reiterating through the directories when it runs out of unique sets of images.
    Parameters
    ----------
    path : String
        Location to base directory of class folders
    classMap : dictionary
        Maps class names to the appropriate categorical output.
    labelShape : tuple
        Size of the tuple
    regression : boolean
        Indicates the type of model being trained to return the appropriate output.

    Yields
    ------
    image : np.array
        np.array containing the image
    labelOutput : np.array
        np.array containing the label
    """
    os.chdir(path)
    cwd = os.getcwd()
    # Get the class directories.
    classDirs = os.listdir()
    lenClassDirs = len(classDirs)
    # initialize lists to store information in.
    dirLength = [0]*lenClassDirs
    imageNames = [0]*lenClassDirs
    subDirOrder = [0]*lenClassDirs
    whichSubDir = [0]*lenClassDirs
    subDirLen = [0]*lenClassDirs
    numOfSubDirs = [0]*lenClassDirs
    pastDirsLenForEpoch = [0]*lenClassDirs
    # Determine the number of classes to generate.
    n_classes = len(np.unique(list(classMap.values()))) + 1
    
    # Initialize lists of subdirectories and first lists of image names 
    for c in range(len(classDirs)):
        subDirs =os.listdir(classDirs[c] + "/data/")
        random.shuffle(subDirs)
        subDirOrder[c] = subDirs
        numOfSubDirs[c] = len(subDirs)
        tmpImages = os.listdir(cwd +'/'+ classDirs[c] +'/data/' + subDirs[0])
        # initialize sub-directories for later.
        for s in range(len(subDirs)):
            tmpImages = os.listdir(cwd +'/'+ classDirs[c] +'/data/' + subDirs[s])
            tmpLen = len(tmpImages)
            dirLength[c] += tmpLen
            if s == 0:
                subDirLen[c] = tmpLen
                imageNames[c] = tmpImages
                random.shuffle(imageNames[c])
            
    im = 0

    while True:
        listPos = im//lenClassDirs
        whichDir = im%lenClassDirs
        # Get Which image in the sub-directory to use.
        whichImInSubDir = listPos%dirLength[whichDir]-pastDirsLenForEpoch[whichDir]
        
        # Check if need to change the subdir
        if whichImInSubDir >= subDirLen[whichDir] or whichImInSubDir < 0:
            # change the subdirectory to pull files from
            whichSubDir[whichDir]+=1
            
            # Reset the sub-directories if required
            if whichSubDir[whichDir] >= numOfSubDirs[whichDir]:
                whichSubDir[whichDir] = 0
                random.shuffle(subDirOrder[whichDir])
                whichImInSubDir = 0
                pastDirsLenForEpoch[whichDir] = 0
            else:
                pastDirsLenForEpoch[whichDir] += subDirLen[whichDir]
               
            # Get the list of images in the new subdirectory, length of new subdirectory, and shuffle them          
            imageNames[whichDir] = os.listdir(cwd + '/' + classDirs[whichDir] + "/data/" + subDirOrder[whichDir][whichSubDir[whichDir]])     
            subDirLen[whichDir] = len(imageNames[whichDir])
            random.shuffle(imageNames[whichDir])
            whichImInSubDir = listPos%dirLength[whichDir]-pastDirsLenForEpoch[whichDir]
            
        # Define the paths to images.
        imagePath = subDirOrder[whichDir][whichSubDir[whichDir]] + "/" + imageNames[whichDir][whichImInSubDir]
        beginDir = cwd + '/' + classDirs[whichDir]

        # Load in regular image.
        image = io.imread(beginDir + "/data/" + imagePath)
        # Normalize  the image between 0 and 1.
        if(np.max(image) > 1):
            image = image/255
        
        image = image.astype("float16")
        imShape = np.shape(image)
        label = np.zeros((imShape[0],imShape[0],n_classes))
        tmpLab = io.imread(beginDir + "/label/" + imagePath)    

        # modify label
        label[: , : , 0] = (tmpLab >= 1).astype("uint8")
        label[: , : , classMap[classDirs[whichDir]]] = (tmpLab < 1).astype("uint8") 

        # If it's a regression model sum the classes to get the proportions
        if regression == True:
            labelShape = np.shape(label)
            labelOutput = np.sum(np.sum(label,axis = 0),axis = 0)/(labelShape[0]*labelShape[1])
        else:
            labelOutput = label
        im += 1
        yield(image,labelOutput)


def batchGenerator(batch_size,generator,data_shape,label_shape):
    """Creates batches of images and their labels to feed the model.

    Parameters
    ----------
    batch_size : int
        Size of the batch to be returned each iteration.
    generator : generator
        Generator that produces pairs of images and their labels.        
    data_size : tuple
        Shape of data batch to be generated.
    label_size : tuple
        Shape of label batch to be generated.        
    
    Yields
    ------
    data : 4-D np.array
        4-D array of data
    label : np.array
        np.array of labels 
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
                data[i] = batch[0]
                label[i] = batch[1]
        if batchCnt%1000 == 0:
            print(batchCnt)
        yield(data,label)

def getBatchGenerators(batch_size,path,shape,classMap,regression):
    """Creates training and validation batch generators.

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
    regression: Boolean
        Indicates the type of model the data is being generated for. Default is False.

    Returns
    ------
    trainGenerator : generator
        Batch Generator for training data
    validationGenerator : generator
        Batch Generator for validation data
    """
    # number of classes is unique values + 1 for background
    n_classes = len(np.unique(list(classMap.values()))) + 1

    if regression == True:
        label_shape = (batch_size,n_classes)
    else:
        label_shape = (batch_size, shape[0], shape[1], n_classes)

    data_shape = (batch_size, shape[0], shape[1], 3)
   
   # create the data generators
    if n_classes > 2:
        trainGen = multiClassGenerator(path + "/train/",classMap,(shape[0],shape[1],n_classes),regression)
        validateGen = multiClassGenerator(path + "/validate/",classMap,(shape[0],shape[1],n_classes),regression)
    else:
        trainGen = singleClassGenerator(path + "/train/",classMap,regression)
        validateGen = singleClassGenerator(path + "/validate/",classMap,regression)
    
    # create the batch generators
    trainGenerator = batchGenerator(batch_size,trainGen,data_shape,label_shape)
    validateGenerator = batchGenerator(batch_size,validateGen,data_shape,label_shape)

    return (trainGenerator,validateGenerator)