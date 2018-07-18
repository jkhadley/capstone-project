# Author(s): Joseph Hadley
# Date Created : 2018-06-18
# Date Modified: 2018-07-17
# Description: Generator to feed the keras models with
#----------------------------------------------------------------------------------------------------------------
import os
from skimage import io
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import random
#----------------------------------------------------------------------------------------------------------------
#                                               Setup data generators
#----------------------------------------------------------------------------------------------------------------
def generateData(params):
    #path,whichData,n_classes,batch_size,target_size
    os.chdir(params['path'])
    # Generator Parameters
    rot = 0
    width_shift = 0.1
    height_shift = 0.1
    seed = 123

    data_datagen = ImageDataGenerator(
        #featurewise_center=True,
        rotation_range= rot,
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        horizontal_flip=True
        )

    image_generator = data_datagen.flow_from_directory(
        './' + params['which_data'],
        classes = ["data"],
        class_mode = None,
        color_mode = "rgb",
        target_size = params['target_size'],
        batch_size = params['batch_size'],
        save_to_dir = None,
        seed = seed 
        )

    label_generator = data_datagen.flow_from_directory(
        './' + params['which_data'],
        classes = ["label"],
        class_mode = None,
        color_mode = "grayscale",
        target_size = params['target_size'],
        batch_size = params['batch_size'],
        save_to_dir = None,
        seed = seed 
        )

    # zip the generators together
    train_gen = zip(image_generator,label_generator)
    
    for (img,label) in train_gen:
        img,label = normalizeData(img,label)
        yield (img,label)

def normalizeData(image, label):
    if(np.max(image) > 1):
        image = image/np.max(image)
    if(np.max(label) > 1):
        label = label/np.max(label)
        label[label > 0.5] = 1
        label[label < 0.5] = 0

    return (image,label)

def genMulticlassData(params):
    classMap = params['classMap']
    
    os.chdir(params['path'])
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

    totLength = 0
    labelShape = (params['shape'][0],params['shape'][1],params['n_classes'])
    
    for c in range(len(classDirs)):
        
        subDirs =os.listdir(classDirs[c] + "/data/") 
        
        random.shuffle(subDirs)
        subDirOrder[c] = subDirs
        numOfSubDirs[c] = len(subDirs)

        for j in range(len(subDirs)):
            tmpImages = os.listdir(cwd +'/'+ classDirs[c] +'/data/' + subDirs[j])
            dirLength[c]+=len(tmpImages)
            # initialize subdirectories for later
            if j == 0:
                subDirLen[c] += len(tmpImages)
                imageNames[c] = tmpImages
                random.shuffle(imageNames[c])

    totLength = sum(dirLength)
    
    for im in range(totLength*params['epochs']):
        
        listPos = im//lenClassDirs
        whichDir = im%lenClassDirs

        # here is the problemo
        whichImInSubDir = listPos%dirLength[whichDir]-pastDirsLenForEpoch[whichDir]
        
        # check if need to change the subdir
        if whichImInSubDir >= subDirLen[whichDir] or whichImInSubDir < 0:
            # change the subdirectory to pull files from
            whichSubDir[whichDir]+=1
            
            # reset the images in the directory
            if whichSubDir[whichDir] >= numOfSubDirs[whichDir]:
                whichSubDir[whichDir] = 0
                whichImInSubDir = 0
                pastDirsLenForEpoch[whichDir] = 0
            else:
                pastDirsLenForEpoch[whichDir] += subDirLen[whichDir]
                whichImInSubDir = listPos%dirLength[whichDir]-pastDirsLenForEpoch[whichDir]
            
            # get list of images in the new subdirectory, length of new subdirectory, and shuffle them          
            imageNames[whichDir] = os.listdir(cwd + '/' + classDirs[whichDir] + "/data/" + subDirOrder[whichDir][whichSubDir[whichDir]])     

            subDirLen[whichDir] = len(imageNames[whichDir])
            
            random.shuffle(imageNames[whichDir])

        # long paths that i have to enter later
        imagePath = subDirOrder[whichDir][whichSubDir[whichDir]] + "/" + imageNames[whichDir][whichImInSubDir]
        beginDir = cwd + '/' + classDirs[whichDir]

        # load in regular image
        image = io.imread(beginDir + "/data/" + imagePath)
        # normalize image
        if(np.max(image) > 1):
            image = image/np.max(image)
    
        # initialize label
        label = np.zeros(shape = labelShape)

        # load in label 
        tmpLab = io.imread(beginDir + "/label/" + imagePath)    

        # modify label
        label[: , : , 0] = (tmpLab >= 1).astype(int)
        label[: , : , classMap[classDirs[whichDir]]] = (tmpLab < 1).astype(int) 
        
        yield(image,label)

def generateMultiClassBatch(batch_size,params):
    
    data_size  = (batch_size, params['shape'][0], params['shape'][1], 3)
    label_size = (batch_size, params['shape'][0], params['shape'][1], params['n_classes'])

    gen = genMulticlassData(params)
    batchCnt = 0
    while True:
        batchCnt +=1
        # initialize arrays
        data = np.zeros(shape = data_size)
        label = np.zeros(shape = label_size)
        # use generator to populate matrices
        for i,batch in enumerate(gen):
            if i >= batch_size:
                break
            else:
                data[i,:,:,:] = batch[0]
                label[i,:,:,:] = batch[1]
        if batchCnt%5000 == 0:
            print(batchCnt)
        yield(data,label)

#----------------------------------------------------------------------------------------------------------------
#                                               test the generator
#----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    path = "../../../data/groundcover2016/tmp/"
    cwd = os.getcwd()
    os.chdir(path)
    print("Directory fine")
    p = os.getcwd()

    classMap = {
    'maize' : 1,
    'maizevariety': 1,
    'wheat': 2,
    'mungbean':3
    }

    batch_size = 10

    trainGenParams = {
        'path' : p,
        'classMap' : classMap,
        'epochs' : 5,
        'n_classes': 4,
        'shape':(256,256),
    } 

    #gen = genMulticlassData(trainGenParams)
    gen = generateMultiClassBatch(batch_size,trainGenParams)
    for i,batch in enumerate(gen):
        if(i > 500000):
            break
        #plt.imshow(batch[0][0])
        #plt.imshow(batch[1][0,:,:,1])
        #plt.show()
        
            
