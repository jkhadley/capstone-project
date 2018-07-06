# Author(s): Joseph Hadley
# Date Created : 2018-06-18
# Date Modified: 2018-06-20
# Description: Create a generator to train the keras model with
#----------------------------------------------------------------------------------------------------------------
import os
from skimage import io
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
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

def generateMultiClassBatch(batch_size,params):
    
    data_size = (batch_size,params['shape'][0],params['shape'][1],3)
    label_size = (batch_size,params['shape'][0],params['shape'][1],params['n_classes'])

    data = np.zeros(shape = data_size)
    label = np.zeros(shape = label_size)

    for i in range(batch_size):
        tmp = generateMulticlassData(params)
        data[i,:,:,:] = tmp[0]
        label[i,:,:,:] = tmp[1]
    
    yield(data,label)


def generateMulticlassData(params):
    classMap = params['classMap']
    
    os.chdir(params['path'])
    cwd = os.getcwd()
    # get class directories
    classDirs = os.listdir()
    # get list of images in class directories & generate randomized order of images to go through
    images = []
    whichImage = []
    dirLengths = []
    totLength = 0
    labelShape = (params['shape'][0],params['shape'][1],params['n_classes'])

    listLoc = [0]*params['n_classes']
    
    for c in classDirs:
        tmpImages = os.listdir(cwd +'/'+ c + '/data')
        tmpLen = len(tmpImages)
        images.append(tmpImages)
        whichImage.append(np.random.randint(0,tmpLen,size = tmpLen))
        dirLengths.append(tmpLen)
        totLength += tmpLen
    
    for im in range(totLength*params['epochs']):
        i = im//params['n_classes']
        j = listLoc[i]%dirLengths[i]
        listLoc[i] += 1
    
        # load in regular image
        image = io.imread(cwd + "/data/" + images[i][j])
        # normalize image
        if(np.max(image) > 1):
            image = image/np.max(image)
    
        # initialize label
        label = np.zeros(shape = labelShape)

        # load in label 
        tmpLab = io.imread(cwd + "/label/" + images[i][j])    

        # modify label
        label[: , : , 0] = (tmpLab >= 1).astype(int)
        label[: , : , classMap[classDirs[i]]] = (tmpLab < 1).astype(int) 

        yield(image,label)

def normalizeData(image, label):
    if(np.max(image) > 1):
        image = image/np.max(image)
    if(np.max(label) > 1):
        label = label/np.max(label)
        label[label > 0.5] = 1
        label[label < 0.5] = 0

    return (image,label)

def generateRescaledData(path,dataFolder,labelFolder,rescaleSize,stride):
    cwd = os.getcwd()
    os.chdir(path)
    
    totalDir = os.getcwd()
    
    # make file paths 1 variable
    imgDir = totalDir + dataFolder
    labelDir = totalDir + labelFolder

    # get list of files from each directory
    imgFiles = np.sort(os.listdir(imgDir))
    labelFiles = np.sort(os.listdir(labelDir))

    if np.all(imgFiles == labelFiles):
        for i in range(len(imgFiles)):
            # load in images
            img = io.imread(imgDir + '/' + imgFiles[i],1)
            label = io.imread(labelDir + '/' + imgFiles[i],0)
            
            w1,_, _ = img.shape
            w2,_ = label.shape            

            if w1 != w2:
                img = rotate(img,90)
                       
            # use tensorflow to get patches of images
            with tf.Session() as _:
                imgPatches = tf.extract_image_patches(images = np.expand_dims(img,axis = 0),
                                                ksizes = [1,rescaleSize[0],rescaleSize[1],1],
                                                strides = [1,stride[0],stride[1],1],
                                                rates = [1,1,1,1],
                                                padding = "VALID").eval()

                labelPatches = tf.extract_image_patches(images = np.expand_dims(np.expand_dims(label,axis = 0),axis = -1),
                                                ksizes = [1,rescaleSize[0],rescaleSize[1],1],
                                                strides = [1,stride[0],stride[1],1],
                                                rates = [1,1,1,1],
                                                padding = "VALID").eval()
                #imgArray = tf.reshape(imgPatches, [45, 256, 256, 3])
                #labArray = tf.reshape(labelPatches, [45, 256, 256,1])
            
            imgDim = np.shape(imgPatches)
            labDim = np.shape(labelPatches)
            
            # initialize patch arrays
            imgArray = np.zeros(((imgDim[1]-1)*(imgDim[2]-1),rescaleSize[0],rescaleSize[1],3))
            labArray = np.zeros(((labDim[1]-1)*(labDim[2]-1),rescaleSize[0],rescaleSize[1],1))

            # reshape patches from tensorflow into images
            for i in range(imgDim[1]-1):
                for j in range(imgDim[2]-1):
                    ind = (imgDim[1]-1)*i + j
                    
                    tmpImg = imgPatches[0,i,j,:].reshape(rescaleSize[0],rescaleSize[1],3)
                    tmpLab = labelPatches[0,i,j,:].reshape(rescaleSize[0],rescaleSize[1],1)
                    # normalize data
                    if np.max(tmpImg) > 1:
                        tmpImg = tmpImg/np.max(tmpImg)
                    if np.max(tmpLab) > 1:
                        tmpLab = tmpLab/np.max(tmpLab)

                    imgArray[ind,:,:,:] = tmpImg
                    labArray[ind,:,:,:] = tmpLab
            
            yield (imgArray,labArray)
    
    os.chdir(cwd)

#----------------------------------------------------------------------------------------------------------------
#                                               test the generator
#----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    path = "../../data/groundcover2016/maize/train/"
    classMap = {
    'maize' : 1,
    'maizevariety': 1,
    'wheat': 2,
    'mungbean':3
    }

    batch_size = 10

    trainGenParams = {
        'path' : path,
        'classMap' : classMap,
        'epochs' : 5,
        'n_classes': 4,
        'shape':(256,256),
    } 

    gen = generateMulticlassData(trainGenParams)

    for i,batch in enumerate(gen):
        if(i > 1):
            break
        print(np.max(batch[0][0]))
        plt.subplot(231)
        plt.imshow(batch[0][4].astype('uint8'))
        plt.axis("off")
        plt.subplot(232)
        plt.imshow(batch[0][5].astype('uint8'))
        plt.axis("off")
        plt.subplot(233)
        plt.imshow(batch[0][6].astype('uint8'))
        plt.axis("off")
        plt.subplot(234)
        plt.imshow(batch[1][4])
        plt.axis("off")
        plt.subplot(235)
        plt.imshow(batch[1][5])
        plt.axis("off")
        plt.subplot(236)
        plt.imshow(batch[1][6])
        plt.axis("off")       
        plt.show()
            
