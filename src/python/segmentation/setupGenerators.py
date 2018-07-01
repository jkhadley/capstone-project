# Author(s): Joseph Hadley
# Date Created : 2018-06-18
# Date Modified: 2018-06-20
# Description: Create a generator to train the keras model with
#----------------------------------------------------------------------------------------------------------------
import os
import cv2
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
#----------------------------------------------------------------------------------------------------------------
#                                               Setup data generators
#----------------------------------------------------------------------------------------------------------------
def generateData(path,whichData,save_dir,batch_size,target_size):
    os.chdir(path)
    # Generator Parameters
    rot = 0
    width_shift = 0.1
    height_shift = 0.1
    #target_size = (2048,1152)
    seed = 123

    cwd = os.getcwd()

    if save_dir == False:
        data_aug_dir = None
        label_aug_dir = None
    else:
        data_aug_dir = cwd + "\\"+ whichData +"\\augmentData"
        label_aug_dir = cwd + "\\"+ whichData +"\\augmentLabel"

    data_datagen = ImageDataGenerator(
        #featurewise_center=True,
        rotation_range= rot,
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        horizontal_flip=True
        )

    image_generator = data_datagen.flow_from_directory(
        './' + whichData,
        classes = ["data"],
        class_mode = None,
        color_mode = "rgb",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = data_aug_dir,
        save_prefix = "aug",
        seed = seed 
        )

    label_generator = data_datagen.flow_from_directory(
        './' + whichData,
        classes = ["label"],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = label_aug_dir,
        save_prefix = "aug",
        seed = seed 
        )

    # zip the generators together
    train_gen = zip(image_generator,label_generator)
    
    for (img,label) in train_gen:
        img,label = normalizeData(img,label)
        yield (img,label)

def simpleGenerator():
    pass

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
            img = cv2.imread(imgDir + '/' + imgFiles[i],1)
            label = cv2.imread(labelDir + '/' + imgFiles[i],0)
            
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
    path = "../../data/groundcover2016/maize/size1"
    #gen = generateData(path,"train",True,2)
    
    '''
    for i,_ in enumerate(gen):
        if(i > 3):
            break
    '''
    gen = generateRescaledData(path + "/train","/data","/label",(256,256),(256,256))
    
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
            
