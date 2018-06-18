# Author(s): Joseph Hadley
# Date Created : 2018-06-12
# Date Modified: 2018-06-16
# Description: Try to create a CNN using keras that successfully implements semantic segmentation on the corn images
#----------------------------------------------------------------------------------------------------------------
import os
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential
from skimage import io
#----------------------------------------------------------------------------------------------------------------
#                                               Setup
#----------------------------------------------------------------------------------------------------------------
# define path to data
path = "../../data/groundcover2016/maize/data"

# Training variables to tweak
epochs = 10
conv_kernel_size = (3,3)
conv_stride = (1,1)
activation = 'relu'
pool_size = (2,2)
pool_stride = (2,2)

# Firm Training variables
num_of_classes = 2 # just trying to predict if plant or not right now
width = 2048
height = 1152
input_shape = (width,height,3)
#----------------------------------------------------------------------------------------------------------------
#                                           Build the CNN
#----------------------------------------------------------------------------------------------------------------
def unet():
    model = Sequential()

    conv_depth = 64

    model.add(Conv2D(conv_depth,kernel_size = conv_kernel_size,strides = conv_stride,input_shape=input_shape))
    model.add(Conv2D(conv_depth,kernel_size = conv_kernel_size,strides = conv_stride))


    for i in range(0,4):
        # Pool down
        model.add(MaxPooling2D(pool_size = pool_size, strides = pool_stride))

        conv_depth *= 2
        print(conv_depth)
        # Convolution on the 1st pooled layer
        model.add(Conv2D(conv_depth,kernel_size = conv_kernel_size,strides = conv_stride))
        model.add(Conv2D(conv_depth,kernel_size = conv_kernel_size,strides = conv_stride))

    
    # Upsample
    #model.add(UpSampling2D(size = ()))

if __name__ == "__main__":
    unet()
