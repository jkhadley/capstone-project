# Author(s): Joseph Hadley
# Date Created : 2018-06-12
# Date Modified: 2018-06-20
# Description: Create a U-Net model using keras
#----------------------------------------------------------------------------------------------------------------
import os
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate,Input
from keras.models import Sequential,Model
from keras.optimizers import Adam
from skimage import io


#----------------------------------------------------------------------------------------------------------------
#                                           Build the CNN
#----------------------------------------------------------------------------------------------------------------
def unet(n_classes,input_shape,activation,init_w):

    # Training variables to tweak
    conv_kernel_size = (3,3)
    conv_stride = (1,1)
    pool_size = (2,2)
    pool_stride = (2,2)
    pad = "same"
    up_size = (2,2) 

    conv_depth = 64

    # first two convolutional layers
    inputs = Input(input_shape)
    conv1 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(inputs)
    conv1 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv1)

    # pool down to next layer
    pool1 = MaxPooling2D(pool_size = pool_size,strides = pool_stride)(conv1)

    conv_depth *= 2 # 128

   # convolute down again
    conv2 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(pool1)
    conv2 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv2)
    
    # pool down again
    pool2 = MaxPooling2D(pool_size = pool_size,strides = pool_stride)(conv2)

    conv_depth *= 2 # 256

   # Convolution
    conv3 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(pool2)
    conv3 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv3)
    
    # pool down
    pool3 = MaxPooling2D(pool_size = pool_size,strides = pool_stride)(conv3)

    conv_depth *= 2 #512

   # Convolution
    conv4 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(pool3)
    conv4 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv4)
    
    # pool down 
    pool4 = MaxPooling2D(pool_size = pool_size,strides = pool_stride)(conv4)

    conv_depth *=2 # 1024

   # Convolution
    conv5 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(pool4)
    conv5 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv5)

    conv_depth = 512  
    # do upsampling
    up1 = UpSampling2D(size = up_size)(conv5)
    conv6 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(up1)
    
    # add in skip info
    cat1 = concatenate([conv4,conv6],axis = 3)
    conv6 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(cat1)
    conv6 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv6)

    conv_depth = 256  
    # do upsampling
    up2 = UpSampling2D(size = up_size)(conv6)
    conv7 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(up2)
    
    # add in skip info
    cat2 = concatenate([conv3,conv7],axis = 3)
    conv7 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(cat2)
    conv7 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv7)
    
    conv_depth = 128  
    
    # do upsampling
    up3 = UpSampling2D(size = up_size)(conv7)
    conv8 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(up3)
    
    # add in skip info
    cat3 = concatenate([conv2,conv8],axis = 3)
    conv8 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(cat3)
    conv8 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv8)
    
    conv_depth = 64  
    
    # do upsampling
    up4 = UpSampling2D(size = up_size)(conv8)
    conv9 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(up4)
    
    # add in skip info
    cat4 = concatenate([conv1,conv9],axis = 3)
    conv9 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(cat4)
    conv9 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv9)

    # add on last convolutional layer    
    outputs = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    # define the inputs and outputs
    model = Model(input = inputs,output = outputs)

    # define optimizers
    model.compile(optimizer = Adam(lr = 1e-4),loss = 'binary_crossentropy',metrics = ['accuracy'])
    #model.summary()
    return model

if __name__ == "__main__":
    # just run the funtion to make sure it compiles ok
    unet(2,(2048,1152,3),"relu","he_normal")
