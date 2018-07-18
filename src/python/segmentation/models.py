# Author(s): Joseph Hadley
# Date Created : 2018-06-12
# Date Modified: 2018-06-30
# Description: Create segmentation models to use for keras
#----------------------------------------------------------------------------------------------------------------
import os
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate,Input,Dropout
from keras.models import Sequential,Model,load_model
from keras.optimizers import Adam,SGD
#----------------------------------------------------------------------------------------------------------------
#                                      Implement U-net Architecture
#----------------------------------------------------------------------------------------------------------------
def unet(params):
    # unpack paramaters
    conv_depth = params['conv_depth']
    dropout = params["dropout"]
    activation = params['activation']
    init_w = params['init_w']

    # check if initial weights is a file try and load in the parameters
    if os.path.isfile(params['init_w']) == True:
        model = load_model(params['init_w'])
    else:

        # Training variables to tweak
        conv_kernel_size = (3,3)
        conv_stride = (1,1)
        pool_size = (2,2)
        pool_stride = (2,2)
        pad = "same"
        up_size = (2,2) 

        # first two convolutional layers
        inputs = Input(params['input_shape'])
        conv1 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(inputs)
        conv1 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv1)

        # pool down to next layer
        pool1 = MaxPooling2D(pool_size = pool_size,strides = pool_stride)(conv1)

        conv_depth *= 2 

        # convolute down again
        conv2 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(pool1)
        conv2 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv2)
        
        # pool down again
        pool2 = MaxPooling2D(pool_size = pool_size,strides = pool_stride)(conv2)

        conv_depth *= 2 

        # Convolution
        conv3 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(pool2)
        conv3 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv3)
        
        # pool down
        pool3 = MaxPooling2D(pool_size = pool_size,strides = pool_stride)(conv3)

        conv_depth *= 2 
        # Convolution
        conv4 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(pool3)
        conv4 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv4)
        
        # pool down 
        pool4 = MaxPooling2D(pool_size = pool_size,strides = pool_stride)(conv4)

        conv_depth *=2 

        # Convolution
        conv5 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(pool4)
        conv5 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv5)
        drop = Dropout(dropout)(conv5)


        conv_depth /= 2
        conv_depth = int(conv_depth)  
        # do upsampling
        up1 = UpSampling2D(size = up_size)(drop)
        conv6 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(up1)
        
        # add in skip info
        cat1 = concatenate([conv4,conv6],axis = 3)
        conv6 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(cat1)
        conv6 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv6)

        conv_depth /= 2
        conv_depth = int(conv_depth)
        # do upsampling
        up2 = UpSampling2D(size = up_size)(conv6)
        conv7 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(up2)
        
        # add in skip info
        cat2 = concatenate([conv3,conv7],axis = 3)
        conv7 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(cat2)
        conv7 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv7)
        
        conv_depth /= 2
        conv_depth = int(conv_depth)
        # do upsampling
        up3 = UpSampling2D(size = up_size)(conv7)
        conv8 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(up3)
        
        # add in skip info
        cat3 = concatenate([conv2,conv8],axis = 3)
        conv8 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(cat3)
        conv8 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv8)
        
        conv_depth /= 2
        conv_depth = int(conv_depth)
        # do upsampling
        up4 = UpSampling2D(size = up_size)(conv8)
        conv9 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(up4)
        
        # add in skip info
        cat4 = concatenate([conv1,conv9],axis = 3)
        conv9 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(cat4)
        conv9 = Conv2D(conv_depth,activation = activation,kernel_size = conv_kernel_size,strides = conv_stride,padding = pad,kernel_initializer = init_w)(conv9)

        # add on last convolutional layer    
        outputs = Conv2D(params['n_classes'], 1, activation = params['output_activation'])(conv9)

        # define the inputs and outputs
        model = Model(input = inputs,output = outputs)

        # define optimizers  
        if params['opimizer'] == "adam":
            optimizer = Adam(lr = params['lr'])
        elif params['optimizer'] == 'sgd':
            optimizer = SGD(lr=params['lr'])
        else:
            print('Optimizer not set up yet!')

        # define the model optimizer and loss function            
        model.compile(optimizer = optimizer, loss =params['loss'], metrics = ['accuracy'])

    # print the summary of the model
    # model.summary()
    return model
#------------------------------------------------------------------------------------------------------------------------
#                                               Implement DeepLab Architecture
#------------------------------------------------------------------------------------------------------------------------
def deepLab(params):
    pass

if __name__ == "__main__":
    params = {
        'n_classes' : 2,
        'lr' : 0.01,
        'dropout' : 0.5,
        'conv_depth' : 2,
        'input_shape' : (2048,1152,3),
        'activation' : "relu",
        'init_w' : "he_normal"
    }
    # just run the funtion to make sure it compiles ok
    unet(params)

