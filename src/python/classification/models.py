# Author(s): Joseph Hadley
# Date Created : 2018-07-17
# Date Modified: 2018-07-17
# Description: Create classifier models to use with keras
#----------------------------------------------------------------------------------------------------------------
import os
import numpy as np 
import matplotlib.pyplot as plt 
from keras.layers import Conv2D, MaxPooling2D, concatenate,Input,Dropout,Dense
from keras.models import Sequential,Model,load_model
from keras.optimizers import Adam,SGD
from skimage import io

def alexNet(params):
    
    # unpack some of the parameters
    activation = params['activation']

    if os.path.isfile(params['init_w']) == True:
        model = load_model(params['init_w'])
    else:
        init_w = params['init_w']
        # make the model
        depths = [3,96,256,384,384,256]
        
        inputs = Input((256,256,3))
        conv1 = Conv2D(depths[0],  kernel_size = (12,12),strides = (4,4),activation = activation,kernel_initializer = init_w)(inputs)
        conv2 = Conv2D(depths[1], kernel_size = (5,5)  ,strides = (1,1),activation = activation,kernel_initializer = init_w)(conv1)
        
        pool1 = MaxPooling2D(pool_size = (5,5),strides = (2,2))(conv2)
        
        conv3 = Conv2D(depths[2],kernel_size = (3,3)  ,strides = (1,1),activation = activation,padding = "same",kernel_initializer = init_w)(pool1)
        
        pool2 = MaxPooling2D(pool_size = (3,3),strides = (2,2))(conv3)
        # add final convolutional layers
        conv4 = Conv2D(depths[3],kernel_size = (3,3)  ,strides = (1,1),activation = activation,padding = "same",kernel_initializer = init_w)(pool2)
        conv5 = Conv2D(depths[4],kernel_size = (3,3)  ,strides = (1,1),activation = activation,padding = "same",kernel_initializer = init_w)(conv4)
        conv6 = Conv2D(depths[5],kernel_size = (3,3)  ,strides = (1,1),activation = activation,padding = "same",kernel_initializer = init_w)(conv5)

        # add dense layers
        dense1 = Dense(params['fc_size'],kernel_initializer = init_w)(conv6)
        dense2 = Dense(params['fc_size'],kernel_initializer = init_w)(dense1)
        outputs = Dense(params["num_of_classes"],kernel_initializer = init_w,activation= params['output_activation'])(dense2)

        # define the inputs and outputs
        model = Model(input = inputs,output = outputs)

        # define optimizers  
        optimizer = SGD(lr=params['lr'],momentum = params['momentum'])

        # define the model optimizer and loss function            
        model.compile(optimizer = optimizer, loss = params['loss'], metrics = ['accuracy'])
    
    return model

if __name__ == "__main__":
    params = {
        'init_w' : 'he_normal',
        'lr' : 0.01,
        'activation' :  "relu",
        'loss' : 'categorical_crossentropy',
        'num_of_classes' : 4,
        'output_activation' : "softmax",
        'dropout' : 0.5,
        'momentum': 0,
        'fc_size': 32
    }

    model = alexNet(params)
    model.summary()