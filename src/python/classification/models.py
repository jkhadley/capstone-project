# Author(s): Joseph Hadley
# Date Created : 2018-07-17
# Date Modified: 2018-07-17
# Description: Create classifier models to use with keras
#----------------------------------------------------------------------------------------------------------------
import os
import numpy as np 
import matplotlib.pyplot as plt 
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate,Input,Dropout
from keras.models import Sequential,Model,load_model
from keras.optimizers import Adam,SGD
from skimage import io

def alexNet(params):
    
    # unpack some of the parameters
    conv_stride = params['conv_stride']    
    
    if os.path.isfile(params['init_w']) == True:
        model = load_model(params['init_w'])
    else:
        # make the model
        
        inputs = Input(params['input_shape'])
        #conv1 = Conv2D(activation = activation,strides = conv_stride,kernel_initializer = init_w)