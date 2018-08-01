# Author(s): Joseph Hadley
# Date Created  : 2018-07-25
# Date Modified : 2018-07-29
# Description   : Generic model creator class to simplify training 
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate,Input,Dropout, Lambda
from keras.models import Sequential,Model,load_model
from keras.losses import mean_squared_error
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from generators import getBatchGenerators
from callbacks import BatchLogger
from metrics import f1Score,recall,precision
import keras.backend as K
import numpy as np
import os

class ModelTrainer():

    def __init__(self,dataPath,resultsPath,modelPath):
        # initialize class variables
        self.model = None
        self.modelPath = modelPath
        self.resultsPath = resultsPath
        self.dataPath = dataPath
        self.saveName = None
        self.classMap = None
        self.className = None
        self.conv_depth = 64
        self.batch_size = 15
        self.input_shape = (256,256,3)
        self.n_classes = 2
        self.metrics = ['acc',recall,precision,f1Score]
        self.init_w = 'zeros'
        self.old_weights = None
        self.loss_function = 'categorical_crossentropy'
        self.optimizer = None
        self.regression = False
        self.trainGen = None
        self.validateGen = None
        self.dropout = 0
        self.batch_log_interval = 10
        self.epochs = 5
        self.steps_per_epoch = 0
        self.validation_steps = 0

    def train(self):
        
        # setup model
        self.createModel()
        self.setGenerators()
        self.buildCallbacks()
        self.printParameters()
        #self.model.summary()
        # train model
        _ = self.model.fit_generator(
                    generator = self.trainGen,
                    validation_data = self.validateGen,
                    steps_per_epoch = self.steps_per_epoch,
                    validation_steps = self.validation_steps,
                    epochs = self.epochs,
                    use_multiprocessing = True,
                    callbacks = self.callbacks)
        # clear save paths to avoid overwriting accidentaly
        self.saveName = None

    def trainMore(self,model):
        self.setOldModel(model)
        self.model.compile(optimizer = self.optimizer,loss=self.loss_function,metrics=self.metrics)
        self.setGenerators()
        self.buildCallbacks()
        self.printParameters()
        # fit model to data
        _ = self.model.fit_generator(
            generator = self.trainGen,
            validation_data = self.validateGen,
            steps_per_epoch = self.steps_per_epoch,
            validation_steps = self.validation_steps,
            epochs = self.epochs,
            use_multiprocessing = True,
            callbacks = self.callbacks)


    def createModel(self):

        outputs,inputs = baseUNet(self.input_shape,self.conv_depth,self.n_classes,self.init_w,self.dropout)
        
        if self.regression == True:
            outputs = Lambda(getPropOfGround)(outputs)
            
        model = Model(inputs = inputs,outputs = outputs)
        model.compile(optimizer = self.optimizer,loss=self.loss_function,metrics=self.metrics)

        if self.old_weights != None:
            model.set_weights(self.old_weights)

        self.model = model

    def buildCallbacks(self):
        model_checkpoint = ModelCheckpoint(self.modelPath + '/' + self.saveName + '.hdf5', monitor='loss',verbose=1)
        logger = BatchLogger(self.resultsPath + '/' + self.saveName + "_batch.csv",self.resultsPath + '/' + self.saveName + "_epoch.csv",self.batch_log_interval)
        self.callbacks = [model_checkpoint,logger]

    def setSaveName(self,name):
        self.saveName = name

    def setOldModel(self,model):
        oldModel = load_model(model,custom_objects={'recall': recall,'precision': precision,'f1Score':f1Score})
        self.old_weights = oldModel.get_weights()
        self.input_shape = oldModel.inputs[0].shape[1:]
        self.n_classes = oldModel.outputs[0].shape[-1]
        self.conv_depth = oldModel.layers[1].output_shape[-1]
        self.model = oldModel

    def setRegression(self):
        self.regression = True
        self.loss_function = mse

    def setSegmentation(self):
        self.regression = False
    
    def setClassMap(self,classMap):
        self.classMap = classMap
        self.n_classes = len(np.unique(list(classMap.values()))) + 1
        # find the number of images in the data set
        dirs = list(classMap.keys())
        train_size = dirSize(self.dataPath + "train",dirs)
        validate_size = dirSize(self.dataPath + "validate",dirs)
        # set steps per epochs
        self.steps_per_epoch = round(train_size/self.batch_size)
        self.validation_steps = round(validate_size/self.batch_size)

    def setClassName(self,whichDir):
        self.className = whichDir
        self.setClassMap({whichDir : 1})

    def setOptimizerParams(self,lr,momentum,decay):
        self.optimizer = SGD(lr=lr,momentum=momentum,decay=decay)

    def setWeightInitializer(self,weights):
        self.init_w = weights

    def setGenerators(self):
        shape = (self.input_shape[0],self.input_shape[1])
        self.trainGen,self.validateGen = getBatchGenerators(self.batch_size,self.dataPath,shape,self.classMap,self.regression)
    
    def changeModelSavePath(self,path):
        self.modelPath = path
    
    def changeDropout(self,dropout):
        self.dropout = dropout

    def changeResultsSavePath(self,path):
        self.resultsPath = path

    def changeDataPath(self,path):
        self.dataPath = path
    
    def changeInputShape(self,shape):
        self.input_shape = shape

    def changeLossFunction(self,loss):
        self.loss_function = loss
        
    def changeBatchLogInterval(self,interval):
        self.batch_log_interval = interval
    
    def changeConvolutionalDepth(self,depth):
        self.conv_depth = depth

    def changeMetrics(self, metrics):
        if isinstance(metrics,list) == False:
            metrics = [metrics]
        self.metrics = metrics
      
    def changeBatchSize(self,batch_size):
        self.batch_size = batch_size
    
    def changeEpochs(self,epochs):
        self.epochs = epochs

    def printParameters(self):
        print("----------Model Parameters----------")
        print("Initial Conv. Depth : " + str(self.conv_depth))
        print("Number of Classes   : " + str(self.n_classes))
        print("Dropout             : " + str(self.dropout))
        print("Activation Function : Relu")
        print("Input Shape         : " + str(self.input_shape))
        print("Batch Size          : " + str(self.batch_size))
        print("--------Optimizer Parameters--------")
        print("Learning Rate : " + str(self.optimizer.lr))
        print("Momentum      : " + str(self.optimizer.momentum))
        print("Initial Decay : " + str(self.optimizer.initial_decay))

def getPropOfGround(x):
    return K.sum(K.sum(x,axis = 1),axis = 1)

def mse(y_true,y_pred):
    return K.mean(K.square(y_true - y_pred))

def baseUNet(input_shape,conv_depth,n_classes,init_w,dropout):
    inputs = Input(input_shape)

    conv1 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(inputs)
    conv1 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(conv1)

    # pool down to next layer
    pool1 = MaxPooling2D((2,2),strides = (2,2))(conv1)

    conv_depth *= 2

    # convolute down again
    conv2 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(pool1)
    conv2 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(conv2)
    
    # pool down again
    pool2 = MaxPooling2D((2,2),strides = (2,2))(conv2)

    conv_depth *= 2 

    # Convolution
    conv3 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(pool2)
    conv3 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(conv3)
    
    # pool down
    pool3 = MaxPooling2D((2,2),strides = (2,2))(conv3)

    conv_depth *= 2 
    # Convolution
    conv4 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(pool3)
    conv4 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(conv4)
    
    # pool down 
    pool4 = MaxPooling2D((2,2),strides = (2,2))(conv4)

    conv_depth *=2 

    # Convolution
    conv5 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(pool4)
    conv5 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(conv5)
    drop = Dropout(dropout)(conv5)

    conv_depth /= 2
    conv_depth = int(conv_depth)  
    # do upsampling
    up1 = UpSampling2D(size = (2,2))(drop)
    conv6 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(up1)
    
    # add in skip info
    cat1 = concatenate([conv4,conv6],axis = 3)
    conv6 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(cat1)
    conv6 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(conv6)

    conv_depth /= 2
    conv_depth = int(conv_depth)
    # do upsampling
    up2 = UpSampling2D(size = (2,2))(conv6)
    conv7 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(up2)
    
    # add in skip info
    cat2 = concatenate([conv3,conv7],axis = 3)
    conv7 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(cat2)
    conv7 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(conv7)
    
    conv_depth /= 2
    conv_depth = int(conv_depth)
    # do upsampling
    up3 = UpSampling2D(size = (2,2))(conv7)
    conv8 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(up3)
    
    # add in skip info
    cat3 = concatenate([conv2,conv8],axis = 3)
    conv8 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(cat3)
    conv8 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(conv8)
    
    conv_depth /= 2
    conv_depth = int(conv_depth)
    # do upsampling
    up4 = UpSampling2D(size = (2,2))(conv8)
    conv9 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(up4)
    
    # add in skip info
    cat4 = concatenate([conv1,conv9],axis = 3)
    conv9 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(cat4)
    conv9 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(conv9)

    outputs = Conv2D(n_classes, 1, activation = 'softmax')(conv9)

    return outputs,inputs

def dirSize(path,dirs):
    oldDir = os.getcwd()
    os.chdir(path)
    cwd = os.getcwd()
    length = 0
    if isinstance(dirs,str):
        dirs = [dirs]
    for d in dirs:
        path = cwd + "/" + d + "/data/"
        subdirs = os.listdir(path)
        for s in subdirs:
            length += len(os.listdir(path + s))
    
    os.chdir(oldDir)
    return length

if __name__ == "__main__":
    modelPath = "C://Users//jkhad//Documents//2018summer//eas560//model_checkpoints//unet"
    resultsPath = "C://Users//jkhad//Documents//2018summer//eas560//model_checkpoints//unet"
    dataPath  = "C://Users//jkhad//Documents//2018summer//eas560//data//groundcover2016"

    modelTrainer = ModelTrainer(dataPath,resultsPath,modelPath)

    modelTrainer.setOptimizerParams(lr = 1.0 * (10**-3),momentum = 0.9,decay = 1.0 * (10**-6))
    modelTrainer.setClassName = "mungbean"