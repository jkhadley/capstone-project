from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input, Dropout, Lambda
from keras.losses import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from generators import getBatchGenerators
from callbacks import BatchLogger
from metrics import f1Score, recall, precision, RMSE
from misc import dirSize
import keras.backend as K
import numpy as np
import os


class ModelTrainer():
    """Object to contain the model parameters and train the model."""
    def __init__(self,dataPath,resultsPath,modelPath):
        """Initializes class variables.
        
        Parameters
        ----------
        dataPath : String
            Path to the base directory of the data classes
        resultsPath : String
            Path to where the results csv files should be written
        modelPath : String 
            Path to where the models are stored
        
        Attributes
        ----------
        conv_depth : int, (defualt is 64)
            Depth of initial Convolutional layer
        batch_size : int, (default is 15)
            Number of images to load and train with before updating weights
        """
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
        self.metrics = ['acc']
        self.init_w = 'zeros'
        self.old_weights = None
        self.loss_function = 'categorical_crossentropy'
        self.optimizer = None
        self.regression = False
        self.trainGen = None
        self.validateGen = None
        self.pixels = 256*256
        self.dropout = 0
        self.batch_log_interval = 10
        self.epochs = 5
        self.steps_per_epoch = 0
        self.validation_steps = 0
        self.train_size = 0
        self.validate_size = 0

    def train(self):
        """Trains the model specified by the parameters.

        Creates a model and generators based on the specified 
        parameters and then trains it. It will save the outputs 
        according to callback information that is specified.
        """
        # setup model
        self.createModel()
        self.setGenerators()
        self.buildCallbacks()
        self.printParameters()
        
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
    
    def evaluate(self):
        """Evaluates the model on the training and validation data. 

        Evaluates the trained model that is loaded through the setOldModel method.
        """
        # setup model
        self.optimizer = SGD(lr = 0,momentum=0,decay = 0)
        self.createModel()
        self.setGenerators()
        self.printParameters()
        output = {}

        trainOutput = self.model.evaluate_generator(
            generator = self.trainGen,
            steps=self.steps_per_epoch,
            use_multiprocessing=True,
            verbose=1
        )      
        output['loss'] = trainOutput[0]
        for i in range(len(self.metricsAsString)):
            output[self.metricsAsString[i]] = trainOutput[i+1]


        validationOutput = self.model.evaluate_generator(
            generator = self.validateGen,
            steps=self.validation_steps, 
            use_multiprocessing=True, 
            verbose=1
        )
        output['val_loss'] = validationOutput[0]
        for i in range(len(self.metricsAsString)):
            output["val_" + self.metricsAsString[i]] = validationOutput[i+1]
        
        print("loss     : " + str(output['loss']))
        for i in range(len(self.metricsAsString)):
            tmp = self.metricsAsString[i] 
            print(tmp + " : " + str(output[tmp]))
        print("val_loss : " + str(output['val_loss']))
        for i in range(len(self.metricsAsString)):
            tmp = "val_" + self.metricsAsString[i] 
            print(tmp + " : " + str(output[tmp]))

        
    def continueTraining(self,model):
        """Further trains the specified model.""" 
        self.setOldModel(model)
        self.model.compile(optimizer = self.optimizer,
                        loss=self.loss_function,
                        metrics=self.metrics)
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
        """Creates a U-net model based on the specified parameters.
        
        If the model is not set to a regression model, the output has the same 
        depth and width as the input and as many layers as the number of 
        classes. If the model is set to regression, the output is an array that
        contains the proportion of the image that the class is.
        """
        outputs, inputs = baseUNet(self.input_shape,
                                self.conv_depth,
                                self.n_classes,
                                self.init_w,
                                self.dropout)
        
        if self.regression == True:
            outputs = Lambda(getPropOfGround)(outputs)
            
        model = Model(inputs = inputs,outputs = outputs)
        model.compile(optimizer = self.optimizer,loss=self.loss_function,metrics=self.metrics)

        if self.old_weights != None:
            model.set_weights(self.old_weights)
        self.model = model
    
    def singlePrediction(self,img):
        """Make a prediction using the loaded model on a single image.
        
        Parameters
        ----------
        img : np.array
            Image to make prediction on
        """
        self.optimizer = SGD(lr = 0,momentum=0,decay = 0)
        self.createModel()
        output = self.model.predict(np.expand_dims(img,axis = 0))
        return output

    def buildCallbacks(self):
        """Builds the callbacks that save the model weights and results.

        Saves the model checkpoint and logger to the paths specified by
        modelPath and resultsPath, and then gives them the names 
        specified by saveName.
        """
        model_checkpoint = ModelCheckpoint(self.modelPath + '/' + self.saveName + '.hdf5',
                                monitor='loss',verbose=1)

        logger = BatchLogger(self.resultsPath + '/' + self.saveName + "_batch.csv",
                        self.resultsPath + '/' + self.saveName + "_epoch.csv",
                        self.batch_log_interval,
                        self.metricsAsString)

        self.callbacks = [model_checkpoint,logger]

    def setSaveName(self,name):
        """Sets the name to save the results and model weights with."""
        self.saveName = name

    def setOldModel(self,model):
        """Gets the model parameters from the specified model.

        Gets the weights, input shape, and number of classes from the 
        old model to load into the new model to do more training or 
        switch model type.

        Parameters
        ----------
        model: String
            Path to the old keras model object to be loaded 
        """
        self.modelName = model
        oldModel = load_model(self.modelPath + "/" + model + ".hdf5",custom_objects={'recall': recall,
                                                                        'precision': precision,
                                                                        'f1Score':f1Score})
        self.old_weights = oldModel.get_weights()
        self.input_shape = oldModel.inputs[0].get_shape().as_list()[1:]
        self.n_classes = oldModel.outputs[0].get_shape().as_list()[-1]
        self.conv_depth = oldModel.layers[1].output_shape[-1]
        self.model = oldModel

    def setRegression(self):
        """Set the model to a regression model.
        
        Set the model to a regression model and changes the loss 
        function to MSE.
        """
        self.regression = True
        self.loss_function = mean_squared_error

    def setSegmentation(self):
        """Set the model to a segmentation model.
        
        Sets the model to segmentation and changes the loss function to
        categorical cross-entropy.
        """
        self.regression = False
        self.loss_function = "categorical_crossentropy"
    
    def setClassMap(self,classMap):
        """Set the class map that specifies which directory corresponds to which class.

        Parameters
        ----------
        classMap : dictionary
            Mapping of directories to correct output
        """
        self.classMap = classMap
        self.n_classes = len(np.unique(list(classMap.values()))) + 1
        # find the number of images in the data set
        dirs = list(classMap.keys())
        self.train_size = dirSize(self.dataPath + "train",dirs)
        self.validate_size = dirSize(self.dataPath + "validate",dirs)
        # set steps per epochs
        self.steps_per_epoch = round(self.train_size/self.batch_size)
        self.validation_steps = round(self.validate_size/self.batch_size)

    def setClassName(self,whichDir):
        """Specify the single directory to use on the dataPath.

        Parameters
        ----------
        whichDir: String
            Name of the directory to be used for training
        """
        self.className = whichDir
        self.setClassMap({whichDir : 1})

    def setOptimizerParams(self,lr,momentum,decay):
        """Set the SGD Optimizer parameters used to change the weights.
        
        Parameters
        ----------
        lr : float [0 ->1]
            Learning rate for SGD
        momentum : float [0->1]
            Momentum for SGD
        decay : float[0->1]
            Weight decay for SGD
        """
        self.optimizer = SGD(lr=lr,momentum=momentum,decay=decay)

    def setWeightInitializer(self,weights):
        """Set the weight initializer to use for model initialization.

        Parameters
        ----------
        weights: String
            Weight initializer to use to intialize model with
        """
        self.init_w = weights

    def setGenerators(self):
        """Create the training and validation data generators.

        Uses the batch_size, classMap, and regression parameters to 
        create generators that will generate the appropriate data. 
        """
        shape = (self.input_shape[0],self.input_shape[1])
        self.trainGen,self.validateGen = getBatchGenerators(self.batch_size,
                                                            self.dataPath,
                                                            shape,
                                                            self.classMap,
                                                            self.regression)
    
    def changeModelSavePath(self,path):
        """Change the path that the model is saved to.

        Parameters
        ----------
        path : String
            Path to save the model to
        """
        self.modelPath = path
    
    def changeDropout(self,dropout):
        """Change the dropout for the model.

        Parameters
        ----------
        dropout: float [0->1]
            Proportion of nodes to randomly drop each batch update. 
        """
        self.dropout = dropout

    def changeResultsSavePath(self,path):
        """Change where the logger results are saved to.

        Parameters
        ----------
        path : String
            Path to save the logger results to 
        """
        self.resultsPath = path

    def changeDataPath(self,path):
        """Change the directory to look for the data in.

        Parameters
        ----------
        path : String
            Base directory that the data is located at 
        """
        self.dataPath = path
    
    def changeInputShape(self,shape):
        """Change the Input shape that the model should use.

        Parameters
        ----------
        shape : tuple
            Input shape for the model
        """
        self.input_shape = shape

    def changeLossFunction(self,loss):
        """Change the Loss Function that changes the model weights.

        Parameters
        ----------
        loss : int
            The loss function to evaluate the model with
        """
        self.loss_function = loss
        
    def changeBatchLogInterval(self,interval):
        """Change the interval that the batches are logged at.

        Parameters
        ----------
        interval : int
            Interval that batches will be logged at
        """
        self.batch_log_interval = interval
    
    def changeConvolutionalDepth(self,depth):
        """Change the depth of the initial convolutional layers that
        are used in the model.

        Parameters
        -----------
        depth : int
            Depth of the first convolutional layer
        """
        self.conv_depth = depth

    def changeMetrics(self, metrics):
        """Changes the metrics that will be used to evauluate the 
        model.
        
        Parameters
        ----------
        metrics : list
            List of metrics that will be used to evaluate the model
        """
        if isinstance(metrics,list) == False:
            metrics = [metrics]
        self.metrics = metrics

        whatMetrics = []

        for i in metrics:
            if i == RMSE:
                whatMetrics.append("RMSE")
            elif i == f1Score:
                whatMetrics.append("f1Score")
            elif i == recall:
                whatMetrics.append("recall")
            elif i == precision:
                whatMetrics.append("precision")
            elif i == mean_squared_error:
                whatMetrics.append("mean_squared_error")
            elif i == mean_absolute_error:
                whatMetrics.append("mean_absolute_error")
            elif i == mean_absolute_percentage_error:
                whatMetrics.append("mean_absolute_percentage_error")
            elif isinstance(i,str):
                whatMetrics.append(i)
            else:
                print("I don't know what to do with : " + str(i))

        self.metricsAsString = whatMetrics
      
    def changeBatchSize(self,batch_size):
        """Changes the batch size of the batches that the model will
        be trained on.
        
        Parameters
        ----------
        batch_size : int
            Number of sets of images to train on before updating the 
            weights.
        """
        self.batch_size = batch_size
        self.steps_per_epoch = round(self.train_size/self.batch_size)
        self.validation_steps = round(self.validate_size/self.batch_size)
    
    def changeEpochs(self,epochs):
        """ Changes the number of epochs that the model will train for.
        
        Parameters
        ----------
        epochs : int
            Number of times the model will see all of the data
        """
        self.epochs = epochs

    def printParameters(self):
        """Prints the model parameters."""
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

def baseUNet(input_shape,conv_depth,n_classes,init_w,dropout):
    """Creates a basic U-net segmentation model.

    Parameters
    ----------
    input_shape : tuple
        Size of the input that the model should accept
    conv_depth : int
        Depth of the first convolutional layer
    n_classes : int
        Number of classes that the model should predict
    init_w : String
        Weight initializer to use for the nodes
    dropout : float [0->1]
        Proportion of the middle convolutional layer to randomly ignore
        each training iteration

    Returns
    -------
    outputs : keras functional model
        output layer to compile the model 
    inputs : keras layer
        input layer to compile the model

    """
    inputs = Input(input_shape)

    c1=Conv2D(conv_depth,(3,3),activation='relu',padding='same',kernel_initializer=init_w)(inputs)
    c1=Conv2D(conv_depth,(3,3),activation='relu',padding="same",kernel_initializer=init_w)(c1)

    # pool down to next layer
    pool1 = MaxPooling2D((2,2),strides = (2,2))(c1)

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
    cat4 = concatenate([c1,conv9],axis = 3)
    conv9 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(cat4)
    conv9 = Conv2D(conv_depth,activation = 'relu',kernel_size = (3,3),strides = (1,1),padding = "same",kernel_initializer=init_w)(conv9)

    outputs = Conv2D(n_classes, 1, activation = 'softmax')(conv9)

    return outputs,inputs

# functions for determining the regression output
def getPropOfGround(x):
    """Finds and returns the proportion of the ground for each class."""
    return K.sum(K.sum(x,axis = 1),axis = 1)/65536

