from keras.models import load_model
from skimage import io
import matplotlib.pyplot as plt
from metrics import f1Score,recall,precision
from misc import dirSize
import pandas as pd
import numpy as np
import os
import sys

class ModelInferencer():
    """Class to inference the model and make predictions."""
    def __init__(self,model,dataPath):
        """Load the model and determine various parameters.
        
        Parameters
        ----------
        model : String OR keras model
            Model to inference 
        dataPath : String
            Base directory of where the data is
        """
        if isinstance(model,str):
            self.model = load_model(model,custom_objects={'recall': recall,
                                                        'precision': precision,
                                                        'f1Score':f1Score})
        else:
            self.model = model
        
        self.dataPath = dataPath

        # find the input shape, number of classes, and number of pixels from the layers
        self.inputShape = self.model.inputs[0].get_shape().as_list()[1:]
        self.n_classes = self.model.outputs[0].shape[-1]
        self.pixels = self.inputShape[0]*self.inputShape[1]

        if self.model.outputs[0].shape[1] > 1:
            self.regression = False
        else:
            self.regression = True

    def segmentationPredict(self,img):
        """Makes a segmentation prediction for image of any size.

        Breaks the image into multiple pieces that fit into the model, 
        makes predictions for each piece, and then stitches the 
        predictions back together.

        Parameters
        ----------
        img : np.array
            Image to make prediction on
        
        Returns
        -------
        pred : np.array
            Segmentation prediction on the image
        """
        # get the size of the input image
        l,w,_ = np.shape(img)
        # init array for new image
        pred = np.zeros(shape = (l,w))

        r = l//self.inputShape[0]
        c = w//self.inputShape[1]

        roffset = 0
        coffset = 0
        
        # if image size isnt cleanly divisible by window size, use another window
        if l%self.inputShape[0] != 0:
            roffset = 1
        if w%self.inputShape[1] != 0:
            coffset = 1
    
        x1 = 0

        # Crop the image
        for j in range(r + roffset):
            y1 = 0
            x2 = (j+1)*self.inputShape[0] 

            for k in range(c + coffset):
                # find upper bounds of window
                y2 = (k+1)*self.inputShape[1] 
                
                # check if outer dimension is larger than image size and adjust
                if y2 > w or x2 > l:
                    croppedArea = np.zeros(self.inputShape)
                    if y2 > w and x2 > l:             
                        croppedArea[0:(x2-l),0:(y2-w)] = img[x1:,y1:]
                    elif y2 > w and x2 < l:
                        croppedArea[:,0:(y2-w)] = img[x1:x2,y1:]
                    elif x2 > l and y2 < w:
                        croppedArea[0:(x2-l),:] = img[x1:,y1:y2]                
                else:
                    # crop area of picture
                    croppedArea = img[x1:x2,y1:y2]
                    # make prediction using model

                modelPrediction = self.model.predict(np.expand_dims(croppedArea,axis = 0))
                # check if in bounds
                if y2 > w or x2 > l:
                    if y2 > w and x2 > l:             
                        pred[x1:,y1:] = modelPrediction[0,0:(x2-l),0:(y2-w),0]
                    elif y2 > w and x2 < l:
                        pred[x1:x2,y1:] = modelPrediction[0,:,0:(y2-w),0]
                    elif x2 > l and y2 < w:
                        pred[x1:,y1:y2] = modelPrediction[0,0:(x2-l),:,0] 
                else:
                    # update prediction image
                    pred[x1:x2,y1:y2] = modelPrediction[0,:,:,0]
            
                # update the bounds
                y1 = y2
            # update the lower x bound
            x1 = x2 
        return pred

    def regressionPredict(self,img):
        """Makes a regression prediction for image of any size.

        Breaks the image into multiple pieces that fit into the model, 
        makes predictions for each piece, and then sums the predictions
        for each piece.

        Parameters
        ----------
        img : np.array
            Image to make prediction on
        
        Returns
        -------
        pred : np.array
            Regression prediction on the image
        """
        # get the size of the input image
        l,w,_ = np.shape(img)
        # init array for new image
        pred = np.zeros(self.n_classes)

        r = l//self.inputShape[0]
        c = w//self.inputShape[1]

        roffset = 0
        coffset = 0
        
        # if image size isnt cleanly divisible by window size, use another window
        if l%self.inputShape[0] != 0:
            roffset = 1
        if w%self.inputShape[1] != 0:
            coffset = 1
    
        x1 = 0
        # Crop the image
        for j in range(r + roffset):
            y1 = 0
            x2 = (j+1)*self.inputShape[0] 

            for k in range(c + coffset):
                # find upper bounds of window
                y2 = (k+1)*self.inputShape[1] 
                
                # check if outer dimension is larger than image size and adjust
                if y2 > w or x2 > l:
                    croppedArea = np.zeros(self.inputShape)
                    if y2 > w and x2 > l:             
                        croppedArea[0:(x2-l),0:(y2-w)] = img[x1:,y1:]
                    elif y2 > w and x2 < l:
                        croppedArea[:,0:(y2-w)] = img[x1:x2,y1:]
                    elif x2 > l and y2 < w:
                        croppedArea[0:(x2-l),:] = img[x1:,y1:y2]                
                else:
                    # crop area of picture
                    croppedArea = img[x1:x2,y1:y2]
                    # make prediction using model

                modelPrediction = self.model.predict(np.expand_dims(croppedArea,axis = 0))
                output = np.sum(np.sum(modelPrediction,axis = 1),axis = 1)
                pred += output[0]/(l*w)
        return pred

    def predict(self,img):
        """Makes appropriate prediction for image based on model type.

        Parameters
        ----------
        img : np.array
            Image to make prediction on
        
        Returns
        -------
        pred : np.array
            prediction on the image
        """
        if self.regression == True:
            output = self.regressionPredict(img)
        else:
            output = self.segmentationPredict(img)

        return output

    def getSegmentationAccuracy(self):
        s1 = (self.batch_size,self.inputShape[0],self.inputShape[1],self.inputShape[2])
        s2 = (self.batch_size,self.inputShape[0],self.inputShape[1],self.n_classes)
        
        self.confMat = np.zeros((self.n_classes,self.n_classes))

        self.images = np.zeros(s1)
        self.labels = np.zeros(s2)
        self.truthClass = np.zeros(self.batch_size)

        # create generators
        self.trainGen = self.imageSetGenerator("train")
        self.validateGen = self.imageSetGenerator("validate")

        # go to train path
        while self.buildBatch(self.validateGen) == True:
            self.batchPredict()
            self.segmentationBatchError()

    def getRegressionAccuracy(self):
        s1 = (self.batch_size,self.inputShape[0],self.inputShape[1],self.inputShape[2])
        s2 = (self.batch_size,self.n_classes)

        self.confMat = np.zeros((self.n_classes,self.n_classes))

        self.images = np.zeros(s1)
        self.labels = np.zeros(s2)
        self.truthClass = np.zeros(self.batch_size)

        self.trainGen = self.imageSetGenerator("train")
        self.validateGen = self.imageSetGenerator("validate")

        # go to train path
        while self.buildBatch(self.validateGen) == True:
            self.batchPredict()
            self.regressionBatchError()

    def regressionBatchError(self):

        for i in range(len(self.truthClass)):
            trueClass = self.truthClass[i]
            self.confMat[trueClass,:] = self.prediction[i]

    def segmentationBatchError(self):

        for i in range(len(self.truthClass)):
            total = 0
            trueClass = int(self.truthClass[i])
            
            # get the ground error
            tmp = self.labels[i,:,:,0] - self.prediction[i,:,:,0]


            for j in range(1,self.n_classes):
                if j == trueClass:
                    a = np.sum(np.abs(self.labels[i,:,:,trueClass] - self.prediction[i,:,:,trueClass]))

                    self.confMat[j,j] += self.pixels - a
                else:
                    tmp = np.sum(self.prediction[i,:,:,j])
                    self.confMat[trueClass,j] += tmp
                    total += tmp

    def batchPredict(self):
        """Makes prediction on the batch that was loaded via a generator."""
        self.prediction = self.model.predict(self.images)

    def buildBatch(self,generator):
        """Builds a batch of images to make predictions on.

        Parameters
        ----------
        generator : generator
            Generator to generator the images and labels with
        """
        try:
            for i,batch in enumerate(generator):
                
                if i >= self.batch_size:
                    break

                self.images[i] = batch[0]
                self.labels[i] = batch[1]
                self.truthClass[i] = batch[2]
            
            return True
        except(StopIteration):
            return False
    
    def setClassMap(self,classMap):
        """Map the directories in the data path to the appropriate class.

        Parameters
        ----------
        classMap : dictionary
            Dictionary that maps the class to the correct output 
        """
        self.classMap = classMap

    def setBatchSize(self,batch_size):
        """Set the prediction batch size.

        Parameters
        ----------
        batch_size : int
            Number of images per batch
        """ 
        self.batch_size = batch_size
       
    def getConfusionMatrix(self):
        return self.confMat

    def imageSetGenerator(self,whichSet):
        """Generates an image, its label, and the class that it comes from.
        Parameters
        ----------
        whichSet : String
            Which directory to get image from

        Yields
        ------
        image : np.array
            Image to make prediction on
        label : np.array
            Ground Truth for image
        truthClass : int
            The class that the image belongs 
        """
        classes = list(self.classMap.keys())

        for c in classes:
            dataPath = self.dataPath + "/" + whichSet +"/" + c + "/data/"
            labelPath = self.dataPath + "/" + whichSet +"/" + c +"/label/"

            subdirs = os.listdir(dataPath)
            label = np.zeros((self.inputShape[0],self.inputShape[1],self.n_classes))
            truthClass = self.classMap[c] 

            for s in subdirs:
                names = os.listdir(dataPath + s)
                for n in names:
                    image = io.imread(dataPath + s + "/" + n)
                    tmpLabel = io.imread(labelPath + s + "/" + n)
                    # normaize image and label and send out
                    if(np.max(image) > 1):
                        image = image/255

                    label[:,:,1] = (tmpLabel < 1)
                    label[:,:,0] = (tmpLabel >= 1)
    
                    if self.regression == True:
                        label = np.sum(np.sum(label,axis = 0),axis = 0)/self.pixels

                    yield(image,label,truthClass)
        return