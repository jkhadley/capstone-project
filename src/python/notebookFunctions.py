# Author(s): Joseph Hadley
# Date Created: 2018-07-01
# Date Modified: 2018-07-25
# Description: Set of functions used to plot the predictions for models and 
    
from keras.models import load_model
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class ModelInferencer():

    def __init__(self,model):
        if isinstance(model,str):
            self.model = load_model(model)
        else:
            self.model = model
        
        self.inputShape = model.inputs[0].shape[1:]
        self.n_classes = model.outputs[0].shape[-1]
        self.pixels = self.inputShape[0]*self.inputShape[1]
        if model.outputs[0].shape[1] > 1:
            self.regression = False
        else:
            self.regression = True

    def segmentationPredict(self,img):
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

                modelPrediction = model.predict(np.expand_dims(croppedArea,axis = 0))
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

                modelPrediction = model.predict(np.expand_dims(croppedArea,axis = 0))
                pred += modelPrediction/(l*w)
        return pred

    def predict(self,img):
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
        self.trainGen = imageSetGenerator(self.dataPath + "/train/",self.classMap,False)
        self.validateGen = imageSetGenerator(self.dataPath + "/validate/",self.classMap,False)

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

        self.trainGen = imageSetGenerator(self.dataPath + "/train/",self.classMap,True)
        self.validateGen = imageSetGenerator(self.dataPath + "/validate/",self.classMap,True)

        # go to train path
        while self.buildBatch(self.validateGen) == True:
            self.batchPredict()
            self.regressionBatchError()

    def regressionBatchError(self):

        for i in range(self.truthClass):
            trueClass = self.truthClass[i]
            self.confMat[trueClass,:] = self.prediction[i]

    def segmentationBatchError(self):

        for i in range(self.truthClass):
            total = 0
            trueClass = self.truthClass[i]
            right = self.pixels - np.sum(np.abs(self.labels[i] - self.prediction[i,:,:,trueClass]))
            total += right

            for j in range(1,self.n_classes):
                if j == trueClass:
                    self.confMat[j,j] += right
                else:
                    tmp = np.sum(self.prediction[i,:,:,j])
                    self.confMat[trueClass,j] += tmp
                    total += tmp

            self.confMat[trueClass,0] += self.pixels - total
    
    def batchPredict(self):
        self.prediction = self.model.predict_on_batch(self.images)

    def buildBatch(self,generator):
        try:
            for batch,i in enumerate(generator):
                
                self.images[i] = batch[0]
                self.labels[i] = batch[1]
                self.truthClass[i] = batch[2]
                
                if i >= self.batch_size:
                    break
            
            return True
        except(StopIteration):
            return False
    
    def setClassMap(self,classMap):
        self.classMap = classMap

    def setBatchSize(self,batch_size):
        self.batch_size = batch_size
    
    def setDataPath(self,path):
        self.dataPath = path    
    
    def getConfusionMatrix(self):
        return self.confMat

def setGenerator(train_path,model,correctClass):
    images = os.listdir(train_path + "/data/")    
    ind = np.random.randint(0,len(images),len(images))  
    inference = ModelInferencer(model)

    for i in ind:
        image = io.imread(train_path + "/data/" + images[i])
        label = io.imread(train_path + "/label/" + images[i])

        prediction = inference.predict(image)
        
        # modify prediction 
        if correctClass != None:
            prediction[prediction == correctClass] = 100
            prediction[prediction != 100] = 0

        yield(image,prediction,label)


def imageSetGenerator(path,classMap,regression):
    classes = list(classMap.keys())

    for c in classes:
        dataPath = path + "/" + c + "/data/"
        labelPath = path + "/" + c +"/label/"

        os.chdir(path)

        subdirs = os.listdir(dataPath)
        
        truthClass = classMap[c] 

        for s in subdirs:
            names = os.listdir(dataPath + s)
            for n in names:
                image = io.imread(dataPath + s + "/" + n)
                label = io.imread(labelPath + s + "/" + n)
                # make label so plant is 1 and ground is zero
                label[label > 0] = 1
                label[label != 1] = 2
                label -= 1

                if regression == True:
                    label = np.sum(label)

                yield(image,label,truthClass)
    
    return

def plotPredictions(params):
    """
    Makes and plots predictions different classes of images.

    Makes predictions for random images for each class specified using the model provided.

    Parameters
    ----------
    num_of_img: int
        Number of images to plot for each class
    model: keras.model OR String
        Keras model or path to keras model to use to make predictions
    path : String
        paths to folder containing classes in the classMap
    classMap : Dictionary (String : Int)
        dictionary of the different folders and the values that the model should predict them to be
    shape : np.array (a x b)
        Input size for model
    fig_height: int
        Defines height of the overall figure
    
    Returns
    -------
    None, Makes a plot showing the outputs for each prediction
    """   
    # load model
    if isinstance(params['model'],str):
        params['model'] = load_model(params['model'])
    
    # initialize generator
    path = params['path']
    classMap = params['classMap']
    classes = list(classMap.keys()) 
    
    numOfClasses = len(classes)

    #initialize figure
    fig, axes = plt.subplots(nrows=params['num_of_img']*numOfClasses, ncols=3, figsize=(20, params['fig_height']))
    # set titles
    axes[0,0].set_title("Original",fontsize = 20)
    axes[0,1].set_title("Prediction",fontsize = 20)
    axes[0,2].set_title("Actual",fontsize = 20)

    for i in range(numOfClasses):
        gen = setGenerator(path + "/" + classes[i],params['model'], classMap[classes[i]])

        for cnt, batch in enumerate(gen):
            if(cnt >= params['num_of_img']):
                break
            else:
                line = cnt + i*params['num_of_img']
                # make plots
                axes[line,0].imshow(batch[0])
                axes[line,0].axis("off")
                axes[line,1].imshow(batch[1,:,:,classes[i]])
                axes[line,1].axis("off")
                axes[line,2].imshow(batch[2])
                axes[line,2].axis("off")
            
    fig.tight_layout()
    plt.show()

def plotEpochAccuracyAndLoss(csv):
    # read in the csv with the results
    df = pd.read_csv(csv)
    # make first plot
    plt.subplot(121)
    plt.plot(df['epoch'],df['Train_Accuracy'], label = "Train")
    plt.plot(df['epoch'],df['Val_Accuracy'],label = "Validation")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    # make secind plot
    plt.subplot(122)
    plt.plot(df['epoch'],df['Train_Loss'], label = "Train")
    plt.plot(df['epoch'],df['Val_Loss'],label = "Validation")
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def plotBatchMetrics(csv):
    df = pd.read_csv(csv)
    plt.subplot(121)
    plt.plot(df['Batch'],df["Accuracy"],label = "Accuracy")
    plt.plot(df['Batch'],df["Recall"],label = "Recall")
    plt.plot(df['Batch'],df["Precision"],label = "Precision")
    plt.plot(df['Batch'],df["F1-Score"],label = "F_1-Score")
    plt.legend()
    plt.title("Accuracy")
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    plt.grid()

    plt.subplot(122)
    plt.plot(df['Batch'],df["Loss"])
    plt.title("Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.grid()

    plt.tight_layout()
    plt.show()

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
    oldModel = "../../model_checkpoints/unet/small_maize_model.hdf5"
    model = load_model(oldModel)
    print(np.shape(model.inputs[0]))
    print("\n")
    print(model.outputs[0].shape)