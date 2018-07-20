# Author(s): Joseph Hadley
# Date Created: 2018-07-01
# Date Modified: 2018-07-17
# Description: Set of functions used to plot the predictions for models and 
    
from keras.models import load_model
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def getTrainPredictions(img,subImgSize,model):
    """
    Makes a prediction for an image

    Takes an input of any size, crops it to specified size, makes predictions for each cropped 
        window, and stitches output together

    Parameters
    ----------
    img : np.array (n x m x 3)
        Image to be transformed
    subImgSize : np.array (a x b)
        Input size for model
    model: keras.model
        Keras model used to make predictions
    
    Returns
    -------
    pred: np.array (n x m)
        Prediction from image
    """

    # get the size of the input image
    l,w,_ = np.shape(img)
    # init array for new image
    pred = np.zeros(shape = (l,w))

    r = l//subImgSize[0]
    c = w//subImgSize[1]

    roffset = 0
    coffset = 0
    
    if l%subImgSize[0] != 0:
        roffset = 1
    if w%subImgSize[1] != 0:
        coffset = 1
 
    x1 = 0
    predX1 = 0
    # Crop the image
    for j in range(r + roffset):
        y1 = 0
        predY1 = 0

        x2 = (j+1)*subImgSize[0] 

        if x2 > l:
            x2 = l
            x1 = l - subImgSize[0]
            
        for k in range(c + coffset):
            # find upper bounds of window
            y2 = (k+1)*subImgSize[1] 
            
            # check if outer dimension is larger than image size and adjust

            if y2 > w:
                y2 = w
                y1 = w - subImgSize[1]

            # crop area of picture
            croppedArea = img[x1:x2,y1:y2,:]
            # make prediction using model
            
            modelPrediction = model.predict(np.expand_dims(croppedArea,axis = 0))
            # update prediction image
            pred[predX1:x2,predY1:y2] = modelPrediction[0,(predX1-x1):,(predY1-y1):,0]
            # update the bounds
            y1 = y2
            predY1 = y1 

        # update the lower x bound
        x1 = x2 
        predX1 = x1

    return pred

def setGenerator(train_path,shape,model,correctClass):
    images = os.listdir(train_path + "/data/")    
    ind = np.random.randint(0,len(images),len(images))  
    
    for i in ind:
        image = io.imread(train_path + "/data/" + images[i])
        label = io.imread(train_path + "/label/" + images[i])
        prediction = getTrainPredictions(image,shape,model)
        
        # modify prediction 
        if correctClass != None:
            prediction[prediction == correctClass] = 100
            prediction[prediction != 100] = 0


        yield(image,prediction,label)
               
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
        gen = setGenerator(path + "/" + classes[i], params['shape'], params['model'], classMap[classes[i]])

        for cnt, batch in enumerate(gen):
            if(cnt >= params['num_of_img']):
                break
            else:
                line = cnt + i*params['num_of_img']
                # make plots
                axes[line,0].imshow(batch[0])
                axes[line,0].axis("off")
                axes[line,1].imshow(batch[1])
                axes[line,1].axis("off")
                axes[line,2].imshow(batch[2])
                axes[line,2].axis("off")
            
    fig.tight_layout()
    plt.show()

def plotEpochAccuracyAndLoss(csv):
    # read in the csv with the results
    df = pd.read_csv(csv)
    # make first plot
    plt.plot(df['epoch'],df['acc'], label = "Train")
    plt.plot(df['epoch'],df['val_acc'],label = "Validation")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    # make secind plot
    plt.subplot(122)
    plt.plot(df['epoch'],df['loss'], label = "Train")
    plt.plot(df['epoch'],df['val_loss'],label = "Validation")
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def plotBatchAccuracyAndLoss(csv):
    df = pd.read_csv(csv)
    plt.subplot(121)
    plt.plot(df['Batch'],df["Accuracy"])
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

    plt.show()