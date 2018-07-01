from keras.models import load_model
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import sys


def getTrainPredictions(img,subImgSize,model):
    # get the size of the input image
    l,w,d = np.shape(img)
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


if __name__ == "__main__":
    subImgSize = (256,256)
    model = load_model('../../model_checkpoints/unet_train_all_reshaped.hdf5')
    img = io.imread('../../data/groundcover2016/maize/test/data/20160106_064722.jpg')
    img2 = getTrainPredictions(img,subImgSize,model)
    # make a plot
    plt.subplot(121)
    plt.imshow(img.astype("uint8"))
    plt.subplot(122)
    plt.imshow(img2.astype("uint8"))
    plt.show()