from modelInferencer import ModelInferencer
from modelTrainer import ModelTrainer
from keras.models import load_model
from keras.losses import mean_squared_error
from generators import singleClassGenerator
#import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import os

# variables
datapath = "/home/ubuntu/project/data/groundcover2016/"
resultspath = "/home/ubuntu/project/results/"
modelpath = "/home/ubuntu/project/model_checkpoints/unet/"
model = "smallNetwork2.hdf5"
gen = singleClassGenerator(datapath + "train/",{"maize" : 1},True)

for i,batch in enumerate(gen):
    image = batch[0]
    label = batch[1]
    if i >1:
        break

# setup the Model Inferencer
inferencer = ModelInferencer(modelpath + model,datapath)
inferencer.setClassMap({'maize' : 1})
inferencer.setBatchSize(1)
imagePrediction = inferencer.segmentationPredict(image)

reg = inferencer.regressionPredict(image)

# get model output
trainer = ModelTrainer(datapath,resultspath,modelpath)
trainer.setClassMap({'maize' : 1})
trainer.setOldModel(model)
trainer.setRegression()
trainer.changeBatchSize(1)


# print the images
#plt.subplot(131)
#plt.imshow(image)
#plt.subplot(132)
#plt.imshow(label)
#plt.subplot(133)
#plt.imshow(imagePrediction)
#plt.show()
trained = trainer.singlePrediction(image)
print("Inferencer Prediction : " + str(reg))
print("Train Prediction      : " + str(trained))
print("label                 : " + str(label))
#print("MSE: " + str(mean_squared_error(trained,reg)))


