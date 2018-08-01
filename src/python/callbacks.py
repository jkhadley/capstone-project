# Author(s): Joseph Hadley
# Date Created:  2018-07-14
# Date Modified: 2018-07-24
# Description: Keras callback to allow for semi-regular updates on the models progress.

import keras
import numpy as np

class BatchLogger(keras.callbacks.Callback):

    def __init__(self,batch_f,epoch_f,batchInterval):
        self.batch_f = batch_f
        self.epoch_f = epoch_f
        self.batchInterval = batchInterval
        self.counter = 0
        self.loss = []
        self.accuracy = []
        self.recall = []
        self.precision = []
        self.f1score = []
        self.batch = 0
        
        # initialize files
        f = open(batch_f,"w")
        f.write("Loss,Accuracy,Recall,Precision,F1-Score,Batch\n")
        f.close()
        
        f = open(epoch_f,"w")
        f.write("Train_Loss,Train_Accuracy,Val_Loss,Val_Accuracy,epoch\n")
        f.close()

    def on_batch_end(self,batch,logs = {}):
        if self.counter == self.batchInterval:
            self.counter = 0
            
            self.loss.append(logs.get('loss'))
            self.accuracy.append(logs.get('acc'))
            self.recall.append(logs.get('recall'))
            self.precision.append(logs.get('precision'))
            self.f1score.append(logs.get('f1Score'))

            self.batch += self.batchInterval
            aveLoss = np.average(self.loss)
            aveAcc = np.average(self.accuracy)
            aveRecall = np.average(self.recall)
            avePrecision = np.average(self.precision)
            aveF1 = np.average(self.f1score)
            f = open(self.batch_f,"a")
            f.write(str(aveLoss) + "," + str(aveAcc) + "," + str(aveRecall) + "," + str(avePrecision)+","+ str(aveF1) + ","+  str(self.batch) + '\n')
            f.close()

            # clear the lists
            self.loss = []
            self.accuracy = []
            self.f1score = []
            self.recall = []
            self.precision = []
        else:
            self.loss.append(logs.get('loss'))
            self.accuracy.append(logs.get('acc'))
            self.recall.append(logs.get('recall'))
            self.precision.append(logs.get('precision'))
            self.f1score.append(logs.get('f1Score'))
            self.counter +=1
    
    def on_epoch_end(self,epoch,logs = {}):
            f = open(self.epoch_f,"a")
            f.write(str(logs.get("loss")) + "," + str(logs.get("acc")) + "," + str(logs.get("val_loss")) + "," + str(logs.get("val_acc")) + "," + str(epoch) + '\n')
            f.close()