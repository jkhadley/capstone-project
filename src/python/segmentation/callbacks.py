import keras
import numpy as np

class WriteBatchResultsToCSV(keras.callbacks.Callback):

    def __init__(self,batch_f,epoch_f,batchInterval):
        self.batch_f = batch_f
        self.epoch_f = epoch_f
        self.batchInterval = batchInterval
        self.counter = 0
        self.loss = []
        self.accuracy = []
        self.batch = 0
        
        # initialize files
        f = open(batch_f,"w")
        f.write("Loss,Accuracy,Batch\n")
        f.close()
        
        f = open(epoch_f,"w")
        f.write("Train_Loss, Train_Accuracy, Val_Loss, Val_Accuracy,epoch\n")
        f.close()

    def on_batch_end(self,batch,logs = {}):
        if self.counter == self.batchInterval:
            self.counter = 0
            
            self.loss.append(logs.get('loss'))
            self.accuracy.append(logs.get('acc'))

            self.batch += self.batchInterval
            aveLoss = np.average(self.loss)
            aveAcc = np.average(self.accuracy)
            
            f = open(self.batch_f,"a")
            f.write(str(aveLoss) + "," + str(aveAcc) + "," + str(self.batch) + '\n')
            f.close()

            # clear the lists
            self.loss = []
            self.accuracy = []
        else:
            self.loss.append(logs.get('loss'))
            self.accuracy.append(logs.get('acc'))
            self.counter +=1
    
    def on_epoch_end(self,epoch,logs = {}):
            f = open(self.epoch_f,"a")
            f.write(str(logs.get("loss")) + "," + str(logs.get("acc")) + "," + str(logs.get("val_loss")) + "," + str(logs.get("val_acc")) + ","  + '\n')
            f.close()