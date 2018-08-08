import keras
import numpy as np

class BatchLogger(keras.callbacks.Callback):
    """Custom keras callback to record the results more frequently"""
    def __init__(self,batch_f,epoch_f,batchInterval,metrics):
        """Initialize the batch Logger.

        Parameters
        ----------
        batch_f : String
            file to put batch results in
        epoch_f : String
            file to put epoch results in
        batchInterval : int
            number of batches between the recording batch results
        metrics : list
            the metrics used to measure model performance
        """

        self.batch_f = batch_f
        self.epoch_f = epoch_f
        self.batchInterval = batchInterval
        self.metrics = metrics
        self.counter = 0
        
        params = []
        params.append('loss')
        params.extend(self.metrics)
        self.batchParams = params
        self.epochParams = params.copy()
        valParams = [('val_' + i) for i in params]
        self.epochParams.extend(valParams)

        self.batchResults = [0] * len(self.batchParams)
        self.epochResults = [0] * len(self.epochParams)

        self.batch = 0

    def on_batch_end(self,batch,logs = {}):
        """Logs the results at the end of the batch.
        
        Parameters
        ----------
        batch : int
            Which batch just finished
        logs : dictionary
            Logged metrics used to evaluate the model
        """
        # add results to the list
        for i in range(len(self.batchParams)):
            self.batchResults[i] += logs.get(self.batchParams[i])


        if self.counter == self.batchInterval -1:
            self.counter = 0

            # divide by batch interval size
            self.batchResults[:] = [i/self.batchInterval for i in self.batchResults]
            
            # convert results to a list of strings
            results = [str(i) for i in self.batchResults]

            # check if batch 0 and open file appropriately
            if self.batch == self.batchInterval-1:
                f = open(self.batch_f,"w")
                f.write(",".join(self.batchParams) +",batch\n")
            else: 
                f = open(self.batch_f,"a")

            # write the batch results to the file                     
            f.write(",".join(results) +','+str(self.batch) + '\n')
            f.close()
            
            # re-initialize list
            self.batchResults = [0] * len(self.batchParams)
        else:
            self.counter += 1
        
        self.batch += 1


    
    def on_epoch_end(self,epoch,logs = {}):
        """Records the results at the end of the epoch.
        
        Parameters
        ----------
        epoch : int
            Which epoch just finished
        logs : dictionary
            Logged metrics used to evaluate the model
        """
        # get the results
        for i in range(len(self.epochParams)):
            self.epochResults[i] = logs.get(self.epochParams[i])        
        
        # convert results to a list of strings
        results = [str(i) for i in self.epochResults]
        
        # open the file
        if epoch == 1:
            f = open(self.epoch_f,"w")
            f.write(",".join(self.epochParams) + ",epoch\n")
        else:
            f = open(self.epoch_f,"a")

        f.write(",".join(results) + "," +str(epoch) + '\n')
        f.close()

