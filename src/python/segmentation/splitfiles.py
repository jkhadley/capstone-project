import os
import numpy as np

def splitDataProperly():
    # change dirctory to data that need to be balanced
    os.chdir('./../../data/groundcover2016/maize/size1/')
    wd = os.getcwd()
    # get counts of data in train, validate and test
    testCnt = len(os.listdir(wd + '/test/data'))
    trainCnt = len(os.listdir(wd + '/train/data'))
    valCnt = len(os.listdir(wd + '/validate/data'))
    print("Test count: " + str(testCnt))
    print("Train count: " + str(trainCnt))
    print("validation count: " + str(valCnt))
    
    # get total number of images
    total = testCnt + trainCnt + valCnt

    testAmnt = round(total*0.125)
    validateAmnt = round(total*0.125)
    trainAmnt = total - (testAmnt + validateAmnt)
    
    print("Test amnt: " + str(testAmnt))
    print("Train amnt: " + str(trainAmnt))
    print("validation amnt: " + str(validateAmnt))
    
    
    if trainCnt < trainAmnt:
        testImages = os.listdir(wd + '/test/data')

        amntToMove = validateAmnt - valCnt

        if amntToMove < (testCnt - testAmnt):
            validateInd = np.random.randint(0,len(testImages),size = amntToMove)
            print("AAAAAAAa")
            for i in range(0,len(testImages)):
                if i in validateInd:
                    os.rename(wd + "/test/data/" + testImages[i],wd + "/validate/data/" + testImages[i])
                    os.rename(wd + "/test/label/" + testImages[i],wd + "/validate/label/" + testImages[i])
                else:
                    # do nothing
                    pass
    # update test count
    testCnt = len(os.listdir(wd + '/test/data'))
    if (testCnt - testAmnt) > 0:
        testImages = os.listdir(wd + '/test/data')
        # move excess to train data
        trainInd = np.random.randint(0,len(testImages),size = (testCnt - testAmnt))
        for i in range(0,len(testImages)):
            if i in trainInd:
                os.rename(wd + "/test/data/" + testImages[i],wd + "/train/data/" + testImages[i])
                os.rename(wd + "/test/label/" + testImages[i],wd + "/train/label/" + testImages[i])
            else:
                # do nothing
                pass
            
# run the function if main

if __name__ == "__main__":
    splitDataProperly()