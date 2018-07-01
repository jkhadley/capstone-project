import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import os 


def splitTrainImages(path,savepath,shape):

    cwd = os.getcwd()
    os.chdir(path)
    d = os.getcwd()

    # initialize counting variables
    i = 0
    irrelavant = 0
    mismatch = 0

    # get list of images
    f = os.listdir("./data")
    # read in image and label together
    img = io.imread(d + '/data/' + f[i])


    for i in range(len(f)):
        # load image
        img = io.imread(d + '/data/' + f[i])
        label = io.imread(d + '/label/' + f[i])

        name = f[i].replace(".jpg","").replace(".jpeg","")

        # get dimensions of the image
        r1,c1,_ = img.shape
        r2,_,_ = img.shape
        print("Image")
        if r1 == r2:

            # get full divisions of shape
            r = r1//shape[0]
            c = c1//shape[1]
            
            x1 = 0
            m = 0
            
            for j in range(r +1):
                y1 = 0
                for k in range(c + 1):
                    
                    #print(m)
                    # find upper bounds of window
                    x2 = (j+1)*shape[0] 
                    y2 = (k+1)*shape[1] 
                    
                    # check if outer dimension is larger than image size and adjust
                    if x2 > r1:
                        x2 = r1
                        x1 = r1 - shape[0]

                    if y2 > c1:
                        y2 = c1
                        y1 = c1 - shape[1]
                    print("X1: " + str(x1) + " ,X2: " + str(x2))
                    print("Y1: " + str(y1) + " ,Y2: " + str(y2))
                    print("X: " + str(x2-x1))
                    print("Y: " + str(y2-y1))
                    '''
                    # crop the image
                    imgCrop = img[x1:x2,y1:y2]
                    labCrop = label[x1:x2,y1:y2]

                    # only save image if less than 98% of image is ground              
                    if np.sum(np.sum(labCrop)) > 0.98*shape[0]*shape[1]*100:
                        irrelavant += 1
                    else:
                        # save images to appropriate directories
                        io.imsave(savepath + '/data/' + name + "_" + str(m) + ".jpg",arr = imgCrop)
                        io.imsave(savepath + '/label/' + name + "_" + str(m)+ ".jpg",arr = labCrop)

                    #save 
                    '''
                    # update the lower bounds of window
                    
                    y1 = y2 
                    m += 1
                x1 = x2 

            break
        else:
            mismatch +=1
    print('mismatched dimensions:' + str(mismatch))
    print("junk sections: " + str(irrelavant)) 

    os.chdir(cwd)  

# test the functions
if __name__ == "__main__":
    splitTrainImages('../../data/groundcover2016/maize/size1/train','../../train',(256,256))
    splitTrainImages('../../data/groundcover2016/maize/size1/validate','../../validate',(256,256))
    