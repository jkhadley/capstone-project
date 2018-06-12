# -*- coding: utf-8 -*-
"""
Author(s): Joseph Hadley
Date Created:  2018-06-08
Date Modified: 2018-06-09
Description: Explore image segmentation techniques
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
#---------------------------------------------------------------------------------------------------------
#                                           Functions
#---------------------------------------------------------------------------------------------------------
def checkSizeOfRandomImages(path,n):
    # inputs: This function takes a path and the number of images to check
    # outputs: outputs a True if the images are all the same size and a False if not
    cwd = os.getcwd()
    os.chdir(path)
    images = os.listdir()
    
    # sample n random images
    if n < len(images):
        randImages = np.random.randint(0,len(images),n)
        tmp = [images[i] for i in randImages]
        images = []
        images = tmp
        tmp = []
        
    # get info for first image
    im = cv2.imread(images[0],1)
    size = np.shape(im)
    consistent = True

    for i in range(1,n):
        # read in new image
        im = cv2.imread(images[i],1)
        newSize = np.shape(im)
        if newSize != size:
            # Check if the orientation is different
            if newSize[0] != size[1] or newSize[1] != size[0]:
                consistent = False
                break

    # change back to starting directory
    os.chdir(cwd)
    return consistent

def plotImageLayers(img):
    # Input(s) : image in the standard format
    # Output(s): None, creates a window with the images

    fig = plt.figure()
    fig.add_subplot(221)
    plt.imshow(img[:,:,0])
    fig.add_subplot(222)
    plt.imshow(img[:,:,1])
    fig.add_subplot(223)
    plt.imshow(img[:,:,2])
    plt.show()    

def plotHistogramsForImageLayers(img):
    # Input(s) : image in the standard format
    # Output(s): None, creates a window with the images

    fig = plt.figure()
    fig.add_subplot(221)
    plt.hist(img[:,:,0].flatten())
    fig.add_subplot(222)
    plt.hist(img[:,:,1].flatten())
    fig.add_subplot(223)
    plt.hist(img[:,:,2].flatten())
    plt.show()

def thresholdLayer(r,c,matrix,threshold):
    # Input(s) :    r, number of rows in the matrix
    #               c, number of columns in the matrix
    #               matrix, layer to be thresholded
    #               threshold, Threshold to be applied to the image
    #
    # Output(s):    matrix, Layer of image filtered through the threshold

    for i in range(0,r):
        for j in range(0,c):
            if matrix[i,j] < threshold:
                matrix[i,j] = 0
            else:
                pass
    return matrix

def thresholdImage(image,t0,t1,t2):
    # Input(s) :    image, image opened from imread to be filtered using a threshold
    #               t0, Threshold to be applied to the first layer of the image (R)
    #               t1, Threshold to be applied to the second layer of the image (G)
    #               t2, Threshold to be applied to the third layer of the image (B)
    #
    # Output(s):    None
    
    # get dimensions of the image
    r,c,d = np.shape(image)
    original = deepcopy(image)
    # apply thresholding to each layer
    image[:,:,0] = thresholdLayer(r,c,image[:,:,0],t0)
    image[:,:,1] = thresholdLayer(r,c,image[:,:,1],t1)
    image[:,:,2] = thresholdLayer(r,c,image[:,:,2],t2)

    # plot the original image and the thresholded image
    fig = plt.figure()
    fig.add_subplot(121)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')
    fig.add_subplot(122)
    plt.imshow(image)
    plt.title("Thresholded Image")
    plt.axis('off')
    plt.show()

#---------------------------------------------------------------------------------------------------
#                                               main
#---------------------------------------------------------------------------------------------------
def main():
    # Change directory to one with images
    os.chdir("../../data/groundcover2016/maize/data")
    # check if images are the same size
    print(checkSizeOfRandomImages(".",10))
    # images arent the same size so the process is going to have to be dynamic
    
    # get a test image
    images = os.listdir()
    f = images[0]
    # import a picture
    img = cv2.imread(f,1)

    '''
    # Look at the different layers of the image
    plotImageLayers(img)
    plotHistogramsForImageLayers(img)
    # doesnt look like there are any clear splits 
    '''
    
    # try and threshold the image
    # thresholdImage(img,0,150,0)

    # Try Edge Detection
    edge = cv2.Canny(img,100,150)
    plt.imshow(edge)
    plt.axis("off")
    plt.show()

    # edge detector looks promising

# Run the main function if this script is being run directly
if __name__ == "__main__":
    main()