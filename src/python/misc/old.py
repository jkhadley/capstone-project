'''
def generateRescaledData(path,dataFolder,labelFolder,rescaleSize,stride):
    cwd = os.getcwd()
    os.chdir(path)
    
    totalDir = os.getcwd()
    
    # make file paths 1 variable
    imgDir = totalDir + dataFolder
    labelDir = totalDir + labelFolder

    # get list of files from each directory
    imgFiles = np.sort(os.listdir(imgDir))
    labelFiles = np.sort(os.listdir(labelDir))

    if np.all(imgFiles == labelFiles):
        for i in range(len(imgFiles)):
            # load in images
            img = io.imread(imgDir + '/' + imgFiles[i],1)
            label = io.imread(labelDir + '/' + imgFiles[i],0)
            
            w1,_, _ = img.shape
            w2,_ = label.shape            

            if w1 != w2:
                img = rotate(img,90)
                       
            # use tensorflow to get patches of images
            with tf.Session() as _:
                imgPatches = tf.extract_image_patches(images = np.expand_dims(img,axis = 0),
                                                ksizes = [1,rescaleSize[0],rescaleSize[1],1],
                                                strides = [1,stride[0],stride[1],1],
                                                rates = [1,1,1,1],
                                                padding = "VALID").eval()

                labelPatches = tf.extract_image_patches(images = np.expand_dims(np.expand_dims(label,axis = 0),axis = -1),
                                                ksizes = [1,rescaleSize[0],rescaleSize[1],1],
                                                strides = [1,stride[0],stride[1],1],
                                                rates = [1,1,1,1],
                                                padding = "VALID").eval()
                #imgArray = tf.reshape(imgPatches, [45, 256, 256, 3])
                #labArray = tf.reshape(labelPatches, [45, 256, 256,1])
            
            imgDim = np.shape(imgPatches)
            labDim = np.shape(labelPatches)
            
            # initialize patch arrays
            imgArray = np.zeros(((imgDim[1]-1)*(imgDim[2]-1),rescaleSize[0],rescaleSize[1],3))
            labArray = np.zeros(((labDim[1]-1)*(labDim[2]-1),rescaleSize[0],rescaleSize[1],1))

            # reshape patches from tensorflow into images
            for i in range(imgDim[1]-1):
                for j in range(imgDim[2]-1):
                    ind = (imgDim[1]-1)*i + j
                    
                    tmpImg = imgPatches[0,i,j,:].reshape(rescaleSize[0],rescaleSize[1],3)
                    tmpLab = labelPatches[0,i,j,:].reshape(rescaleSize[0],rescaleSize[1],1)
                    # normalize data
                    if np.max(tmpImg) > 1:
                        tmpImg = tmpImg/np.max(tmpImg)
                    if np.max(tmpLab) > 1:
                        tmpLab = tmpLab/np.max(tmpLab)

                    imgArray[ind,:,:,:] = tmpImg
                    labArray[ind,:,:,:] = tmpLab
            
            yield (imgArray,labArray)
    
    os.chdir(cwd)
'''