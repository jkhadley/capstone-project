from shutil import copy
import random
import os 

def splitData(path,c,split):

    src = path + "/" + c
    trainDest = path + "/train/" + c
    validateDest = path + "/validate/" + c
    testDest = path + "/test/" + c 

    cwd = os.getcwd()
    os.chdir(src)

    images = os.listdir()
    random.shuffle(images)
    split1 = round(split[0]*len(images))
    split2 = round((split[0] + split[1])*len(images))

    train = images[:split1]
    validate = images[split1:split2]
    test = images[split2:]

    for i in images:
        if i in train:
            copy(src + "/" + i,trainDest + "/" + i)
        elif i in validate:
            copy(src + "/" + i,validateDest + "/" + i)
        else:
            copy(src + "/" + i,testDest + "/" + i)
    
    os.chdir(cwd)

if __name__ == "__main__":
    path = "/home/ubuntu/project/data/maize/"
    #path = "../../data/maize"
    os.chdir(path)

    path = os.getcwd()
    classes = os.listdir()

    os.mkdir("train")
    os.mkdir("validate")
    os.mkdir("test")

    for c in classes:
        os.mkdir("./train/" + c)
        os.mkdir("./validate/" + c)
        os.mkdir("./test/" + c)
        splitData(path, c,[0.8,0.1,0.1])

