import os

def dirSize(path,dirs):
    """Finds the size of the directories specified.
    
    Parameters
    ----------
    path: String
        Location of the directories in question
    dirs: list OR String
        Which directories to include in the length
    
    Returns
    -------
    length : int
        Number of images contained in the directories in question.
    """
    oldDir = os.getcwd()
    os.chdir(path)
    cwd = os.getcwd()
    length = 0
    if isinstance(dirs,str):
        dirs = [dirs]
    for d in dirs:
        path = cwd + "/" + d + "/data/"
        subdirs = os.listdir(path)
        for s in subdirs:
            length += len(os.listdir(path + s))
    
    os.chdir(oldDir)
    
    return length

def combineFiles(first,second):
    """Combine the results from multiple files.

    Useful for combining results from multiple inputs and saves as the 
    first files name with a 2 appended to it.

    Parameters
    ----------
    first : String
        Path to the first file to be combined 
    second : String
        Path to the second file to be combined
    """
    fname = first.replace(".csv","2.csv")
    fnew = open(fname,"w")

    batch = 0

    with open(first,"r") as f:
        for line in f:
            batch = line.split(",")[-1]
            fnew.write(line)

    with open(second,"r") as f:
        i = 0
        for line in f:
            if i != 0:
                line = line.split(",")
                line[-1] += batch
                fnew.write(",".join(line))
            i +=1
    fnew.close()

def countFiles(path):
    """Counts the number of files in the training, validation, and test directories.
    
    Parameters
    ----------
    path : String
        path to the train,validata, and test images
    """
    os.chdir(path)

    dirs = ["train","validate","test"]

    for d in dirs:
        dirCnt = 0
        plant = os.listdir(path + "/" + d)
        for i in plant: 
            tmpDir = path + "/" + d + "/" + i + "/data/"
            subDirs = os.listdir(tmpDir)
            plntCnt = 0
            for j in subDirs:
                plntCnt += len(os.listdir(tmpDir + j))
            print(d + " " + i + ": " + str(plntCnt))
            dirCnt += plntCnt
        print(d + ": " + str(dirCnt))
   