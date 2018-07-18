import os 
p = "/home/ubuntu/project/data/groundcover2016/"
os.chdir(p)

dirs = ["train","validate","test"]

for d in dirs:
    dirCnt = 0
    plant = os.listdir(p + "/" + d)
    for i in plant: 
        tmpDir = p + "/" + d + "/" + i + "/data/"
        subDirs = os.listdir(tmpDir)
        plntCnt = 0
        for j in subDirs:
            plntCnt += len(os.listdir(tmpDir + j))
        print(d + " " + i + ": " + str(plntCnt))
        dirCnt += plntCnt
    print(d + ": " + str(dirCnt))