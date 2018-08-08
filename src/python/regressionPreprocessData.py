from openpyxl import load_workbook
from multiprocessing import Pool
from skimage import io
import pandas as pd
import numpy as np
import os

def consolidateFiles(src,write_dest):
    
    wf = open(write_dest,"w")
    wf.write("imageName,propPlant\n")
    os.chdir(src)
    files = os.listdir()

    for f in files:
        l = 0
        
        # load the workbook and select the worksheet
        wb = load_workbook(f,read_only=True)
        ws = wb['CAN-EYE NADIR Results']

        for row in ws.rows:
            if l > 1:
                c = 0
                # go through each cell
                for cell in row:
                    if c == 0:
                        wf.write(str(cell.value))
                    else:
                        wf.write("," + str(cell.value))
                    c += 1
                wf.write("\n") # carraige return after each row
            l+=1

def consolidateAllFiles():
    src = ['/ratio/maize_ratio','/ratio/bean_ratio','/ratio/wheat_ratio','/ratio/maize_variety_ratio']
    dest = ['/maize.csv','/mungbean.csv','/wheat.csv','/maizevariety.csv']

    # consolidate files
    for i in range(len(src)):
        consolidateFiles(cwd + src[i],cwd + dest[i])
        print(src + " done!\n")

def savePropOfGround(params):
    os.chdir(params['src'])
    pwd = os.getcwd()

    f = open(params['writename'],"w")
    f.write("imageName,propPlant\n")
    
    img = os.listdir()

    for i in img:
        image = io.imread(pwd + "/"  + i)
        l,w = image.shape
        image[image >= 1] = 1
        
        tot = np.sum(image) 
        propGround = 1-tot/(l*w)

        f.write(i + "," + str(propGround) + "\n")

def calcPropOfGround():
    pwd = os.getcwd()
    p = pwd + "/groundcover2016/"
    p2 = pwd + "/ratio/"
    src = ['maize/label','mungbean/label','wheat/label','maizevariety/label']
    dest = ['maizeCalc.csv','mungbeanCalc.csv','wheatCalc.csv','maizevarietyCalc.csv']

    # get list of dictionaries to pass to multiprocessing
    paramList = []
    for i in range(len(src)):
        paramList.append({
            'src' : p + src[i],
            'writename' :p2 +  dest[i]
            })

    pool = Pool()
    pool.map(savePropOfGround,paramList)

def getErrors(params):
    df1 = pd.read_csv(params['path'] +  "/" + params['f1'] + ".csv")
    df2 = pd.read_csv(params['path'] +  "/" + params['f2'] + ".csv")
    df = df1.set_index('imageName').join(df2.set_index('imageName'),lsuffix = "_truth",rsuffix = "_calc")
    df['diff'] = df['propPlant_truth'] - df['propPlant_calc']
    df['mse'] = df['diff']**2

    mspe = df['mse'].sum()/df['mse'].count()
    print(params['f1'] + " MSPE: "  + str(mspe))

def getAllErrors():
    d = os.getcwd()
    f = ['maize','mungbean', 'wheat','maizevariety']

    paramList = []
    for i in range(len(f)):
        paramList.append({
            'path' : d,
            'f1' : f[i],
            'f2' : f[i] + "Calc"})
    
    pool = Pool()
    pool.map(getErrors,paramList)

if __name__ == "__main__":
    path = "../../../data/ratio"
    os.chdir(path)
    cwd = os.getcwd()
    #consolidateAllFiles()
    os.chdir("..")
    cwd = os.getcwd()
    calcPropOfGround()
    os.chdir("ratio")
    getAllErrors()