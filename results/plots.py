import matplotlib.pyplot as plt
import pandas as pd
from skimage import io

a = io.imread("C:\\Users\\jkhad\\Documents\\2018summer\\eas560\\data\\groundcover2016\\label\\maize.jpg")
plt.imshow(a)
plt.show()

df = pd.read_csv("./mulitclassResults.csv")

plt.subplot(141)
plt.plot(df['depth'],df['accuracy'],label = "Accuracy")
plt.plot(df['depth'],df['f1-score'],label = "F1-score")
plt.plot(df['depth'],df['precision'],label = "Precision")
plt.plot(df['depth'],df['recall'],label = "Recall")
plt.title("Multi-Class Segmentation")
plt.xlabel("Depth")
plt.legend()
plt.grid()

plt.subplot(142)
plt.plot(df['depth'],df['rmse'],label = "RMSE")
plt.plot(df['depth'],df['mse'],label = "MSE")
plt.plot(df['depth'],df['mae'],label = "MAE")
plt.title("Multi-Class Regression")
plt.xlabel("Depth")
plt.legend()
plt.grid()
plt.tight_layout()


#---------------------------------------------------
df = pd.read_csv("./maizeDepthResults.csv")

plt.subplot(143)
plt.plot(df['depth'],df['accuracy'],label = "Accuracy")
plt.plot(df['depth'],df['f1-score'],label = "F1-score")
plt.plot(df['depth'],df['precision'],label = "Precision")
plt.plot(df['depth'],df['recall'],label = "Recall")
plt.title("Single-Class Segmentation")
plt.xlabel("Depth")
plt.legend()
plt.grid()

plt.subplot(144)
plt.plot(df['depth'],df['rmse'],label = "RMSE")
plt.plot(df['depth'],df['mse'],label = "MSE")
plt.plot(df['depth'],df['mae'],label = "MAE")
plt.title("Single-Class Regression")
plt.xlabel("Depth")
plt.ylabel("Metrics")
plt.legend()
plt.grid()
plt.tight_layout()
#plt.savefig('depthResults.png')
plt.show()
#---------------------------------------------------

df = pd.read_csv('./maize8_batch.csv')
batch_size = 128

plt.subplot(1,4,1)
plt.plot(df['batch']*batch_size/1000,df['loss'])
plt.xlabel('Thousand of Images')
plt.ylabel('Loss')
plt.title('Single-Class Loss')
plt.grid()

plt.subplot(142)
plt.plot(df['batch']*batch_size/1000,df['acc'],label = "Accuracy")
plt.plot(df['batch']*batch_size/1000,df['recall'],label = "Recall")
plt.plot(df['batch']*batch_size/1000,df['precision'],label = "Precision")
plt.plot(df['batch']*batch_size/1000,df['f1Score'],label = "F1-Score")
plt.xlabel('Thousand of Images')
plt.ylabel('Metrics')
plt.title('Single-Class Metrics')
plt.legend()
plt.grid()

df = pd.read_csv('./multiclass64_batch.csv')
batch_size = 16

plt.subplot(143)
plt.plot(df['batch']*batch_size/1000,df['loss'])
plt.xlabel('Thousand of Images')
plt.ylabel('Loss')
plt.title('Multi-Class Loss')
plt.grid()

plt.subplot(144)
plt.plot(df['batch']*batch_size/1000,df['acc'],label = "Accuracy")
plt.plot(df['batch']*batch_size/1000,df['recall'],label = "Recall")
plt.plot(df['batch']*batch_size/1000,df['precision'],label = "Precision")
plt.plot(df['batch']*batch_size/1000,df['f1Score'],label = "F1-Score")
plt.xlabel('Thousand of Images')
plt.ylabel('Metrics')
plt.title('Multi-Class Metrics')
plt.legend()
plt.grid()

plt.show()

df = pd.read_csv('./maize32FullRegression_batch.csv')
plt.subplot(121)
plt.plot(df['batch']*batch_size/1000,df['loss'])

plt.xlabel('Thousand of Images')
plt.ylabel('Loss')
plt.title('Full Regression Loss')
plt.legend()
plt.grid()

plt.subplot(122)
plt.plot(df['batch']*batch_size/1000,df['RMSE'],label = "RMSE")
plt.plot(df['batch']*batch_size/1000,df['mean_absolute_error'],label = "MAE")
plt.ylabel('Metrics')
plt.title('Full Regression Metrics')
plt.legend()
plt.grid()

plt.show()