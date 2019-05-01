import tensorflow as tf
import os 
import tqdm
#import tqdm_notebook
import pandas as pd
import cv2
fold = os.listdir("input/train/")
train_df = pd.read_csv("input/train.csv")

fold = "input/train/"
train_df.head()

X_tr = []
Y_tr = []
imges = train_df['id'].values
for img_id in imges:
    X_tr.append(cv2.imread(fold + img_id))    
    Y_tr.append(train_df[train_df['id'] == img_id]['has_cactus'].values[0])  

X_tr = np.asarray(X_tr)
X_tr = X_tr.astype('float32')
X_tr /= 255
Y_tr = np.asarray(Y_tr)
