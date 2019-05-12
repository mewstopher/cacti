import tensorflow as tf
import os 
import tqdm
#import tqdm_notebook
import pandas as pd
import numpy as np
import cv2
import sys
sys.path.append("../../")
from cnn_utils.helper_functions import *
from cnn_utils.image_tools import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
fold = os.listdir("../input/train/")
train_df = pd.read_csv("../input/train.csv")


import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
#import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


