
# # Importing Python libraries 

# In[1]:


import os
import cv2
import torch 
import skimage
import numpy as np
import os, os.path
import pandas as pd
from tqdm import tqdm
from os import listdir
from keras import backend as K
import matplotlib.pyplot as plt
from keras.utils import np_utils
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.datasets import load_files
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.layers import Conv2D,MaxPooling2D
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.preprocessing.image import array_to_img, img_to_array, load_img


# # Data Preprocessing same as we did in training time

# In[2]:


import skimage.transform
test_dir = 'testdata/' 
def get_data(folder):
    X = []
    y = []
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in   ['cherry']:
                label = 0
            elif folderName in ['strawberry']:
                label = 1
            elif folderName in ['tomato']:
                label = 2
            else:
                label = 4
            for image_filename in tqdm(os.listdir(folder + folderName)):
                img_file = cv2.imread(folder + folderName + '/' + image_filename)
                
                if img_file.size  > 27000:
                    if img_file is not None:
                        img_file = skimage.transform.resize(img_file, (300, 300, 3))
                        img_arr = np.asarray(img_file)
                        X.append(img_arr)
                        y.append(label)
                else:
                    print('Not pass')
    X = np.asarray(X)  
    y = np.asarray(y)
    return X,y


# In[3]:


X_test, y_test= get_data(test_dir)
y_test = to_categorical(y_test)


# # Loading the Model

# In[4]:


model = load_model('model.h5')


# # Evaluate and  test accuracy

# In[5]:


score = model.evaluate(X_test, y_test, verbose=0)
print('\n', 'Test accuracy:', round(score[1],2))

