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


# # Some Images Cherry Classe

# In[3]:


img = mpimg.imread('./traindata/cherry/cherry_0033.jpg')
print(img.shape)
plt.imshow(img)


# In[4]:


img = mpimg.imread('./traindata/cherry/cherry_0044.jpg')
print(img.shape)
plt.imshow(img)


# In[5]:


img = mpimg.imread('./traindata/cherry/cherry_0090.jpg')
print(img.shape)
plt.imshow(img)


# In[6]:


img = mpimg.imread('./traindata/cherry/cherry_0400.jpg')
print(img.shape)
plt.imshow(img)


# In[7]:


img = mpimg.imread('./traindata/cherry/cherry_0454.jpg')
print(img.shape)
plt.imshow(img)


# In[8]:


img = mpimg.imread('./traindata/cherry/cherry_0777.jpg')
print(img.shape)
plt.imshow(img)


# # Some Images Tomato Class

# In[9]:


img = mpimg.imread('./traindata/tomato/tomato_0601.jpg')
print(img.shape)
plt.imshow(img)


# In[10]:


img = mpimg.imread('./traindata/tomato/tomato_0099.jpg')
print(img.shape)
plt.imshow(img)


# In[11]:


img = mpimg.imread('./traindata/tomato/tomato_0384.jpg')
print(img.shape)
plt.imshow(img)


# In[12]:


img = mpimg.imread('./traindata/tomato/tomato_0250.jpg')
print(img.shape)
plt.imshow(img)


# # Some Images strawberry Class

# In[13]:


img = mpimg.imread('./traindata/strawberry/strawberry_0110.jpg')
print(img.shape)
plt.imshow(img)


# In[14]:


img = mpimg.imread('./traindata/strawberry/strawberry_0093.jpg')
print(img.shape)
plt.imshow(img)


# In[15]:


img = mpimg.imread('./traindata/strawberry/strawberry_0399.jpg')
print(img.shape)
plt.imshow(img)


# In[16]:


img = mpimg.imread('./traindata/strawberry/strawberry_0510.jpg')
print(img.shape)
plt.imshow(img)


# # Training data Information

# In[17]:


train_categories = []
train_samples = []
for i in os.listdir("./traindata/"):
    train_categories.append(i)
    train_samples.append(len(os.listdir("./traindata/"+ i)))
print("Count of Training set is :", sum(train_samples))


# # Distrubution of Fruits with counts in Training Set

# In[18]:


fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 30
fig_size[1] = 30
plt.rcParams["figure.figsize"] = fig_size

index = np.arange(len(train_categories))
plt.bar(index, train_samples, color='orange')
plt.xlabel('Classes', fontsize=25)
plt.ylabel('Count of each Class data', fontsize=25)
plt.xticks(index, train_categories, fontsize=15, rotation=90)
plt.title('Distrubution of Fruits Classes in Data Set', fontsize=35)
plt.show()


# # Data Preprocessing

# In[23]:


import skimage.transform
train_dir = 'traindata/' 
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


# In[24]:


X_test, y_test= get_data(train_dir)
X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.1)
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)


# # Building CNN model

# In[31]:


model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 2,input_shape=(300,300,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 32,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 64,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 128,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(3,activation = 'softmax'))
model.summary()


# # Tuning the CNN model and Training

# In[36]:


batch_size = 20
epochs=10
loss_function='categorical_crossentropy'
optimization='rmsprop'
Metrics=['accuracy']
model.compile(loss=loss_function,
              optimizer=optimization,
              metrics=Metrics)


# In[41]:


history = model.fit(X_train,y_train1,
        batch_size = batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test1))


# # Saving the Model

# In[42]:


model.save('model.h5')


# # Evaluate and  test accuracy

# In[53]:


score = model.evaluate(X_test, y_test1, verbose=0)
print('\n', 'Test accuracy:', round(score[1],2))

