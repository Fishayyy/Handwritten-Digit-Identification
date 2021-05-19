import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

IMAGE_SIZE = 28

target_directories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  
cwd = os.getcwd()
data = []
save_path = f'{cwd}\\processed_images\\'

if not os.path.isdir(save_path):
    os.mkdir(save_path)

for dir in target_directories:
    path = os.path.realpath(f'Data/{dir}/')
    for filename in os.listdir(path):

        #Read in Image
        filepath = f"{path}\\{filename}"
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        #Resize Image
        desired_size = IMAGE_SIZE
        old_size = img.shape[:2] # old_size is in (height, width) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        img = cv2.resize(img, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [255, 255, 255]
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        data.append([img, dir])

X_train = []
Y_train = []

for feature, label in data:
  X_train.append(feature)
  Y_train.append(label)

# Normalize the data
X_train = np.array(X_train) / 255

X_train = X_train.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
Y_train = to_categorical(Y_train, num_classes = 10)

# Split the train and the validation set for the fitting
# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=53)

model = Sequential()

model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation = "relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation = "softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optimizer , loss="categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

epochs = 30 
batch_size = 8

history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs)
