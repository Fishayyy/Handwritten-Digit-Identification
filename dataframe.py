import os
import pandas as pd
import numpy as np
import cv2
import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical

import warnings
warnings.filterwarnings('ignore')

IMAGE_SIZE = 28

target_directories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  
cwd = os.getcwd()
df = pd.DataFrame(columns=['image_data', 'label'])
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

        color = [128, 128, 128]
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        data = np.matrix(img)
        df.loc[filename] = [data, dir]

X = df['image_data']
X_train = X/255.0
X_train = X_train.values.reshape(-1,28,28,1)
y = df['label']
Y_train = to_categorical(y, num_classes = 10)

model = Sequential()

model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28)))
model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))


model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(layers.Dropout(0.25))


model.add(layers.Flatten())
model.add(layers.Dense(256, activation = "relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation = "softmax"))

# model.add(layers.Conv2D(3, (3,3), strides=1, input_shape=(28,28)))
# model.add(layers.MaxPool2D(pool_size=(2,2), strides=1))
# model.add(layers.Conv2D(2, (2,2), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation = "relu"))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(10, activation = "softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optimizer , loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 8 
batch_size = 32

model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=epochs)
