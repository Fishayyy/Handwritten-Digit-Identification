import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt
import itertools

import warnings
warnings.filterwarnings('ignore')

# SET THE IMAGE SIZE
IMAGE_SIZE = 28

def get_data(train_test, processed_unprocessed):
  cwd = os.getcwd()
  data = []

  path = os.path.realpath(f'processed_images/{train_test}/{processed_unprocessed}/')
  for filename in os.listdir(path):
      #Read in Image
      filepath = f"{path}/{filename}"

      img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

      #Resize Image
      desired_size = IMAGE_SIZE
      old_size = img.shape[:2] 

      ratio = float(desired_size)/max(old_size)
      new_size = tuple([int(x*ratio) for x in old_size])

      img = cv2.resize(img, (new_size[1], new_size[0]))

      delta_w = desired_size - new_size[1]
      delta_h = desired_size - new_size[0]
      top, bottom = delta_h//2, delta_h-(delta_h//2)
      left, right = delta_w//2, delta_w-(delta_w//2)

      color = [255, 255, 255]
      img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

      data.append([img, filename[0]])

  return data

choices = ["binary", "grayscale"]
user_input = ""

print("Which dataset would you like to train on?")
while user_input not in choices:
  user_input = input("\'binary\'/\'grayscale\'?:")

train_data = get_data("train", user_input)
test_data = get_data("test", user_input)

X_train = []
Y_train = []
X_test = []
Y_test = []

for feature, label in train_data:
  X_train.append(feature)
  Y_train.append(label)

for feature, label in test_data:
  X_test.append(feature)
  Y_test.append(label)

# Normalize the data
X_train = np.array(X_train) / 255.0
X_test = np.array(X_test) / 255.0

X_train = X_train.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
X_test = X_test.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
Y_train = to_categorical(Y_train, num_classes = 10)
Y_test = to_categorical(Y_test, num_classes= 10)

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
learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

epochs = 30 
batch_size = 8

generator = ImageDataGenerator(
        rotation_range=12,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1)  # randomly shift images vertically (fraction of total height)

generator.fit(X_train)

# Fit the model
history = model.fit_generator(generator.flow(X_train,Y_train, 
                              batch_size=batch_size),
                              epochs = epochs, 
                              steps_per_epoch=X_train.shape[0] // batch_size, 
                              callbacks=[learning_rate_reduction])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 
