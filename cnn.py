#Importing the necessary libraries
import keras
from keras import backend as K
import numpy as np
import pandas as pd
import models as md
import matplotlib.pyplot as plt
import utils

# %% PreProcess Images

train_labels = pd.read_csv('train_labels.csv')
train_images = pd.read_pickle('train_images.pkl')
shape = train_images.shape
train_data = np.zeros(train_images.shape)
train_data = [utils.threshs(train_images[i],230) for i in range(len(train_images))]
train_data = np.array(train_data)
train_data /= 255
train_labels = pd.read_csv('train_labels.csv')
train_labels = train_labels.to_numpy()
labels = train_labels[:,1]
img_rows = train_data.shape[1]
img_cols = train_data.shape[2]
num_category = 10
if K.image_data_format() == 'channels_first':
    train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
from sklearn.model_selection import train_test_split

# convert class vectors to binary class matrices
labels = keras.utils.to_categorical(labels, num_category)

X_train, X_val, y_train, y_val = train_test_split(train_data, labels, 
                                                  train_size = 0.9, test_size = 0.1)
# %% Main Model Implementation
classifier = md.CNN2()
batch_size = 256
num_epoch = 150
classifier.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=1,
          validation_data=(X_val, y_val))
classifier.save('Cons1.h5')  # creates a HDF5 file 'my_model.h5'
Hdeltadrp4 = classifier.history

# %% Test Data Preparations
test_images = pd.read_pickle('test_images.pkl')
shape = test_images.shape
test_data = np.zeros(test_images.shape)
test_data = [utils.threshs(test_images[i],230) for i in range(len(test_images))]
test_data = np.array(test_data)
test_data /= 255
X_test = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)

# %%Prediction
y_pred = classifier.predict_classes(X_test, batch_size=256)
utils.model_submit(y_pred)
# %% Visualization
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()