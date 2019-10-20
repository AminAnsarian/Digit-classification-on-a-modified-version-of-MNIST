#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 23:05:49 2019

@author: amin
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.layers import Activation

def CNN1(optimizer='adadelta'):
    classifier = Sequential()
      
    classifier.add(Conv2D(64, (9, 9), input_shape = (64, 64, 1), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
      
    classifier.add(Conv2D(64, (5, 5), padding = 'same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
    classifier.add(Dropout(rate = 0.25))
      
      
    classifier.add(Conv2D(128, (5, 5), padding = 'same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
    classifier.add(Dropout(rate = 0.25))
      
    classifier.add(Conv2D(256, (3, 3), padding = 'same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
    classifier.add(Dropout(rate = 0.25))
      
    classifier.add(Conv2D(512, (3, 3), padding = 'same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
    classifier.add(Dropout(rate = 0.4))
      
    classifier.add(Conv2D(512, (3, 3), padding = 'same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
    classifier.add(Dropout(rate = 0.4))
      
    classifier.add(Flatten())
    classifier.add(Dense(units = 1024, activation = 'relu'))
    classifier.add(Dropout(rate = 0.4))
    classifier.add(Dense(units = 512, activation = 'relu'))
    classifier.add(Dropout(rate = 0.25))
    classifier.add(Dense(units = 10, activation = 'softmax'))
    classifier.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

def CNN2():
    ##implementing simplenet

    classifier = Sequential()
    classifier.add(Conv2D(64, (9, 9),use_bias=True,strides=1, input_shape = (64, 64, 1), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
     
    classifier.add(Conv2D(64, (5, 5),use_bias=True, strides=1,padding = 'same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
    classifier.add(Dropout(rate = 0.4))
     
     
    classifier.add(Conv2D(128, (5, 5),use_bias=True,strides=1, padding = 'same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
    classifier.add(Dropout(rate = 0.4))
     
    classifier.add(Conv2D(256, (3, 3),use_bias=True,strides=1,padding = 'same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
    classifier.add(Dropout(rate = 0.4))
     
    classifier.add(Conv2D(512, (3, 3),use_bias=True, strides=1, padding = 'same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
    classifier.add(Dropout(rate = 0.4))
     
    classifier.add(Conv2D(512, (3, 3),use_bias=True, strides=1,padding = 'same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
    classifier.add(Dropout(rate = 0.4))
     
    classifier.add(Flatten())
    classifier.add(Dense(units = 1024, activation = 'relu'))
    classifier.add(Dropout(rate = 0.4))
    classifier.add(Dense(units = 512, activation = 'relu'))
    classifier.add(Dropout(rate = 0.25))
    classifier.add(Dense(units = 10, activation = 'softmax'))
    
    classifier.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier