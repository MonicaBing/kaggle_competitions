#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:27:26 2020

@author: kathykiutang
"""

# CNN - keras 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import np_utils
from sklearn.metrics import f1_score


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission_original.csv')

X = train.drop(['label'], axis = 1)
y = train['label']


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101) #val(train)


X = X.to_numpy().reshape(42000,28,28,1)
y = np_utils.to_categorical(y)


model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, 3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=3)

test_samples = test.to_numpy().reshape(28000,28,28,1)

pred = model.predict(test_samples) 

# post processing, convert thatt to binary
    
for i in range(0,len(test)):
    submission['Label'][i] = np.argmax(pred[i])

submission.to_csv('sample_submission.csv', index = False)

## if everything works, include the big set and predit the stuff  ----

