#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:52:56 2020

@author: kathykiutang
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission_original.csv")

X = train.drop(['label'], axis = 1)
y = train['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


score = 0
optimal_n = 0

for n in range (1,11,1):
    model = KNeighborsClassifier(n_neighbors = n)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score_new = f1_score(y_test,pred,average='macro')
    
    if score_new > score:
        optimal_n = n
        optimal_pred = pred #for the validation samples
    
    print(n)


#pred = model.predict(X_test)
# confusion_matrix(y_test, pred)
# f1_score(y_test,pred,average='macro')

model = KNeighborsClassifier(n_neighbors = optimal_n)
model.fit(X_train, y_train) #should be the combination of train and validation
pred = model.predict(test)



for i in range(0,len(test),1):
    submission['Label'][i] = pred[i]
    
submission.to_csv('sample_submission.csv', index = False)



#find the optimal n neighbours 
