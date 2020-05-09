#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:50:16 2020

@author: kathykiutang
"""

#buid knn from scartch 

#find optimal afterwards 

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv') # split out train and vlaidation sets
test = pd.read_csv('test.csv') # dont touch it until very end

X = train.drop(['label'], axis = 1)
y = train['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# take out 1000 samples

K = 10 #numbr of cluster
n = 1
pred = []
distance = []

for i in range (0,len(X_test),1):
    temp_test = X_test.iloc[i]
    for j in range (0, len(X_train),1): #compare one test with all train and save them
        distance.append(np.sqrt(sum(temp_test.subtract(X_train.iloc[j]).apply(np.square)))) #----
        
    index = np.argmin(distance) #find the index of the minimum distance
    pred.append(y_train.iloc[index])
    
    print(i)
    distance = []
    
    
# COMPARE THE VALUE WITH y_test for f1, if good proceed

# if okay then use the whole set as training and do that again, CAN TOUCH TEST 
    



# find the optimal n 
    
# work on the real data with test (train = with the validation set as well)


#----
#ßtemp_test = X_test.iloc[0]
#distance.append(np.sqrt(sum(temp_test.subtract(X_train.iloc[0]).apply(np.square))))
