#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:27:57 2020

@author: kathykiutang
"""

import pandas as pd
import numpy as np 
from keras.utils import np_utils

def convulution(input):
    output = np.zeros((len(input),26,26,8)) 
    filters = np.random.rand(8,3,3)/9 
    for k in range (0,8): # for all 8 filters
        for n in range (0,len(input)): # for all images
            for i in range (0,len(input[n])-2,1): # -3 dimensions but python automatically -1 more
                for j in range (0,len(input[n])-2,1): 
                    overlap = input[n][i:i+3, j:j+3]
                    output[n,i,j,k] =  sum(sum(np.multiply(overlap,filters[k]))) # convulution
        print(k)
    return output

def max_pool(input, X):
    
    after_pool = np.zeros((len(X),13,13,8))
    
    a = 0
    b = 0
    
    for k in range (0,8):
        for n in range (0,len(X)):
            for i in range (0,len(input[n,:,:,k])-1,2): #vertical 0 - 26, 13    
                for j in range (0, len(input[n,:,:,k])-1,2): #horizontal 0 - 26, 13
                    region = input[n,:,:,k][i:i+2, j:j+2]
                    after_pool[n,a,b,k] = np.amax(region)
                    b = b+1          
                b=0
                a = a+1
            a = 0
            b = 0
        a = 0
        b = 0
        print(k)
    return after_pool

def sigmoid (x):
    return 1 / (1 + np.exp(-x))


def compute_loss (Y,Y_hat): # loss function
    m = len(y)
    L = -(1./m)*( np.sum( np.multiply(np.log(Y_hat),Y) ) + np.sum( np.multiply(np.log(1-Y_hat),(1-Y)) ) )
    return L 

#---------------------------------------------------------------------------------------



train = pd.read_csv('train.csv') 
X_train = train.drop(['label'], axis = 1).to_numpy().reshape(len(train),28,28)

y = train['label'] # label on 
y = np_utils.to_categorical(y)
#plt.imshow(image0, cmap='gray')

# convulution - step 1 data preprossing

# filters = np.random.rand(8,3,3)/9 
# output_train = np.zeros((len(train),26,26,8)) 

# for k in range (0,8): # for all 8 filters ------------------------------------------
#     for n in range (0,len(X_train)): # for all images
#         for i in range (0,len(X_train[n])-2,1): # -3 dimensions but python automatically -1 more
#             for j in range (0,len(X_train[n])-2,1): 
#                 overlap = X_train[n][i:i+3, j:j+3]
#                 output_train[n,i,j,k] =  sum(sum(np.multiply(overlap,filters[k]))) # convulution
#     print(k)
            
output_train = convulution(X_train)

# max pooling - step 2 data pre propressing -------------------------------------------
    
# after_pool_train = np.zeros((len(train),13,13,8))

# a = 0
# b = 0

# for k in range (0,8):
#     for n in range (0,len(X_train)):
#         for i in range (0,len(output_train[n,:,:,k])-1,2): #vertical 0 - 26, 13    
#             for j in range (0, len(output_train[n,:,:,k])-1,2): #horizontal 0 - 26, 13
#                 region = output_train[n,:,:,k][i:i+2, j:j+2]
#                 after_pool_train[n,a,b,k] = np.amax(region)
#                 b = b+1          
#             b=0
#             a = a+1
#         a = 0
#         b = 0
#     a = 0
#     b = 0
#     print(k)
    
# training
after_pool_train = max_pool(output_train, X_train)

final_train = after_pool_train.reshape(len(train), 13*13*8)

w = np.random.rand(1352,10)*0.01
b = np.zeros((1,1))
learning_rate = 0.02


for i in range (len(train)): # train it 2k times
    z = np.matmul(w.T, final_train[i]) + b
    A = sigmoid(z)
    
    cost = compute_loss(y[i],A)
    dW = (1/1352)*np.matmul(final_train[i].reshape(1352,1), (A-y[i]))
    db = (1/1352)*np.sum(A-y[i], axis =1, keepdims= True)
    
    w = w-learning_rate * dW
    b = b-learning_rate * db
    
    if (i %100 == 0):
        print("Epoch",i, "cost", cost)
   
print("Final cost:", cost)

#---------------------- prediction ----------------------------------------
test = pd.read_csv('test.csv')
X_test = test.to_numpy().reshape(len(test),28,28)
output_test = np.zeros((len(test),26,26,8))

output_test = convulution(X_test)

# for k in range (0,8): # for all 8 filters
#     for n in range (0,len(X_test)): # for all images
#         for i in range (0,len(X_test[n])-2,1): # -3 dimensions but python automatically -1 more
#             for j in range (0,len(X_test[n])-2,1): 
#                 overlap = X_train[n][i:i+3, j:j+3]
#                 output_test[n,i,j,k] =  sum(sum(np.multiply(overlap,filters[k]))) # convulution
#     print(k)

after_pool_test = max_pool(output_test, X_test)


# a = 0
# b=0

# for k in range (0,8):
#     for n in range (0,len(X_test)):
#         for i in range (0,len(output_test[n,:,:,k])-1,2): #vertical 0 - 26, 13    
#             for j in range (0, len(output_test[n,:,:,k])-1,2): #horizontal 0 - 26, 13
#                 region = output_test[n,:,:,k][i:i+2, j:j+2]
#                 after_pool_test[n,a,b,k] = np.amax(region)
#                 b = b+1          
#             b=0
#             a = a+1
#         a = 0
#         b = 0
#     a = 0
#     b = 0
#     print(k)


X_test = after_pool_test.reshape(len(X_test),13*13*8)

prediction= np.matmul(X_test, w)
submission = pd.read_csv('sample_submission_original.csv')

for i in range(0,28000):
    submission['Label'][i] = np.argmax(prediction[i])
    
submission.to_csv('sample_submission.csv', index = False)




