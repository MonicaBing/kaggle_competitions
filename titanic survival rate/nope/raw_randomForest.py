#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:21:03 2020

@author: kathykiutang
"""

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from numpy import log2 as log
esp = np.finfo(float).eps
import pprint

def age(input):
    for n in range (10,81,10): # 0 9 19 29.....79
        if input <= n:
            return n/10

# import pre-procesed data (titanic)
test_x_og = pd.read_csv('test_x')
train_x_og = pd.read_csv('train_x')

# make sure everythng is binary

passengerId = test_x_og.PassengerId

test_train = train_x_og.append(test_x_og, ignore_index = True)

test_train = test_train.drop(['PassengerId','Fare'], axis = 1)

SibSp_dummies = pd.get_dummies(test_train.SibSp, prefix = "SibSp")
Parch_dummies = pd.get_dummies(test_train.Parch, prefix = "Parch")
FamilySize_dummies = pd.get_dummies(test_train.FamilySize, prefix = "FamilySize")
test_train = pd.concat([test_train, SibSp_dummies, Parch_dummies, FamilySize_dummies], axis = 1)

test_train.Age = test_train['Age'].apply(age)
Age_dummies = pd.get_dummies(test_train.Age, prefix = "Age")

test_train = pd.concat([test_train, Age_dummies], axis =1)

test_train = test_train.drop(['SibSp', 'Parch', 'FamilySize', 'Age'], axis = 1)

# check if if the input are all binary

# output_2 = np.zeros((891,2))
# output_3 = np.zeros((1309-891,3))
# for i in range (891):
#     output_2[i,:] = test_train.loc[i,:].unique()
# n = 0
# for i in range(891,1309):
#     output_3[n,:] = test_train.loc[i,:].unique()
#     n = n + 1

# split test and train 

train_index = len(train_x_og)

final_train = test_train[:train_index]
final_train.to_csv('final_train.csv', index = False)


final_test = test_train[train_index:]


final_test = final_test.drop(['Survived'], axis =1) # dont touch this

"""
buid a decision tree, HERERERER

cost function = gini index
"""

# calculate the entropy at the node, the very top, the one 


def find_entropy(final_train):
    Class = final_train.keys()[0]
    entropy_node = 0
    values = final_train[Class].unique() 
    
    for value in values:# 0,1
        fraction = final_train[Class].value_counts()[value]/len(final_train[Class])
        entropy_node += -fraction*np.log2(fraction)
    return entropy_node
    
    
def find_entropy_feature(final_train, feature):
    #feature = 'Sex'
    Class = final_train.keys()[0]
    target_variables = final_train[Class].unique() # 0,1 
    variables = final_train[feature].unique() # Sex
    entropy_feature =0
    
    for variable in variables: # 0 --> 1 , 0,1, feature
        entropy_each_feature = 0
        for target_variable in target_variables: # 0 --> 1, 0,1, survived
            num = len(final_train[feature][final_train[feature] == variable][final_train[Class] == target_variable])
            den = len(final_train[feature][final_train[feature]==variable])
            fraction = num/(den + esp)
            entropy_each_feature += -fraction*log(fraction + esp)
        fraction2 = den/len(final_train)
        entropy_feature += -fraction2*entropy_each_feature # the orgiginal post has a negative in front

    return abs(entropy_feature)
    
def get_subtable(final_train, node, value):
    return final_train[final_train[node] == value].reset_index(drop = True)

def find_winner(final_train):
    #entropy_feature = []
    information_gain = []

    for key in final_train.keys()[1:]: # ignore survived (label)
        IG = find_entropy(final_train) - find_entropy_feature(final_train, key)
        information_gain.append(IG)
    return final_train.keys()[1:][np.argmax(information_gain)],IG


# def predict(test, tree):
#     for nodes in tree.keys():
#         value = test[nodes]
#         tree = tree[nodes][value]
#         prediction = 0 # initialisation 
        
#         if type(tree) is dict: # is there is a branch 
#             prediction = predict(test,tree) # find the next layer
#             print('next layer')
#         else:
#             prediction = tree # the value 
#             print('one node done')
#             break;
#     return prediction



# for i in range (len(final_test)): #418
#     test_prediction = predict(final_test.iloc[i], tree)

# submission = pd.DataFrame({"PassengerId": passengerid, "Survived": test_prediction})
    


def buildTree(df,tree=None): 

    #Get attribute with maximum information gain
    node,IG = find_winner(df)
    
    #Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    attValue = np.unique(df[node])
    
    #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
   #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 

    for value in attValue:
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable['Survived'],return_counts=True)                        

        if len(counts) == 1 or IG <= 0.0001: # 100% pure
            tree[node][value] = clValue[0]
            print('terminate')
            
        else: # not 100% pure 
            if IG > 0.0001:
                tree[node][value] = buildTree(subtable)  
                print('extend')
                          
                   
    return tree

  
tree = buildTree(final_train)
#pprint.pprint(buildTree(final_train))



def predict(inst,tree):
    #This function is used to predict for any input variable 
    
    #Recursively we go through the tree that we built earlier

    for nodes in tree.keys():        
        
        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = 0
            
        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break;                            
        
    return prediction


predict_values = [[ ]]

for i in range (len(final_test)):
    predict_values.append(int(predict(final_test.iloc[i],tree)))








