#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 19:43:49 2020

@author: kathykiutang
"""

import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log

final_train = pd.read_csv('final_train.csv')

# dataset = {'Taste':['Salty','Spicy','Spicy','Spicy','Spicy','Sweet','Salty','Sweet','Spicy','Salty'],
#         'Temperature':['Hot','Hot','Hot','Cold','Hot','Cold','Cold','Hot','Cold','Hot'],
#         'Texture':['Soft','Soft','Hard','Hard','Hard','Soft','Soft','Soft','Soft','Hard'],
#         'Eat':['No','No','Yes','No','Yes','Yes','No','Yes','Yes','Yes']}

# df = pd.DataFrame(dataset,columns=['Taste','Temperature','Texture','Eat'])


entropy_node = 0  #Initialize Entropy
values = final_train.Survived.unique()  #Unique objects - 'Yes', 'No'
for value in values:
    fraction = final_train.Survived.value_counts()[value]/len(final_train.Survived)  
    entropy_node += -fraction*np.log2(fraction)
    
# attribute = 'Sex'
# target_variables = final_train.Survived.unique()  #This gives all 'Yes' and 'No'
# variables = final_train[attribute].unique()    #This gives different features in that attribute (like 'Sweet')
# entropy_attribute = 0
# for variable in variables:
#     entropy_each_feature = 0
#     for target_variable in target_variables:
#         num = len(final_train[attribute][final_train[attribute]==variable][final_train.Survived ==target_variable]) #numerator
#         den = len(final_train[attribute][final_train[attribute]==variable])  #denominator
#         fraction = num/(den+eps)  #pi
#         entropy_each_feature += -fraction*log(fraction+eps) #This calculates entropy for one feature like 'Sweet'
#     fraction2 = den/len(final_train)
#     entropy_attribute += -fraction2*entropy_each_feature   #Sums up all the entropy ETaste

def find_entropy(df):
    Class = df.keys()[0]   #To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy
  
  
def find_entropy_attribute(df,attribute):
  Class = df.keys()[0]   #To make the code generic, changing target variable class name
  target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps)
      fraction2 = den/len(df)
      entropy2 += -fraction2*entropy
  return abs(entropy2)


def find_winner(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[1:]:
#         Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    return df.keys()[1:][np.argmax(IG)]


def get_subtable(df, node,value):
  return df[df[node] == value].reset_index(drop=True)


def buildTree(df,tree=None): 
    print('1')
    Class = df.keys()[0]   #To make the code generic, changing target variable class name
    
    #Here we build our decision tree

    #Get attribute with maximum information gain
    node = find_winner(df)
    
    #Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    attValue = np.unique(df[node])
    
    #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[node] = {}
        print('2')
    
   #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 

    for value in attValue:
        print('3')
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable['Survived'],return_counts=True)                        
        
        if len(counts)==1:#Checking purity of subset
            print('4') 
            tree[node][value] = clValue[0]
        if len(counts) != 1:        
            print('5')
            tree[node][value] = buildTree(subtable) #Calling the function recursively 
            
                   
    return tree


"""
find the 10 highlest IG and then reform the df and train it, record this
"""

    





















