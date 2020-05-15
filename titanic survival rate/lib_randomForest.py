#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:26:20 2020

@author: kathykiutang
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv') 

# fill in all the missing values for both test and train 
titanic_whole = train.append(test, ignore_index = True)

train_index = len(train)

# fill in the missing values for age
titanic_whole['Title'] = titanic_whole.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

normalized_title = { # group differnet title tgt
            'Mr':"Mr",
            'Mrs': "Mrs",
            'Ms': "Mrs",
            'Mme':"Mrs",
            'Mlle':"Miss",
            'Miss':"Miss",
            'Master':"Master",
            'Dr':"Officer",
            'Rev':"Officer",
            'Col':"Officer",
            'Capt':"Officer",
            'Major':"Officer",
            'Lady':"Royalty",
            'Sir':"Royalty",
            'the Countess':"Royalty",
            'Dona':"Royalty",
            'Don':"Royalty",
            'Jonkheer':"Royalty"
            
}

titanic_whole.Title = titanic_whole.Title.map(normalized_title)

grouped = titanic_whole.groupby(['Sex','Title','Pclass'])
print(grouped.Age.median())

titanic_whole.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))

# fill in the missing value for fare
titanic_whole.Fare = titanic_whole.Fare.fillna(titanic_whole.Fare.mean())

# fill in the missing values for cabin 
titanic_whole.Cabin = titanic_whole.Cabin.fillna('u') # unkonwn

# fill in the missing values for embarked
most_embarked = titanic_whole.Embarked.value_counts().index[0]
titanic_whole.Embarked = titanic_whole.Embarked.fillna(most_embarked)

# feature engineering 

# group SibSp and Parch tgt
titanic_whole['FamilySize'] = titanic_whole['SibSp'] + titanic_whole['Parch'] + 1

titanic_whole.Cabin = titanic_whole.Cabin.map(lambda x:x[0])

# convert all object into numerical values 
titanic_whole.select_dtypes('object').columns

titanic_whole.Sex = titanic_whole.Sex.map({"male":0, "female":1})

title_dummies = pd.get_dummies(titanic_whole.Title, prefix = "Title")
cabin_dummies = pd.get_dummies(titanic_whole.Cabin, prefix = "Cabin")
embarked_dummies = pd.get_dummies(titanic_whole.Embarked, prefix = "Embarked")
Pclass_dummies = pd.get_dummies(titanic_whole.Pclass, prefix = "Pclass")

# stack them tgt 
titanic_dummies = pd.concat([titanic_whole, title_dummies, cabin_dummies,
                             embarked_dummies, Pclass_dummies], axis = 1)

titanic_dummies.drop(['Pclass','Title','Cabin','Embarked','Name','Ticket'], axis = 1, inplace = True)

# recover train and test set 
train_x = titanic_dummies[:train_index]
test_x = titanic_dummies[train_index:]

train_x.Survived = train_x.Survived.astype(int)

X = train_x.drop('Survived', axis = 1).values
y = train_x.Survived.values

train_X , val_X , train_Y , val_Y = train_test_split(X , y,test_size = 0.2,shuffle = True)

# find the best grid 
params = dict(
            max_depth = [n for n in range(9,15)],
            min_samples_split = [n for n in range(4, 11)], 
            min_samples_leaf = [n for n in range(2, 5)],     
            n_estimators = [n for n in range(10, 60, 10)],
)

model_forest = RandomForestClassifier()

forest_gs = GridSearchCV(param_grid=params,
                         estimator=model_forest,
                         cv=5)

forest_gs.fit(train_X,train_Y)

prediction = forest_gs.predict(val_X)

print(mean_absolute_error(prediction, val_Y))

# predict on the test set
X_test = test_x.drop('Survived', axis = 1).values

test_prediction = forest_gs.predict(X_test)

passengerid = test.PassengerId

submission = pd.DataFrame({"PassengerId": passengerid, "Survived": test_prediction})
submission.to_csv('submission_kathy', index = False)






