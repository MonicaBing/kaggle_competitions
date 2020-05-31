#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:11:39 2020

@author: kathykiutang
"""

import pandas as pd
from pandas import Series, DataFrame

import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style('whitegrid')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

titanic_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# drop unnecessaey cols

titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test_df = test_df.drop(['Name', 'Ticket'], axis = 1)

# EMBARKED

titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')

# visulisation

sns.factorplot('Embarked', 'Survived', data = titanic_df, size = 4, aspect = 3)

fig, (axis1, axis2, axis3) = plt.subplots(1,3,figsize = (15,5))
sns.countplot(x='Embarked', data = titanic_df, ax=axis1)
sns.countplot(x='Survived', hue = 'Embarked', data = titanic_df, order=[1,0], ax = axis2)

embarked_perc = titanic_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean()
sns.barplot(x='Embarked', y = 'Survived', data = embarked_perc, order=['S', 'C', 'Q'], ax = axis3)

# dummies 

embarked_dummies_titanic = pd.get_dummies(titanic_df['Embarked'])
embarked_dummies_titanic.drop(['S'], axis = 1, inplace = True)

embarked_dummies_test = pd.get_dummies(test_df['Embarked'])
embarked_dummies_test.drop(['S'], axis = 1, inplace= True)

# combine

titanic_df = titanic_df.join(embarked_dummies_titanic)
test_df = test_df.join(embarked_dummies_test)

# FARE - test only 
test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True) 

titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare'] = test_df['Fare'].astype(int)

fare_not_survived = titanic_df['Fare'][titanic_df['Survived'] == 0]
fare_survived = titanic_df['Fare'][titanic_df['Survived'] == 1]

average_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])

titanic_df['Fare'].plot(kind='hist', figsize = (15,3), bins = 100, xlim = (0,50))

average_fare.index.names = std_fare.index.names = ['Survived'] # replace index name 
average_fare.plot( kind = 'bar', legend = False)

# AGE 
fig, (axis1, axis2) = plt.subplots(1,2,figsize = (15,4))

axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# get the average, std and nan sum in Age
average_age_train = titanic_df['Age'].mean()
std_age_train = titanic_df['Age'].std()
count_nan_age_train = titanic_df['Age'].isnull().sum()

average_age_test = test_df['Age'].mean()
std_age_test = test_df['Age'].std()
count_nan_age_test = test_df['Age'].isnull().sum()

rand_train = np.random.randint(average_age_train - std_age_train, average_age_train + std_age_train,
                               size = count_nan_age_train)

rand_test = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test,
                              size = count_nan_age_test)

# plot the original valuees
titanic_df['Age'].dropna().astype(int).hist(bins = 70, ax = axis1)

# fill in the missing values 
titanic_df['Age'][np.isnan(titanic_df['Age'])] = rand_train
test_df['Age'][np.isnan(test_df['Age'])] = rand_test

titanic_df['Age'] = titanic_df['Age'].astype(int)
test_df['Age'] = test_df['Age'].astype(int)

titanic_df['Age'].hist(bins = 70, ax = axis2)

facet = sns.FacetGrid(titanic_df, hue= 'Survived', aspect = 4)
facet.map(sns.kdeplot, 'Age', shade= True)
facet.set(xlim= (0,titanic_df['Age'].max()))
facet.add_legend()


fig, axis1 = plt.subplots(1,1,figsize = (18,4))
average_age = titanic_df[['Age', 'Survived']].groupby(['Age'], as_index = False).mean()
sns.barplot(x = 'Age', y='Survived', data = average_age)

# cabin, too many nan values, drop it 

titanic_df.drop('Cabin', axis =1, inplace = True)
test_df.drop('Cabin', axis = 1, inplace = True)

# family (new parameters)
titanic_df['Family'] = titanic_df['Parch'] + titanic_df['SibSp']
titanic_df['Family'].loc[titanic_df['Family']>0] = 1
titanic_df['Family'].loc[titanic_df['Family']==0] = 0

test_df['Family'] = test_df['Parch'] + test_df['SibSp']
test_df['Family'].loc[test_df['Family']>0] = 1
test_df['Family'].loc[test_df['Family']==0] = 0


titanic_df.drop(['Parch', 'SibSp'], axis =  1)
test_df.drop(['Parch', 'SibSp'], axis = 1)

fig, (axis1, axis2) = plt.subplots(1,2,sharex = True, figsize = (10,5))

sns.countplot(x = 'Family', data = titanic_df, order = [1,0], ax = axis1)

family_perc = titanic_df[['Family','Survived']].groupby('Family', as_index = False).mean()
sns.barplot(x= 'Family', y = 'Survived', data = family_perc, order = [1,0], ax=axis2)

# create a feature person from sex and age 

def get_person (passenger):
    age, sex = passenger 
    return 'child' if age < 16 else sex


titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person, axis = 1)
test_df['Person'] = test_df[['Age','Sex']].apply(get_person, axis = 1)


titanic_df.drop('Sex', axis = 1, inplace = True)
test_df.drop('Sex', axis = 1, inplace = True)

person_dummies_titanic = pd.get_dummies(titanic_df['Person'])
person_dummies_titanic.columns = ['Child', 'Femail', 'Male']
person_dummies_titanic.drop(['Male'], axis = 1, inplace = True)

person_dummies_test = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child', 'Femail', 'Male']
person_dummies_test.drop(['Male'], axis = 1, inplace = True)

titanic_df = titanic_df.join(person_dummies_titanic)
test_df = test_df.join(person_dummies_test)

fig, (axis1, axis2) = plt.subplots(1,2,figsize=(10,5))

sns.countplot(x='Person', data = titanic_df, ax=axis1)

# average survived rate per person
person_perc = titanic_df[['Person','Survived']].groupby('Person', as_index = False).mean()
sns.barplot(x = 'Person', y = 'Survived', data = person_perc, ax=axis2, order = ['male', 'female',
                                                                                 'child'])

titanic_df.drop(['Person'], axis = 1, inplace = True)
test_df.drop(['Person'], axis = 1, inplace = True)

#Pclass

sns.factorplot('Pclass', 'Survived', order=[1,2,3], data = titanic_df)

pclass_dummies_titanic = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies_titanic.columns =['Class1', 'Class2', 'Class3']
pclass_dummies_titanic.drop(['Class3'], axis =1, inplace = True)

pclass_dummies_test = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns =['Class1', 'Class2', 'Class3']
pclass_dummies_test.drop(['Class3'], axis =1, inplace = True)

titanic_df.drop('Pclass', axis = 1, inplace = True)
test_df.drop('Pclass', axis = 1, inplace = True)

titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df = test_df.join(pclass_dummies_test)



#drop Embarked

titanic_df = titanic_df.drop('Embarked', axis = 1)
test_df = test_df.drop('Embarked', axis = 1)

PassengerId = test_df['PassengerId']


# train and test sets

X_train = titanic_df.drop('Survived', axis =1)
Y_train = titanic_df['Survived']
X_test = test_df.drop('PassengerId', axis =1).copy()

# logistic regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
# it automatically train it with X_train, Y_train, then predict it usign X_train, 
#since it is not overfitting, logreg.predict(X_train) wont give Y_train
logreg.score(X_train, Y_train)

# SVM

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svc = svc.predict(X_test)
svc.score(X_train, Y_train) 

# random forest

random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(X_train, Y_train)
Y_pred_rf = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)

# knn

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
knn.score(X_train, Y_train)

# Gaussian naive bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_gauss = gaussian.predict(X_test)
gaussian.score(X_train, Y_train)

# use ranbdom forest since it gives the highest score

submission = pd.DataFrame({
    'PassengerId': PassengerId,
    'Survived': Y_pred
    })

submission.to_csv('titanic3.csv', index = False) # we dont want index

