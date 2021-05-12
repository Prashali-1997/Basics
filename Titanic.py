# -*- coding: utf-8 -*-
"""
Created on Wed May 12 12:19:39 2021

@author: Lenovo
"""

#Importing data
import pandas as pd #loading entire panda library in out script
import matplotlib.pyplot as plt

train = pd.read_csv(r'D:\Kaggle\Titanic\train.csv')
train.head()


test = pd.DataFrame()    #created blank dataframe
test = pd.read_csv(r'D:\Kaggle\Titanic\test.csv')

#Data expolatory analysis
train.shape     #no. of rows & columns

#Counting survivers in train dataset: red - dead; green - alive

train['Survived'].value_counts()
plt.bar(list(train['Survived'].value_counts().keys()), list(train['Survived'].value_counts()), color = ['red', 'green'])
plt.title("Survivor Count")
plt.xlabel("Survivers")
plt.ylabel("Count")

#Counting Pclass in train dataset: 1- green, 2- blue, 3- orange

train['Pclass'].value_counts()
plt.bar(list(train['Pclass'].value_counts().keys()), list(train['Pclass'].value_counts()), color = ['green', 'blue', 'orange'])
plt.title("Pclass Count")
plt.xlabel("Pclass")
plt.ylabel("Count")

#Counting sex in train dataset: male - blue; pink - female

train['Sex'].value_counts()
plt.bar(list(train['Sex'].value_counts().keys()), list(train['Sex'].value_counts()), color = ['blue', 'pink' ])
plt.title("Sex Distributtion")
plt.ylabel("Count")

#Age distribution
plt.hist(train['Age'])
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

#Checking null values

sum(train['Survived'].isnull())
sum(train['Age'].isnull())

#Drop null values
titanic_train = train.dropna()

#Building a model: independent variable- Age; dependent variable- Survived
'''
titanic_train.shape
sum(titanic_train['Survived'].isnull())
sum(titanic_train['Age'].isnull())

x = titanic_train[['Age']]
y = titanic_train[['Survived']]

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(x,y)

#Predicting values

sum(test['Age'].isnull())    #Checking null values
test = test.dropna() #dropping null values
sum(test['Age'].isnull())    #Checking null values

xtest = test[['Age']]
test['Survived'] = dtc.predict(xtest)
#test['ypredict'] = dtc.predict(xtest)
'''


x = train[['Sex']]
y = train[['Survived']]

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x = x.apply(LabelEncoder().fit_transform)
x

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(x,y)

#Predicting values

sum(test['Sex'].isnull())    #Checking null values


x = test[['Sex']]
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x = x.apply(LabelEncoder().fit_transform)
x

xtest = x
test['Survived'] = dtc.predict(xtest)
#test['ypredict'] = dtc.predict(xtest)

test.drop(['Pclass','Name','Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 1, inplace = True)
test.reset_index(drop=True, inplace=True)
test.to_csv(r'C:\Users\Lenovo\Downloads\Titanic test.csv')
df_readfile = pd.read_csv(r'C:\Users\Lenovo\Downloads\Titanic test.csv')
print(df_readfile)
