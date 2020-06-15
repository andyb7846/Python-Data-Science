# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:05:04 2020

@author: Abrown191
"""

#importing libraries
import numpy as np #Importing of Library and Adding and Alias
import matplotlib.pyplot as plt #as plt!
import pandas as pd

#importing dataset
dataset = pd.read_csv('Salary_Data.csv')#remember your quotes!
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#splitting dataset into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/3, random_state = 0)

#feature scaling
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train) #Scale the training data
x_test = sc_x.fit_transform(x_test)#Scale the testing data
"""