# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:54:55 2020

@author: abrown191
"""
#Data preprocessing
#Importing the Libraries

import numpy as np #Importing of Library and Adding and Alias
import matplotlib.pyplot as plt #as plt!
import pandas as pd

#Importing the Dataset
#SET A WORKING DIRECTORY FOLDER!!
# file explorer - machine learning a-z - part 1 - Section2etc
# PRESS F5 to set directory
dataset = pd.read_csv('Data.csv')#remember your quotes!
x = dataset.iloc[:, :-1].values    #[LINES, COLUMNS] : = ALL
#dependent variable vector
y = dataset.iloc[:, 3].values

#Missing Data Tutorial -replacing NaNs
#Replace missing data with the mean of the columns that the mean is in.
#           CTRL+I = Info on clicked value

# ------------------------------------------------------------------------MISSING DATA

from sklearn.impute import SimpleImputer #import the imputer
imputer = SimpleImputer(missing_values = np.NaN, strategy = 'mean') #characterise the imputer
imputer = imputer.fit(x[:,1:3]) #Upper band +1 to maximum #fit the imputer to the relevant parts
x[:, 1:3] = imputer.transform(x[:, 1:3]) #set x to display the imputer


#------------------------------------------------------------------------CATEGORICAL DATA

from sklearn.preprocessing import LabelEncoder
labelencoder_x= LabelEncoder()
x[:, 0]=labelencoder_x.fit_transform(x[:, 0])

#------------------------------------------------------------------------DUMMY ENCODING

#---ENCODING MEANS ASSIGNING A NUMBER TO A NAME

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0]) #The Column you wish to split based on answers.

x=onehotencoder.fit_transform(x).toarray() #You dont need to sub specify as you already have.

#Tranforms Y into an encoded variable vector
labelencoder_y= LabelEncoder()
y=labelencoder_y.fit_transform(y)

#-----------------------------------------------------------------------SPLITTING DATASETS -IMPORTANT
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)#0.5 = 50% of the Data goes to the test. The other 50% goes to the train.
#You should usually pick 0.2, for 20% to the test set.

#The training set allows you to train the model to predict the information accurately.
#The test set allows you to ensure what the training set has learned is carried out accurately.

#------------------------------------------------------------------------FEATURE SCALING - IMPORTANT
#Models are based on euclidean distances. Even if models arn't the we still need to do feature scaling.

#Feature Scaling: Feature scaling helps normalise data so that it can then be predicted.
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train) #Scale the training data
x_test = sc_x.fit_transform(x_test)#Scale the testing data

#--------------------------------------------------------------DATA PREPROCESSING TEMPLATE

#importing libraries
import numpy as np #Importing of Library and Adding and Alias
import matplotlib.pyplot as plt #as plt!
import pandas as pd

#importing dataset
dataset = pd.read_csv('Data.csv')#remember your quotes!
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#splitting dataset into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

#feature scaling
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train) #Scale the training data
x_test = sc_x.fit_transform(x_test)#Scale the testing data
"""

#----------------------------------------------------------------SIMPLE LINEAR REGRESSION MODEL


