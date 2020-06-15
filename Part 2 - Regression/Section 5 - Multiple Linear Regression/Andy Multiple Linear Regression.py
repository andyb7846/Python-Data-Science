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
dataset = pd.read_csv('50_Startups.csv')#remember your quotes!
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding Categorical Data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

#avoiding dummy variable trap - IMPORTANT!
x = x [:, 1:]

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
#Fitting multiple linear regressions to the training set.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) #Fitting this PARTICULAR regression.

#Predicting Test Set Results
y_pred = regressor.predict(x_test) #Predicting the profit of the startups: VECTOR

#Building optimal model using backwards elimination
import statsmodels.api as sm
x=np.append(arr = np.ones((50,1)).astype(int),values = x, axis = 1)
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog =x_opt).fit()
regressor_OLS.summary()
x_opt = x[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog =x_opt).fit()
regressor_OLS.summary()
x_opt = x[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog =x_opt).fit()
regressor_OLS.summary()
x_opt = x[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog =x_opt).fit()
regressor_OLS.summary()
x_opt = x[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog =x_opt).fit()
regressor_OLS.summary()

#Backward Elimination in Python with p-values
import statsmodels.api as sm
def backwardEliminationP(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
x_opt = X[:, [0, 1, 2, 3, 4, 5]]
x_Modeled = backwardElimination(X_opt, SL)

#Backward Elimination with p-values and Adjusted R Squared
import statsmodels.api as sm
def backwardEliminationPRSQ(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
x_Modeled = backwardElimination(x_opt, SL)

