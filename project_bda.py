#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 18:04:03 2017

@author: Flore
"""
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import cross_validation
from toolbox_bda import *

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


##############################################################################
### LOAD DATA ################################################################
##############################################################################
data_mat = pd.read_csv('student-mat.csv', delimiter=';')
data_por = pd.read_csv('student-por.csv', delimiter=';')
print("Mathematics")
print(data_mat.describe())
print("Portuguese")
print(data_por.describe())

#Creating X/y & selecting the wanted features for MAT
y_mat = data_mat.values[:,-1]
X_mat = data_mat.values[:,:-1]
X_mat_selected = np.delete(X_mat, [0,3,4,5,8,9,10,11,15,16,17,19,20,21,22,23,24,25], 1) 
#Features kept: 
#0 : Sex, 1: Age, 2: Medu, 3: Fedu, 4: traveltime, 5: studytime, 6: failures, 
#7: activities, 8: Dalc, 9: Walc, 10: health, 11: absences, 12: G1, 13: G2
#Separating data for students that passed (G3>10) and others
y_mat_pass = y_mat[y_mat>10]
X_mat_pass = X_mat_selected[y_mat>10]
#And deleting G3 = 0 (students absent)
y_mat_int = y_mat[y_mat<=10]
y_mat_fail = y_mat_int[y_mat_int!=0]
X_mat_int = X_mat_selected[y_mat<=10]
X_mat_fail = X_mat_int[y_mat_int!=0]
#print(data_mat.columns)

#Creating X and y + selecting the wanted features for POR
y_por = data_por.values[:,-1]
X_por = data_por.values[:,:-1]
X_por_selected = np.delete(X_por, [0,3,4,5,8,9,10,11,15,16,17,19,20,21,22,23,24,25], 1) 
#Features kept: 
#0 : Sex, 1: Age, 2: Medu, 3: Fedu, 4: traveltime, 5: studytime, 6: failures, 
#7: activities, 8: Dalc, 9: Walc, 10: health, 11: absences, 12: G1, 13: G2
#Separating data for students that passed (G3>10) and others
y_por_pass = y_por[y_por>10]
X_por_pass = X_por_selected[y_por>10]
#And deleting G3 = 0 (students absent)
y_por_int = y_por[y_por<=10]
y_por_fail = y_por_int[y_por_int!=0]
X_por_int = X_por_selected[y_por<=10]
X_por_fail = X_por_int[y_por_int!=0]
#print(data_por.columns)


##############################################################################
### DATA VISUALIZATION #######################################################
##############################################################################
#Plots
plt.figure(2, figsize=(5,5))
plt.scatter(X_mat_pass[:,0],y_mat_pass, alpha = 0.1, s=100, label='Passed')
plt.scatter(X_mat_fail[:,0],y_mat_fail, alpha = 0.1, s=100, label='Failed')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Sex')
plt.ylabel('Grade Term 3')

##############################################################################
### PRE-PROCESSING AND FEATURES ENGINEERING ##################################
##############################################################################

##############################################################################
### DIMENSION REDUCTION ######################################################
##############################################################################
X = prepro(data_mat)
y = y_mat
X = dim_reduc(X,y,5)

##############################################################################
### CROSS VALIDATION #########################################################
##############################################################################
X, X_test, y, y_test = cross_validation.train_test_split(X,y, test_size=0.4, random_state=42)

##############################################################################
### MODELING #################################################################
##############################################################################

##############################################################################
#Linear Regression normalized 
LR= LinearRegression(normalize=True)
#LR.fit(X,y)
#y_test = LR.predict(X_test)
#my_solution = pd.DataFrame(y_test)
#my_solution.to_csv("my_solution_linear.csv")
print('Linear Regression R^2 : ' , mean(cross_val_score(LR,X,y,cv=10, scoring = 'r2')))
print('Linear Regression RMSE : ' , mean(sqrt(-cross_val_score(LR,X,y,cv=10, scoring = 'neg_mean_squared_error'))))

##############################################################################
#Ridge Regression normalized
ridge = Ridge(normalize=True, alpha = 1)
#ridge.fit(X,y)
#y_test = ridge.predict(X_test)
#my_solution = pd.DataFrame(y_test)
#my_solution.to_csv("my_solution_ridge.csv")
print('Ridge Regression R^2 : ' , mean(cross_val_score(ridge,X,y,cv=10, scoring = 'r2')))
print('Ridge Regression RMSE : ' , mean(sqrt(-cross_val_score(ridge,X,y,cv=10, scoring = 'neg_mean_squared_error'))))

##############################################################################
#K-nearest neighbors
knn = KNeighborsRegressor(n_neighbors = 1, algorithm = 'brute', p = 1)
#knn.fit(X,y)
#y_test = knn.predict(X_test)
#my_solution = pd.DataFrame(y_test)
#my_solution.to_csv("my_solution_knn.csv")
print('1-Nearest Neighbor R^2 : ' , mean(cross_val_score(knn,X,y,cv=10, scoring = 'r2')))
print('1-Nearest Neighbor RMSE : ' , mean(sqrt(-cross_val_score(knn,X,y,cv=10, scoring = 'neg_mean_squared_error'))))

##############################################################################
#Decision Tree Regressor
dtr = DecisionTreeRegressor(min_samples_leaf=6, max_depth=10)
#dtr.fit(X,y)
#y_test = dtr.predict(X_test)
#my_solution = pd.DataFrame(y_test)
#my_solution.to_csv("my_solution_dtr.csv")
print('Decision Tree Regressor R^2 : ' , mean(cross_val_score(dtr,X,y,cv=10, scoring = 'r2')))
print('Decision Tree Regressor RMSE : ' , mean(sqrt(-cross_val_score(dtr,X,y,cv=10, scoring = 'neg_mean_squared_error'))))

##############################################################################
#AdaBoost Regressor on the decision tree regressor
abr = AdaBoostRegressor(dtr,n_estimators =10)
#abr.fit(X,y)
#y_test = abr.predict(X_test)
#my_solution = pd.DataFrame(y_test)
#my_solution.to_csv("my_solution_abr.csv")
print('AdaBoost Regressor R^2 : ' , mean(cross_val_score(abr,X,y,cv=10, scoring = 'r2')))
print('AdaBoost Regressor RMSE : ' , mean(sqrt(-cross_val_score(abr,X,y,cv=10, scoring = 'neg_mean_squared_error'))))




















