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


##############################################################################
### FUNCTIONS ################################################################
##############################################################################

#Cross validation
def compute_score(classifier,X,y):
    xval = cross_val_score(classifier,X,y,cv=10)
    return mean(xval)

#Dimension reduction
def dim_reduc(X,y,nb):
    ch2 = SelectKBest(chi2,k=nb)
    X = ch2.fit_transform(X, y)
    return X

#Preprocessing: 
def prepro(X):

    #Converting binary variables to 0/1.
    variables_map={'no':0, 'yes':1}
    X['schoolsup']=X['schoolsup'].map(variables_map)
    X['higher']=X['higher'].map(variables_map)
    X['internet']=X['internet'].map(variables_map)
    
    address_map={'R':0, 'U':1}
    X['address']=X['address'].map(address_map)
    
    #We assign num values to Mjob:
    X["Mjob"][X["Mjob"] == "at_home"] = 0 
    X["Mjob"][X["Mjob"] == "health"] = 1 
    X["Mjob"][X["Mjob"] == "other"] = 2 
    X["Mjob"][X["Mjob"] == "services"] = 3 
    X["Mjob"][X["Mjob"] == "teacher"] = 4
    
    #We assign num values to Fjob:
    X["Fjob"][X["Fjob"] == "at_home"] = 0 
    X["Fjob"][X["Fjob"] == "health"] = 1 
    X["Fjob"][X["Fjob"] == "other"] = 2 
    X["Fjob"][X["Fjob"] == "services"] = 3 
    X["Fjob"][X["Fjob"] == "teacher"] = 4
    
    #We assign num values to guardian:
    X["guardian"][X["guardian"] == "father"] = 0 
    X["guardian"][X["guardian"] == "mother"] = 1 
    X["guardian"][X["guardian"] == "other"] = 2 

    
    #Split some variables into dummies.
    sex_split = pd.get_dummies(X['sex'])
    address_split = pd.get_dummies(X['address'])
    Medu_split = pd.get_dummies(X['Medu'])
    Fedu_split = pd.get_dummies(X['Fedu'])
    Mjob_split = pd.get_dummies(X['Mjob'])
    Fjob_split = pd.get_dummies(X['Fjob'])
    guardian_split = pd.get_dummies(X['guardian'])
    traveltime_split = pd.get_dummies(X['sex'])
    studytime_split = pd.get_dummies(X['address'])
    failures_split = pd.get_dummies(X['Medu'])
    schoolsup_split = pd.get_dummies(X['Fedu'])
    higher_split = pd.get_dummies(X['Mjob'])
    internet_split = pd.get_dummies(X['Fjob'])
    goout_split = pd.get_dummies(X['guardian'])
    Dalc_split = pd.get_dummies(X['Medu'])
    health_split = pd.get_dummies(X['Fedu'])
    
    #Adding dummies to X. 
    X = X.join(sex_split)
    X = X.join(address_split)
    X = X.join(Medu_split)
    X = X.join(Fedu_split)
    X = X.join(Mjob_split)
    X = X.join(Fjob_split)
    X = X.join(guardian_split)
    X = X.join(traveltime_split)
    X = X.join(studytime_split)
    X = X.join(failures_split)
    X = X.join(schoolsup_split)
    X = X.join(higher_split)
    X = X.join(internet_split)
    X = X.join(goout_split)
    X = X.join(Dalc_split)
    X = X.join(health_split) 
    
    #Removing unnecessary columns (from dummies, plus those with no weak correlation with target G3). 
    to_del = ['school', 'famsize', 'Pstatus','reason','famsup', 'paid', 'activities', 'nursery', 'romantic', 'famrel',
              'freetime', 'Walc', 'absences', 'sex', 'address','Medu','Fedu', 'Mjob', 'Fjob', 'guardian', 'traveltime', 'studytime', 'failures', 
              'schoolsup', 'higher', 'internet', 'goout', 'Dalc', 'health'] 
    for col in to_del : del X[col]
    
    
    return X