# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:48:48 2022

@author: anuaq
"""
import numpy as np
import pandas as pd
import pickle
df=pd.read_excel('iris.xls')

X = df[['SL','SW','PL','PW']]
y = df['Classification']

from sklearn.model_selection import train_test_split
#Split the Data into Training and Testing sets with test size as 30%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100,random_state=0,criterion='entropy') 
 # fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

pickle.dump(clf,open('model.pickle','wb'))