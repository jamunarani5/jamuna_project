# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:57:30 2018

@author: ssn
"""
import pandas as pd 
dataset=pd.read_csv("wine.csv")
x=dataset.iloc[:,0:13].values
y=dataset.iloc[:,13].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
explained_variance=pca.explained_variance_ratio_
print(explained_variance)

