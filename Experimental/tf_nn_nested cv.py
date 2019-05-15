# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:01:13 2019

@author: caspe
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler, GridSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE, RFECV
from fancyimpute import IterativeImputer
import tensorflow as tf
from nested_cv import nested_cv

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

def fill_ii(df):
    df_filled_ii = pd.DataFrame(IterativeImputer().fit_transform(df.values))
    df_filled_ii.columns = df.columns
    df_filled_ii.index = df.index

    return df_filled_ii

def data_engineering(train, test):
    train = train.drop(train.index[0])
    
    cc_data = pd.concat([train, test], sort=True)
    cc_data = cc_data.drop(['Id', 'SalePrice','Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
    
    train["SalePrice"] = np.log1p(train["SalePrice"])
    y = train['SalePrice']
    
    cc_data = pd.get_dummies(cc_data, prefix_sep='_')
    
    cc_data = fill_ii(cc_data)
    
    X_train = cc_data[:train.shape[0]]
    X_test = cc_data[train.shape[0]:]
    
    return X_train,X_test,y

X,X_test,y = data_engineering(train,test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasRegressor

def create_neural_network_model(first_neuron=200,
                                 activation='relu',
                                 kernel_initializer='uniform',
                                 dropout_rate=0,
                                 optimizer='Adam'):
    model = Sequential()
    columns = X.shape[1]
    
    #model.add(Dense(first_neuron, activation=activation, input_shape=(columns,)))
    #model.add(Dense(first_neuron, activation=activation))
    #model.add(Dense(1))
    
    model.add(Dense(first_neuron, kernel_initializer='normal', activation='relu', input_shape=(columns,)))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    
    model.compile(
        loss='mean_squared_error', 
        optimizer = 'adam', 
        metrics=['mean_squared_error']
    )
    
    return model

model = KerasRegressor(build_fn=create_neural_network_model)

# Prepare the Grid
param_grid = dict(epochs=[50,100,150], 
                  batch_size=[512,1024], 
                  optimizer=['Adam', 'Nadam'],
                  dropout_rate=[0.0],
                  activation=['relu', 'elu'],
                  kernel_initializer=['uniform', 'normal'],
                  first_neuron=[100, 150, 200])

outer_score, best_inner_score, best_params = nested_cv(X, y, model, param_grid, 5, 5, sqrt_of_score = True)
print(outer_score, best_inner_score, best_params)
