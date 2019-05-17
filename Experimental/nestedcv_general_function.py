# -*- coding: utf-8 -*-
"""
Created on Sat May  4 01:50:41 2019

@author: caspe
"""
import pandas as pd
from fancyimpute import IterativeImputer
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from nested_cv import NestedCV
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasRegressor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

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

def create_neural_network_model(first_neuron=64,
                                activation='relu',
                                optimizer='Adam',
                                dropout_rate=0.1):
    model = Sequential()
    columns = X.shape[1]
    
    model.add(Dense(64, activation=activation, input_shape=(columns,)))
    model.add(Dense(128, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(8, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    
    model.compile(
        loss='mean_squared_error', 
        optimizer = 'adam', 
        metrics=['mean_squared_error']
    )
    
    return model


models_to_run = [KerasRegressor(build_fn=create_neural_network_model,verbose=0),RandomForestRegressor(), xgb.XGBRegressor()]

models_param_grid = [
                    { # 1st param grid, corresponding to KerasRegressor
                            'epochs' :              [50,100,150,200],
                            'batch_size' :          [512,1024],
                            'optimizer' :           ['Adam', 'Nadam'],
                            'dropout_rate' :        [0.0],
                            'activation' :          ['relu', 'elu'],
                            'first_neuron' :        [100, 150, 200]
                    },
                    { # 2nd param grid, corresponding to RandomForestRegressor
                            'max_depth': [3, None],
                            'n_estimators': [100,200,300,400,500,600,700,800,900,1000],
                            'max_features' : [50,100,150,200]
                    }, 
                    { # 3rd param grid, corresponding to XGBRegressor
                            'learning_rate': [0.05],
                            'colsample_bytree': np.linspace(0.3, 0.5),
                            'n_estimators': [100,200,300,400,500,600,700,800,900,1000],
                            'reg_alpha' : (1,1.2),
                            'reg_lambda' : (1,1.2,1.4)
                    }
                    ]
NUM_TRIALS = 50

RF_scores = []
XGB_scores = []
NN_scores = []

for trial in range(NUM_TRIALS):
    for i,model in enumerate(models_to_run):
        nested_CV_search = NestedCV(model=model, params_grid=models_param_grid[i], outer_kfolds=5, inner_kfolds=5, 
                          cv_options={'sqrt_of_score':True, 'randomized_search_iter':30})
        nested_CV_search.fit(X=X,y=y)
        model_param_grid = nested_CV_search.best_params
        print('\nCumulated best parameter grids was:\n{0}'.format(model_param_grid))
        
        gscv = GridSearchCV(estimator=model,param_grid=model_param_grid,scoring='neg_mean_squared_error',cv=5)
        gscv.fit(X,y)
        
        print('\nFitting with optimal parameters:\n{0}'.format(gscv.best_params_))
        gscv.predict(X_test)
        score = np.sqrt(-gscv.best_score_)
        
        if(type(model).__name__ == 'KerasRegressor'):
            NN_scores.append(score)
        elif(type(model).__name__ == 'RandomForestRegressor'):
            RF_scores.append(score)
        elif(type(model).__name__ == 'XGBRegressor'):
            XGB_scores.append(score)
        
        print('\nFinal score for {0} was {1}'.format(type(model).__name__,score))

plt.figure()

rf, = plt.plot(RF_scores, color='b')
xgb, = plt.plot(XGB_scores, color='r')
nn, = plt.plot(NN_scores, color='g')

plt.legend([rf, xgb, nn],
           ["Random Forest", "XGBoost", "Neural Networks"],
           bbox_to_anchor=(0, .4, .5, 0))

plt.title('Test scores as RMSLE with hyperparameter optimization',
          x=.5, y=1.1, fontsize="15")