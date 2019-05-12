# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:40:28 2019

@author: caspe
"""

import pandas as pd
from fancyimpute import IterativeImputer
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, ParameterSampler
import numpy as np
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from nested_cv import nested_cv

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
models_to_run = [RandomForestRegressor(), xgb.XGBRegressor(), lgb.LGBMRegressor()]
models_param_grid = [ # 1st param grid, corresponding to RandomForestRegressor
                    {
                            'max_depth': [3, None],
                            'n_estimators': np.random.randint(10,20,20)
                    }, 
                    { # 2nd param grid, corresponding to XGBRegressor
                            'colsample_bytree': np.linspace(0.3, 0.5),
                            'n_estimators': np.random.randint(10,20,20)
                    },
                    {
                            'learning_rate': [0.05],
                            'n_estimators': np.random.randint(10,20,20),
                            'num_leaves': np.random.randint(10,30,10),
                            'reg_alpha' : (1,1.2),
                            'reg_lambda' : (1,1.2,1.4)
                    }
                    ]

outer_score = [ [] for i in range(len(models_to_run)) ]
best_inner_score = [ [] for i in range(len(models_to_run)) ]
best_params = [ [] for i in range(len(models_to_run)) ]

for i,model in enumerate(models_to_run):
    outer_score[i], best_inner_score[i], best_params[i] = nested_cv(X, y, model, models_param_grid[i], 
                                                           5, 5, sqrt_of_score = True)

for i,results in enumerate(zip(outer_score, best_inner_score, best_params)):
    print('Outer scores, inner score and best params for model {0}: \n{1}\n{2}\n{3}\n'.format(type(models_to_run[i]).__name__,results[0],results[1],results[2]))