# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:42:06 2019

@author: caspe
"""

import pandas as pd
from fancyimpute import IterativeImputer
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

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

def multiple_nested_cv(X_df, y_df, models, params_grid, outer_kfolds, 
                       inner_kfolds, outer_metric = mean_squared_error,
                       inner_metric = 'neg_mean_squared_error',
                       sqrt_of_score = False):
    outer_cv = KFold(n_splits=outer_kfolds,shuffle=True)
    inner_cv = KFold(n_splits=inner_kfolds,shuffle=True)
    
    all_params_all_scores = []
    
    outer_score = []
    inner_score = []
    inner_params = []
    variance = []
    
    for index, model in enumerate(models):
        print('\n{0} <-- Running this model now'.format(type(model).__name__))
        
        if(index >= 1):
            outer_score = []
            inner_score = []
            inner_params = []
            variance = []
        
        for (i, (train_index,test_index)) in enumerate(outer_cv.split(X_df,y_df)):
            print('\n{0}/{1} <-- Current outer fold'.format(i+1,outer_kfolds))
            X_train_outer, X_test_outer = X.iloc[train_index], X.iloc[test_index]
            y_train_outer, y_test_outer = y.iloc[train_index], y.iloc[test_index]
            
            GSCV = GridSearchCV(estimator=model,param_grid=params_grid[index],
                                scoring=inner_metric,cv=inner_cv,n_jobs=-1)
            
            # Inner loop: Find best parameters from dict of hyperparameters
            GSCV.fit(X_train_outer,y_train_outer)
            
            # Outer loop: Test best parameters
            pred = GSCV.predict(X_test_outer)
            
            # Append variance
            variance.append(np.var(pred))
            
            # Append best dict of parameters from hyperparameters
            inner_params.append(GSCV.best_params_)
            
            if sqrt_of_score:
                inner_score.append(np.sqrt(-GSCV.best_score_))
                outer_score.append(np.sqrt(outer_metric(y_test_outer,pred)))
            else:
                inner_score.append(-GSCV.best_score_)
                outer_score.append(outer_metric(y_test_outer,pred))
            
            print('Best parameters was: {0}'.format(inner_params[i]))
            print('Outer score: {0}'.format(outer_score[i]))
            print('Inner score: {0}'.format(inner_score[i]))
        
        # Plot score vs variance
        plt.figure()
        plt.subplot(211)
        
        variance_plot, = plt.plot(variance,color='r')
        score, = plt.plot(outer_score, color='b')
        
        plt.legend([variance_plot, score],
                   ["Variance", "Score"],
                   bbox_to_anchor=(0, .4, .5, 0))
        
        plt.title("{0}: Score VS Variance".format(type(model).__name__),
                  x=.5, y=1.1, fontsize="15")
        
        all_params_all_scores.append([inner_params,outer_score,inner_score])
    
    return all_params_all_scores

X,X_test,y = data_engineering(train,test)

models_to_run = [lgb.LGBMRegressor()]
'''{
    'learning_rate': [0.05],
    'n_estimators': [10,20,30],
    'num_leaves': [10,15,20,25,30],
    'boosting_type' : ['gbdt'],
    'colsample_bytree' : np.linspace(0.3, 0.5),
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4]
    }'''
#https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.LGBMRegressor
models_param_grid = [
                    {
                    'learning_rate': [0.05],
                    'n_estimators': (10, 20),# 30, 50, 100, 200, 400, 600, 800, 1000),
                    'num_leaves': (10,20,30),
                    'reg_alpha' : (1,1.2),
                    'reg_lambda' : (1,1.2,1.4)
                    }
                    ]

model_results = multiple_nested_cv(X_df=X, y_df=y, models=models_to_run, 
                                   params_grid=models_param_grid,
                                   outer_kfolds=5, inner_kfolds=5,sqrt_of_score=True)