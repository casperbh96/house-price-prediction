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

def multiple_nested_cv(X_df, y_df, models, params_grid, outer_kfolds, 
                       inner_kfolds, metric = mean_squared_error,
                       sqrt_of_score = False, randomized_search = False,
                       randomized_search_iter = 10):
    outer_cv = KFold(n_splits=outer_kfolds,shuffle=True)
    inner_cv = KFold(n_splits=inner_kfolds,shuffle=True)
    
    outer_score = []
    variance = []
    
    for index, model in enumerate(models):
        print('\n{0} <-- Running this model now'.format(type(model).__name__))
        if(index >= 1):
            outer_score = []
            variance = []
        
        for (i, (train_index,test_index)) in enumerate(outer_cv.split(X_df,y_df)):
            print('\n{0}/{1} <-- Current outer fold'.format(i+1,outer_kfolds))
            X_train_outer, X_test_outer = X.iloc[train_index], X.iloc[test_index]
            y_train_outer, y_test_outer = y.iloc[train_index], y.iloc[test_index]
            inner_params = []
            inner_scores = []
            
            for (j, (train_index_inner,test_index_inner)) in enumerate (inner_cv.split(X_train_outer,y_train_outer)):
                X_train_inner, X_test_inner = X_train_outer.iloc[train_index_inner], X_train_outer.iloc[test_index_inner]
                y_train_inner, y_test_inner = y_train_outer.iloc[train_index_inner], y_train_outer.iloc[test_index_inner]
                best_score = None
                best_grid = {}
                
                #for param_dict in search_function:
                for param_dict in ParameterSampler(param_distributions=params_grid[index],n_iter=randomized_search_iter) if randomized_search else ParameterGrid(param_grid=params_grid[index]):
                    model.set_params(**param_dict)
                    model.fit(X_train_inner,y_train_inner)
                    inner_pred = model.predict(X_test_inner)
                    print(param_dict)
                    internal_grid_score = metric(y_test_inner,inner_pred)
                    if best_score == None or internal_grid_score < best_score:
                        if sqrt_of_score:
                            best_score = np.sqrt(internal_grid_score)
                        else:
                            best_score = internal_grid_score
                        best_grid = param_dict
                
                inner_params.append(best_grid)
                inner_scores.append(best_score)
            
            best_inner_grid = {}
            best_inner_score = None
            
            for idx, score in enumerate(inner_scores):
                if best_inner_score == None or score < best_inner_score:
                    best_inner_score = score
                    best_inner_grid = inner_params[idx]
            
            model.set_params(**best_inner_grid)
            model.fit(X_train_outer,y_train_outer)
            pred = model.predict(X_test_outer)
            
            if sqrt_of_score:
                outer_score.append(np.sqrt(metric(y_test_outer,pred)))
            else:
                outer_score.append(metric(y_test_outer,pred))
            
            # Append variance
            variance.append(np.var(pred))
            
            print('Best parameters was: {0}'.format(best_inner_grid))
            print('Outer score: {0}'.format(outer_score[i]))
            print('Inner score: {0}'.format(best_inner_score))
        
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

X,X_test,y = data_engineering(train,test)

models_to_run = [RandomForestRegressor(), xgb.XGBRegressor()]
models_param_grid = [ # 1st param grid, corresponding to RandomForestRegressor
                    {
                            'max_depth': [3, None],
                            'n_estimators': np.random.randint(100,1000,10)
                    }, 
                    { # 2nd param grid, corresponding to XGBRegressor
                            'colsample_bytree': np.linspace(0.3, 0.5),
                            'n_estimators': np.random.randint(100,1000,10)
                    }
                    ]

multiple_nested_cv(X_df=X, y_df=y, models=models_to_run, 
                                   params_grid=models_param_grid,
                                   outer_kfolds=5, inner_kfolds=5,sqrt_of_score=True,
                                   randomized_search = True, randomized_search_iter = 50)
