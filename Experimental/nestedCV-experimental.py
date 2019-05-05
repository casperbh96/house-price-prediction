# -*- coding: utf-8 -*-
"""
Created on Sat May  4 01:50:41 2019

@author: caspe
"""
import operator
from sklearn import model_selection
from sklearn import ensemble
from sklearn.datasets import load_boston
import pandas as pd
from fancyimpute import IterativeImputer
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np
from sklearn.metrics import mean_squared_error,mean_squared_log_error
from grid_search import NestedGridSearchCV
from mpi4py import MPI
from sklearn.model_selection import check_cv
from sklearn.base import is_classifier
from sklearn.model_selection._validation import _fit_and_score

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
    
    #train["SalePrice"] = np.log1p(train["SalePrice"])
    y = train['SalePrice']
    
    cc_data = pd.get_dummies(cc_data, prefix_sep='_')
    
    cc_data = fill_ii(cc_data)
    
    X_train = cc_data[:train.shape[0]]
    X_test = cc_data[train.shape[0]:]
    
    return X_train,X_test,y

def nested_grid_search(X,y,estimator, param_grid, scoring=mean_squared_error, outer_cv=None, inner_cv=None):
    for i, (train_index_outer, test_index_outer) in enumerate(outer_cv.split(X,y)):
        X_train, X_test = X[:train_index_outer.shape[0]], X[:test_index_outer.shape[0]]
        y_train, y_test = y[:train_index_outer.shape[0]], y[:test_index_outer.shape[0]]
        outer_cv.random_state=i+1
        
        inner_loop_scores = []
        inner_loop_params = []
        print(X_train.shape[0],y_train.shape[0],X_test.shape[0],y_test.shape[0])
        for j, (train_index_inner, test_index_inner) in enumerate(cv_strategy.split(X_train,y_train)):
            X_train_val, X_test_val = X_train[:train_index_inner.shape[0]], X_train[:test_index_inner.shape[0]]
            y_train_val, y_test_val = y_train[:train_index_inner.shape[0]], y_train[:test_index_inner.shape[0]]
            inner_cv.random_state=j+1
            
            print(X_train_val.shape[0],y_train_val.shape[0],X_test_val.shape[0],y_test_val.shape[0])
            model = estimator
            model.fit(X_train_val,y_train_val)
            pred = model.predict(X_test_val)
            
            inner_loop_scores.append(scoring(y_test_val,pred))
            inner_loop_params.append(model.get_params())
            
        print(inner_loop_scores)
        #print(inner_loop_params)
        

X,X_test,y = data_engineering(train,test)

rf = RandomForestRegressor(n_estimators=10)

#For integer/None inputs, if classifier is True and y is either binary or multiclass,
#StratifiedKFold is used. In all other cases, KFold is used.
cv_strategy = check_cv(5, y, classifier=is_classifier(rf))

outer_kf = KFold(n_splits=5,shuffle=True)
inner_kf = KFold(n_splits=5,shuffle=True)

outer_loop_accuracy_scores = []
inner_loop_won_params = []
inner_loop_accuracy_scores = []

model = RandomForestRegressor()
params = {'max_depth': [3, None],
          'n_estimators': (10, 20)#, 30, 50, 100, 200, 400, 600, 800, 1000)
          }
features=X
target=y

# Looping through the outer loop, feeding each training set into a GSCV as the inner loop
for train_index,test_index in outer_kf.split(features,target):
    GSCV = GridSearchCV(estimator=model,param_grid=params,cv=inner_kf)
    
    # GSCV is looping through the training data to find the best parameters. This is the inner loop
    GSCV.fit(features[:train_index.shape[0]],target[:train_index.shape[0]])
    
    # The best hyper parameters from GSCV is now being tested on the unseen outer loop test data.
    pred = GSCV.predict(features[:test_index.shape[0]])
    
    # Appending the "winning" hyper parameters and their associated accuracy score
    inner_loop_won_params.append(GSCV.best_params_)
    outer_loop_accuracy_scores.append(mean_squared_error(target[:test_index.shape[0]],pred))
    inner_loop_accuracy_scores.append(GSCV.best_score_)

for i in zip(inner_loop_won_params,outer_loop_accuracy_scores,inner_loop_accuracy_scores):
    print(i)

print('Mean of outer loop accuracy score:',np.mean(outer_loop_accuracy_scores))

#nested_grid_search(X=X,y=y,estimator=rf,param_grid=None,scoring=mean_squared_log_error,outer_cv=cv_strategy,inner_cv=cv_strategy)


'''
NUM_TRIALS = 30

p_grid = {'max_depth': [3, None],
          'n_estimators': (10, 20, 30, 50, 100, 200, 400, 600, 800, 1000)
          }

rf = RandomForestRegressor(n_jobs=-1, random_state=1)

non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)

for i in range(NUM_TRIALS):
    print('NumTrial {0}/{1}'.format(i+1,NUM_TRIALS))
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=rf, param_grid=p_grid, cv=inner_cv)
    clf.fit(X, y)

    # Nested CV with parameter optimization
    nested_score = cross_val_score(clf, X=X, y=y, scoring='neg_mean_squared_error', cv=outer_cv)
    nested_scores[i] = np.sqrt((-nested_score).mean())


# Plot scores on each trial for nested and non-nested CV
plt.figure()
plt.subplot(211)
nested_line, = plt.plot(nested_scores, color='b')
plt.ylabel("Score", fontsize="14")
plt.xlabel("Model", fontsize="14")
plt.legend([nested_line],
           ["Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
plt.title("Generalization Error with Random Forest",
          x=1, y=1.1, fontsize="15")
plt.show()


outer_scores = []

# outer cross-validation
outer = model_selection.KFold(n_splits=3, shuffle=True)
for outer_fold, (train_index_outer, test_index_outer) in enumerate(outer.split(X,y)):
    X_train_outer, X_test_outer = X[:train_index_outer.shape[0]], X[:test_index_outer.shape[0]]
    y_train_outer, y_test_outer = y[:train_index_outer.shape[0]], y[:test_index_outer.shape[0]]

    estimators = (10,20,30,200)
    features = [3,5,7]
    for estimator in estimators:
        for feature in features:
            scores = []
            for outer_fold, (train_index_inner, test_index_inner) in enumerate(outer.split(X_train_outer,y_train_outer)):
                X_train_inner, X_test_inner = X_train_outer[:train_index_inner.shape[0]], X_train_outer[:test_index_inner.shape[0]]
                y_train_inner, y_test_inner = y_train_outer[:train_index_inner.shape[0]], y_train_outer[:test_index_inner.shape[0]]
                
                clf = ensemble.RandomForestRegressor(n_estimators=estimator, max_features=feature, n_jobs=-1, random_state=1)
                clf.fit(X_train_inner, y_train_inner)
                scores.append(clf.score(X_test_inner, y_test_inner))
    
    # calculate mean score for folds
    outer_scores.append(np.mean(scores))
    print(outer_scores)
    # get maximum score index
    index, value = min(enumerate(outer_scores), key=operator.itemgetter(1))
    
    print('Best parameter 1: {0}'.format(estimators[index]))
    print('Best parameter 2: {0}'.format(features[index]))
'''