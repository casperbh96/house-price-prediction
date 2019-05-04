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
from grid_search import NestedGridSearchCV
from mpi4py import MPI

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

def nested_grid_search(X,y,estimator, param_grid, scoring=None, outer_cv=None, inner_cv=None):
    for i, (train_index_outer, test_index_outer) in enumerate(outer_cv.split(X,y)):
        X_train, X_test = X[train_index_outer, :], X[test_index_outer, :]
        y_train, y_test = y[train_index_outer], y[test_index_outer]
        
        #For integer/None inputs, if classifier is True and y is either
        #binary or multiclass, StratifiedKFold is used.
        #In all other cases, KFold is used.
        cv_strategy = check_cv(inner_cv, y_train, classifier=is_classifier(estimator))
        
        for j, (train_index_inner, test_index_inner) in enumerate(cv_strategy.split(X_train,y_train)):
            X_train_val, X_test_val = X_train[train_index_inner, :], X_train[test_index_inner, :]
            y_train_val, y_test_val = y_train[train_index_inner], y_train[test_index_inner]
            
            #hhttps://github.com/scikit-learn/scikit-learn/blob/0.21.X/sklearn/model_selection/_validation.py#L393
            testing = _fit_and_score(estimator=clone(estimator),
                                 X=X_train_val, y=y_train_val,
                                 scorer=scoring, train=train_index_inner,
                                 test=test_index_inner, fit_parameters=param_grid)
        

X,X_test,y = data_engineering(train,test)

estimator = RandomForestRegressor()

param_grid = {'max_depth': [3, None],
              'n_estimators': (10, 20)#, 30, 50, 100, 200, 400, 600, 800, 1000)
}

nested_cv = NestedGridSearchCV(estimator, param_grid, 'neg_mean_squared_error',
                               cv=3, inner_cv=3)
nested_cv.fit(X, y)

if MPI.COMM_WORLD.Get_rank() == 0:
    for i, scores in enumerate(nested_cv.grid_scores_):
        scores.to_csv('grid-scores-%d.csv' % (i + 1), index=False)

    print(nested_cv.best_params_)

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