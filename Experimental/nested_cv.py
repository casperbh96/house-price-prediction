import pandas as pd
from fancyimpute import IterativeImputer
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error
import xgboost as xgb



def nested_cv(features_data, expected_data, model, model_params, outer_kfolds, inner_kfolds, outer_metric = mean_squared_error, inner_metric = 'neg_mean_squared_error',sqrt_of_score = False):
    '''A general method to handle nested validation for RandomForest and XGBRegressor.

    Parameters
    ----------
    features_data (DataFrame)
         Feature data which represent X.
    expected_data (Series)
         Y data.
    model (Model)
          the model/algorithm that you will use, for now only XGBRegressor and RandomForest are supported.
    model_params (dict)
         contains the required parameters to run the model.
    outer_kfolds (int)
         the number of outer KFolds
    inner_kfolds (int)
         the number of inner KFolds
    outer_metric (def)
         scoring metrix
    inner_metric (str)
         inner evaluation metrix
    sqrt_of_score (bool)
         default it False.

    Returns
    -------
    inner_params
         Best Inner Params.
    outer_score
         Outer Score List.
    inner_score
         Inner scorer List.
    '''
    outer_cv = KFold(n_splits=outer_kfolds,shuffle=True)
    inner_cv = KFold(n_splits=inner_kfolds,shuffle=True)
    
    outer_score = []
    inner_score = []
    inner_params = {}
    variance = []

    def score_to_inner_params(best_score_params):
        print(best_score_params)
        for key,value in best_score_params.items():
            print(key)
            if key in inner_params :
                if value not in inner_params[key]:
                    inner_params[key].append(value)
            else:
                inner_params[key] = [value]

    for (i, (train_index,test_index)) in enumerate(outer_cv.split(features_data,expected_data)):
            print('\n{0}/{1} <-- Current outer fold'.format(i+1,outer_kfolds))
            X_train_outer, X_test_outer = features_data.iloc[train_index], features_data.iloc[test_index]
            y_train_outer, y_test_outer = expected_data.iloc[train_index], expected_data.iloc[test_index]
                        
            GSCV = RandomizedSearchCV(estimator=model,param_distributions=model_params,
                                scoring=inner_metric,cv=inner_cv,n_jobs=-1)
            
            # Inner loop: Find best parameters from dict of hyperparameters
            GSCV.fit(X_train_outer,y_train_outer)
            
            # Outer loop: Test best parameters
            pred = GSCV.predict(X_test_outer)
            
            # Append variance
            variance.append(np.var(pred))
            
            # Append best dict of parameters from hyperparameters
            score_to_inner_params(GSCV.best_params_)
            
            if sqrt_of_score:
                inner_score.append(np.sqrt(-GSCV.best_score_))
                outer_score.append(np.sqrt(outer_metric(y_test_outer,pred)))
            else:
                inner_score.append(-GSCV.best_score_)
                outer_score.append(outer_metric(y_test_outer,pred))
            
            print('Best parameters was: {0}'.format(GSCV.best_params_))
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
    return inner_params,outer_score,inner_score
