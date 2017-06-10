# coding=utf-8
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
# from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV


'''
{'n_estimators': 40} 0.894365079365
'''
def tune_n_estimators(train, predictors, target):
    param_test = {
     'n_estimators':range(10,170,10)
    }
    gsearch = GridSearchCV(
        estimator = XGBClassifier(
            # learning_rate =0.1,
            # # n_estimators=140,
            # n_estimators=10,
            # # max_depth=5,
            # min_child_weight=1,
            # gamma=0,
            # subsample=0.8,
            # colsample_bytree=0.8,
            # objective= 'binary:logistic',
            # nthread=4,
            # scale_pos_weight=1,
            seed=27),
        param_grid = param_test,
        scoring='roc_auc',
        n_jobs=2,
        iid=False,
        cv=5)
    gsearch.fit(train[predictors],train[target])
    ##write in yuce.txt
    # f=open('score.txt','w')
    # for scores in gsearch.grid_scores_:
    #     f.write(str(scores)+"\r\n")
    # f.close()
    for scores in gsearch.grid_scores_:
        print(scores)
    print(str(gsearch.best_params_) + " " + str(gsearch.best_score_))



'''
learning_rate: [0.01, 0.02, 0.03, 0.04, 0.05, ...... 0.19, 0.20]
{'learning_rate': 0.04} 0.891031746032
'''
def tune_learning_rate(train, predictors, target):
    param_test = {
     'learning_rate':[i/100.0 for i in range(1,21)]
    }
    gsearch = GridSearchCV(
        estimator = XGBClassifier(
            # learning_rate =0.1,
            # # n_estimators=140,
            # n_estimators=10,
            # # max_depth=5,
            # min_child_weight=1,
            # gamma=0,
            # subsample=0.8,
            # colsample_bytree=0.8,
            # objective= 'binary:logistic',
            # nthread=4,
            # scale_pos_weight=1,
            seed=27),
        param_grid = param_test,
        scoring='roc_auc',
        n_jobs=2,
        iid=False,
        cv=5)
    gsearch.fit(train[predictors],train[target])
    for scores in gsearch.grid_scores_:
        print(scores)
    print(str(gsearch.best_params_) + " " + str(gsearch.best_score_))


'''
max_depth: [3, 4, 5, 6, 7, 8, 9, 10]
{'max_depth': 4} 0.881984126984
'''
def tune_max_depth(train, predictors, target):
    param_test = {
     'max_depth':[3, 4, 5, 6, 7, 8, 9, 10]
    }
    gsearch = GridSearchCV(
        estimator = XGBClassifier(
            # learning_rate =0.1,
            # # n_estimators=140,
            # n_estimators=10,
            # # max_depth=5,
            # min_child_weight=1,
            # gamma=0,
            # subsample=0.8,
            # colsample_bytree=0.8,
            # objective= 'binary:logistic',
            # nthread=4,
            # scale_pos_weight=1,
            seed=27),
        param_grid = param_test,
        scoring='roc_auc',
        n_jobs=2,
        iid=False,
        cv=5)
    gsearch.fit(train[predictors],train[target])
    for scores in gsearch.grid_scores_:
        print(scores)
    print(str(gsearch.best_params_) + " " + str(gsearch.best_score_))


'''
'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8]
{'min_child_weight': 6} 0.890396825397
'''
def tune_min_child_weight(train, predictors, target):
    param_test = {
     'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8]
    }
    gsearch = GridSearchCV(
        estimator = XGBClassifier(
            # learning_rate =0.1,
            # # n_estimators=140,
            # n_estimators=10,
            # # max_depth=5,
            # min_child_weight=1,
            # gamma=0,
            # subsample=0.8,
            # colsample_bytree=0.8,
            # objective= 'binary:logistic',
            # nthread=4,
            # scale_pos_weight=1,
            seed=27),
        param_grid = param_test,
        scoring='roc_auc',
        n_jobs=2,
        iid=False,
        cv=5)
    gsearch.fit(train[predictors],train[target])
    for scores in gsearch.grid_scores_:
        print(scores)
    print(str(gsearch.best_params_) + " " + str(gsearch.best_score_))

'''
{'n_estimators': 20, 'learning_rate': 0.19, 'max_depth': 3, 'min_child_weight': 1} 0.902619047619
'''
def tune_parameter(train, predictors, target):
    param_test = {
     'learning_rate':[i/100.0 for i in range(1,21)],
     'max_depth':[3, 4, 5, 6, 7, 8, 9, 10],
     'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8],
     'n_estimators':range(10,170,10)
    }
    gsearch = GridSearchCV(
        estimator = XGBClassifier(
            # learning_rate =0.1,
            # # n_estimators=140,
            # n_estimators=10,
            # # max_depth=5,
            # min_child_weight=1,
            # gamma=0,
            # subsample=0.8,
            # colsample_bytree=0.8,
            # objective= 'binary:logistic',
            # nthread=4,
            # scale_pos_weight=1,
            seed=27),
        param_grid = param_test,
        scoring='roc_auc',
        n_jobs=2,
        iid=False,
        cv=5)
    gsearch.fit(train[predictors],train[target])
    for scores in gsearch.grid_scores_:
        print(scores)
    print(str(gsearch.best_params_) + " " + str(gsearch.best_score_))


if __name__ == '__main__':
    train = pd.read_csv('../../Data/train-gao.csv')
    target = 'Kind'
    IDcol = 'ID'
    predictors = [x for x in train.columns if x not in [target,IDcol]]

    print('-------------------单个参数调试------------------------')
    tune_n_estimators(train, predictors, target)
    tune_learning_rate(train, predictors, target)
    tune_max_depth(train, predictors, target)
    tune_min_child_weight(train, predictors, target)
    print('-------------------所有参数一起调试------------------------')
    tune_parameter(train, predictors, target)