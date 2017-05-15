import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

#1
def modelfit(train, predictors):
    alg = XGBClassifier(
         learning_rate =0.1,
         # n_estimators=1000,
         n_estimators=10,
         max_depth=5,
         min_child_weight=1,
         gamma=0,
         subsample=0.8,
         colsample_bytree=0.8,
         objective= 'binary:logistic',
         nthread=2,
         scale_pos_weight=1,
         seed=27)
    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(train[predictors].values, label=train[target].values)
    cvresult = xgb.cv(xgb_param,
        xgtrain,
        num_boost_round=alg.get_params()['n_estimators'],
        nfold=5,
        metrics='auc',
        early_stopping_rounds=50)
    alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(train[predictors], train['Disbursed'],eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predict(train[predictors])
    dtrain_predprob = alg.predict_proba(train[predictors])[:,1]

    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(train['Disbursed'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(train['Disbursed'], dtrain_predprob)
    print('----------------this is pic start: -------------------------------')
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    # plt.show()
#2
def tuneMaxDepthAndWeight(train, predictors, target):
    param_test = {
     'max_depth':range(3,10,2),
     'min_child_weight':range(1,3,1)
    }
    gsearch = GridSearchCV(
        estimator = XGBClassifier(
            learning_rate =0.1,
            # n_estimators=140,
            n_estimators=10,
            # max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective= 'binary:logistic',
            nthread=4,
            scale_pos_weight=1,
            seed=27),
        param_grid = param_test,
        scoring='roc_auc',
        n_jobs=2,
        iid=False,
        cv=3)
    gsearch.fit(train[predictors],train[target])
    print('----------------this is line-------------------------------')
    for scores in gsearch.grid_scores_:
        print(scores)
    print(gsearch.best_params_)
    print(gsearch.best_score_)
#3
def tuneGamma(train, predictors, target):
    param_test3 = {
     'gamma':[i/10.0 for i in range(0,5)]
     }
    gsearch3 = GridSearchCV(
            estimator = XGBClassifier( 
                    learning_rate =0.1, 
                    n_estimators=10, 
                    max_depth=4, 
                    min_child_weight=6, 
                    gamma=0, 
                    subsample=0.8, 
                    colsample_bytree=0.8, 
                    objective= 'binary:logistic', 
                    nthread=4, 
                    scale_pos_weight=1,
                    seed=27), 
                param_grid = param_test3, 
                scoring='roc_auc',
                n_jobs=4,
                iid=False, 
                cv=5)
    gsearch3.fit(train[predictors],train[target])
    print('----------------this is line-------------------------------')
    for scores in gsearch3.grid_scores_:
        print(scores)
    print(gsearch3.best_params_)
    print(gsearch3.best_score_)
    
#4    tun subsample & colsample_bytree 
def tuneSubsampleColsample(train, predictors, target):    
    param_test4 = {
        'subsample':[i/10.0 for i in range(6,8)],
        'colsample_bytree':[i/10.0 for i in range(6,8)]
        }
    gsearch4 = GridSearchCV(
        estimator = XGBClassifier(
            learning_rate =0.1, 
            n_estimators=10, 
            max_depth=3, 
            min_child_weight=4, 
            gamma=0.1, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            objective= 'binary:logistic', 
            nthread=4, 
            scale_pos_weight=1,
            seed=27), 
        param_grid = param_test4, 
        scoring='roc_auc',
        n_jobs=4,
        iid=False, 
        cv=5)
    gsearch4.fit(train[predictors],train[target])
    print('----------------this is line-------------------------------')
    for scores in gsearch4.grid_scores_:
        print(scores)
    print(gsearch4.best_params_)
    print(gsearch4.best_score_)
#5    
def tuneAlphaLambda(train, predictors, target):    
    param_test5 = {
        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
        }
    gsearch5 = GridSearchCV(
        estimator = XGBClassifier( 
            learning_rate =0.1, 
            n_estimators=10, 
            max_depth=4, 
            min_child_weight=6, 
            gamma=0.1, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            objective= 'binary:logistic', 
            nthread=4, 
            scale_pos_weight=1,
            seed=27), 
        param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch5.fit(train[predictors],train[target])
    print('----------------this is line-------------------------------')
    for scores in gsearch5.grid_scores_:
        print(scores)
    print(gsearch5.best_params_)
    print(gsearch5.best_score_)
#6    
def tuneLearning_rate(train, predictors, target):
    param_test6 = {
     'learning_rate':[ 0.3,0.35]
    }
    gsearch6 = GridSearchCV(
        estimator = XGBClassifier( 
            learning_rate =0.1, 
            n_estimators=10, 
            max_depth=4, 
            min_child_weight=6, 
            gamma=0.1, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            objective= 'binary:logistic', 
            nthread=4, 
            scale_pos_weight=1,
            seed=27), 
        param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch6.fit(train[predictors],train[target])
    print('----------------this is line-------------------------------')
    for scores in gsearch6.grid_scores_:
        print(scores)
    print(gsearch6.best_params_)
    print(gsearch6.best_score_)

if __name__ == '__main__':
    rcParams['figure.figsize'] = 12, 4
    train = pd.read_csv('train_modified.csv')
    target = 'Disbursed'
    IDcol = 'ID'
    predictors = [x for x in train.columns if x not in [target,IDcol]]
    
    #1 find tree number
#    modelfit(train, predictors)
    #2 tune maxdepth and wight
#    tuneMaxDepthAndWeight(train, predictors, target)
    #3 tune gamma
#    tune Gamma(train, predictors, target)
    #4 tun subsample & colsample_bytree
#    tuneSubsampleColsample(train, predictors, target)
    #5 tune AlphaLambda
#    tuneAlphaLambda(train, predictors, target)
    #6 tun learning
    tuneLearning_rate(train, predictors, target)