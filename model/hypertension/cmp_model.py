# coding=utf-8
import pandas as pd
import xgboost as xgb
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier


def drawAucLine():
    pass


'''
AUC Score (Train): 0.849074
             precision    recall  f1-score   support

          0       0.86      0.30      0.44        40
          1       0.74      0.98      0.84        81

avg / total       0.78      0.75      0.71       121
'''
def randomForest(train_xy, test_xy):
    test_ID = test_xy.ID
    test_y = test_xy.Kind
    test_xy = test_xy.drop(['ID'], axis=1)
    test_xy = test_xy.drop(['Kind'], axis=1)

    train_y = train_xy.Kind
    train_xy = train_xy.drop(['ID'], axis=1)
    train_xy = train_xy.drop(['Kind'], axis=1)


    rf_model = RandomForestClassifier(
            n_estimators = 90,
            max_depth = 3)
    # 训练
    rf_model.fit(train_xy, train_y)
    y_pred_proba = rf_model.predict_proba(test_xy)[:, 1]
    # 输出auc值
    print "AUC Score (Train): %f" % metrics.roc_auc_score(test_y, y_pred_proba)
    # 输出混淆矩阵
    y_pred = rf_model.predict(test_xy)
    report = classification_report(test_y, y_pred)
    print(report)



def gbdt(train_xy, test_xy):
    test_ID = test_xy.ID
    test_y = test_xy.Kind
    test_xy = test_xy.drop(['ID'], axis=1)
    test_xy = test_xy.drop(['Kind'], axis=1)

    train_y = train_xy.Kind
    train_xy = train_xy.drop(['ID'], axis=1)
    train_xy = train_xy.drop(['Kind'], axis=1)


    gbdt_model = GradientBoostingClassifier(
            n_estimators = 90,
            max_depth = 3,
            learning_rate = 0.17)
    # 训练
    gbdt_model.fit(train_xy, train_y)
    y_pred_proba = gbdt_model.predict_proba(test_xy)[:, 1]
    # 输出auc值
    print "AUC Score (Train): %f" % metrics.roc_auc_score(test_y, y_pred_proba)
    # 输出混淆矩阵
    y_pred = gbdt_model.predict(test_xy)
    report = classification_report(test_y, y_pred)
    print(report)


def xgboost(train_xy, test_xy):
    test_ID = test_xy.ID
    test_y = test_xy.Kind
    test_xy = test_xy.drop(['ID'], axis=1)
    test_xy = test_xy.drop(['Kind'], axis=1)

    train_y = train_xy.Kind
    train_xy = train_xy.drop(['ID'], axis=1)
    train_xy = train_xy.drop(['Kind'], axis=1)


    xgb_model = XGBClassifier(
            n_estimators = 90,
            learning_rate = 0.17,
            max_depth = 3,
            min_child_weight = 1,
            gamma = 0.1,
            subsample = 0.8,
            colsample_bytree = 0.8,
            objective = 'binary:logistic',
            nthread = 4,
            seed = 12)
    # 训练
    xgb_model.fit(train_xy, train_y)
    y_pred_proba = xgb_model.predict_proba(test_xy)[:, 1]
    # 输出auc值
    print "AUC Score (Train): %f" % metrics.roc_auc_score(test_y, y_pred_proba)
    # 输出混淆矩阵
    y_pred = xgb_model.predict(test_xy)
    report = classification_report(test_y, y_pred)
    print(report)


def bagging(train_xy, test_xy):
    test_ID = test_xy.ID
    test_y = test_xy.Kind
    test_xy = test_xy.drop(['ID'], axis=1)
    test_xy = test_xy.drop(['Kind'], axis=1)

    train_y = train_xy.Kind
    train_xy = train_xy.drop(['ID'], axis=1)
    train_xy = train_xy.drop(['Kind'], axis=1)


    bag_model = BaggingClassifier(
            n_estimators = 90)
    # 训练
    bag_model.fit(train_xy, train_y)
    y_pred_proba = bag_model.predict_proba(test_xy)[:, 1]
    # 输出auc值
    print "AUC Score (Train): %f" % metrics.roc_auc_score(test_y, y_pred_proba)
    # 输出混淆矩阵
    y_pred = bag_model.predict(test_xy)
    report = classification_report(test_y, y_pred)
    print(report)


def booster(train_xy, test_xy):
    test_ID = test_xy.ID
    test_y = test_xy.Kind
    test_xy = test_xy.drop(['ID'], axis=1)
    test_xy = test_xy.drop(['Kind'], axis=1)

    train_y = train_xy.Kind
    train_xy = train_xy.drop(['ID'], axis=1)
    train_xy = train_xy.drop(['Kind'], axis=1)


    booster_model = AdaBoostClassifier(
            n_estimators = 90)
    # 训练
    booster_model.fit(train_xy, train_y)
    y_pred_proba = booster_model.predict_proba(test_xy)[:, 1]
    # 输出auc值
    print "AUC Score (Train): %f" % metrics.roc_auc_score(test_y, y_pred_proba)
    # 输出混淆矩阵
    y_pred = booster_model.predict(test_xy)
    report = classification_report(test_y, y_pred)
    print(report)


if __name__ == '__main__':
    train_xy = pd.read_csv("Data/train-gao.csv")
    test_xy = pd.read_csv("Data/test-gao.csv")
    print("---------------randomForest--------------")
    randomForest(train_xy, test_xy)
    print("---------------gbdt--------------")
    gbdt(train_xy, test_xy)
    print("---------------xgboost--------------")
    xgboost(train_xy, test_xy)
    print("---------------bagging--------------")
    bagging(train_xy, test_xy)
    print("---------------booster--------------")
    booster(train_xy, test_xy)

'''
---------------randomForest--------------
AUC Score (Train): 0.858333
             precision    recall  f1-score   support

          0       0.92      0.30      0.45        40
          1       0.74      0.99      0.85        81

avg / total       0.80      0.76      0.72       121

---------------gbdt--------------
AUC Score (Train): 0.868210
             precision    recall  f1-score   support

          0       0.85      0.57      0.69        40
          1       0.82      0.95      0.88        81

avg / total       0.83      0.83      0.82       121

---------------xgboost--------------
AUC Score (Train): 0.875309
             precision    recall  f1-score   support

          0       0.79      0.55      0.65        40
          1       0.81      0.93      0.86        81

avg / total       0.80      0.80      0.79       121

---------------bagging--------------
AUC Score (Train): 0.831019
             precision    recall  f1-score   support

          0       0.82      0.57      0.68        40
          1       0.82      0.94      0.87        81

avg / total       0.82      0.82      0.81       121

---------------booster--------------
AUC Score (Train): 0.838272
             precision    recall  f1-score   support

          0       0.77      0.60      0.68        40
          1       0.82      0.91      0.87        81

avg / total       0.81      0.81      0.80       121

[Finished in 3.5s]
'''
    
