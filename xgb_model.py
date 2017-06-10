# coding=utf-8
import pandas as pd
import xgboost as xgb
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier

'''
AUC Score (Train): 0.847222
'''

def drawAucLine():
    pass


# 训练模型并预测出结果-1
def train_model(train_xy, test_xy, random_seed):
    test_ID = test_xy.ID
    test_y = test_xy.Kind
    test_xy = test_xy.drop(['ID'], axis=1)
    test_xy = test_xy.drop(['Kind'], axis=1)
    dtest = xgb.DMatrix(test_xy)


    train_y = train_xy.Kind
    train_xy = train_xy.drop(['ID'], axis=1)
    train_xy = train_xy.drop(['Kind'], axis=1)
    # train, val = train_test_split(train_xy, test_size=0.2, random_state=random_seed)
    dtrain = xgb.DMatrix(train_xy, label=train_y)
    params = {
        'booster': 'gbtree',  # gbtree used
        'objective': 'binary:logistic',
        # 'early_stopping_rounds': 50,
        # 'scale_pos_weight': 0.63,  # 正样本权重
        'eval_metric': 'auc',
        # 'gamma': 0,
        # 'max_depth': 3,
        # # 'lambda': 550,
        # 'subsample': 0.8,
        # 'colsample_bytree': 0.8,
        # 'min_child_weight': 1,
        # 'eta': 0.08,
        # 'seed': random_seed,
        'nthread': 3,
        'silent': 1
    }
    # model = xgb.train(params, dtrain, num_boost_round=90)
    model = xgb.train(params, dtrain)
    print('best_ntree_limit:', model.best_ntree_limit)
    predict_y = model.predict(dtest, ntree_limit=model.best_ntree_limit) # 预测结果为概率
    print(predict_y)
    # print "Accuracy : %.4g" % metrics.accuracy_score(test_y, predict_y)#评估模型准确率的函数
    print "AUC Score (Train): %f" % metrics.roc_auc_score(test_y, predict_y)
    fea_importance = model.get_fscore()
    # fea_importance.sort()
    # print(fea_importance)
    # fea_importance.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')
    # plt.show()
    # print(fea_importance)


# 训练模型并预测出结果-2
def train_model2(train_xy, test_xy, n_estimators, learning_rate, max_depth, min_child_weight, random_seed):
    test_ID = test_xy.ID
    test_y = test_xy.Kind
    test_xy = test_xy.drop(['ID'], axis=1)
    test_xy = test_xy.drop(['Kind'], axis=1)

    train_y = train_xy.Kind
    train_xy = train_xy.drop(['ID'], axis=1)
    train_xy = train_xy.drop(['Kind'], axis=1)


    xgb_model = XGBClassifier(
            n_estimators = n_estimators,
            learning_rate = learning_rate,
            max_depth = max_depth,
            min_child_weight =min_child_weight,
            gamma = 0.1,
            subsample = 0.8,
            colsample_bytree = 0.8,
            objective = 'binary:logistic',
            nthread = 4,
            seed = random_seed)
    # 训练
    xgb_model.fit(train_xy, train_y)
    y_pred_proba = xgb_model.predict_proba(test_xy)[:, 1]
    # 输出auc值
    print "AUC Score (Train): %f" % metrics.roc_auc_score(test_y, y_pred_proba)
    # 输出混淆矩阵
    y_pred = xgb_model.predict(test_xy)
    report = classification_report(test_y, y_pred)
    print(report)


# 训练模型并预测出结果-3(default)
def train_model3(train_xy, test_xy, random_seed):
    test_ID = test_xy.ID
    test_y = test_xy.Kind
    test_xy = test_xy.drop(['ID'], axis=1)
    test_xy = test_xy.drop(['Kind'], axis=1)

    train_y = train_xy.Kind
    train_xy = train_xy.drop(['ID'], axis=1)
    train_xy = train_xy.drop(['Kind'], axis=1)


    xgb_model = XGBClassifier(
            # n_estimators = n_estimators,
            # learning_rate = learning_rate,
            # max_depth = max_depth,
            # min_child_weight =min_child_weight,
            # gamma = 0.1,
            # subsample = 0.8,
            # colsample_bytree = 0.8,
            objective = 'binary:logistic',
            nthread = 4,
            # seed = random_seed
            )
    # 训练
    xgb_model.fit(train_xy, train_y)
    y_pred_proba = xgb_model.predict_proba(test_xy)[:, 1]
    # 输出auc值
    print "AUC Score (Train): %f" % metrics.roc_auc_score(test_y, y_pred_proba)
    # 输出混淆矩阵
    y_pred = xgb_model.predict(test_xy)
    report = classification_report(test_y, y_pred)
    print(report)


# 训练模型并预测出结果-4(cholesterol、ca、fat、na、protein、calories和carbohydrate)
def train_model4(train_xy, test_xy, n_estimators, learning_rate, max_depth, min_child_weight, random_seed):
    test_ID = test_xy.ID
    test_y = test_xy.Kind
    test_xy = test_xy.drop(['ID'], axis=1)
    test_xy = test_xy.drop(['Kind'], axis=1)
    test_xy = test_xy.loc[:, ['cholesterol', 'ca', 'fat', 'na', 'protein', 'calories', 'carbohydrate']]


    train_y = train_xy.Kind
    train_xy = train_xy.drop(['ID'], axis=1)
    train_xy = train_xy.drop(['Kind'], axis=1)
    train_xy = train_xy.loc[:, ['cholesterol', 'ca', 'fat', 'na', 'protein', 'calories', 'carbohydrate']]


    xgb_model = XGBClassifier(
            n_estimators = n_estimators,
            learning_rate = learning_rate,
            max_depth = max_depth,
            min_child_weight =min_child_weight,
            gamma = 0.1,
            subsample = 0.8,
            colsample_bytree = 0.8,
            objective = 'binary:logistic',
            nthread = 4,
            seed = random_seed)
    # 训练
    xgb_model.fit(train_xy, train_y)
    y_pred_proba = xgb_model.predict_proba(test_xy)[:, 1]
    # 输出auc值
    print "AUC Score (Train): %f" % metrics.roc_auc_score(test_y, y_pred_proba)
    # 输出混淆矩阵
    y_pred = xgb_model.predict(test_xy)
    report = classification_report(test_y, y_pred)
    print(report)


if __name__ == '__main__':
    train_xy = pd.read_csv("Data/train-gao.csv")
    test_xy = pd.read_csv("Data/test-gao.csv")
    # n_estimators, learning_rate(eta), max_depth, min_child_weight
    print("---默认值-----------")
    train_model3(train_xy, test_xy, 12)
    print("---n_estimators=140, learning_rate(eta)=0.2, max_depth=6, min_child_weight=4-----")
    train_model2(train_xy, test_xy, 140, 0.2, 6, 4, 12)
    print("---n_estimators=120, learning_rate(eta)=0.17, max_depth=8, min_child_weight=3-----")
    train_model2(train_xy, test_xy, 120, 0.17, 3, 2, 12)
    print("---n_estimators=140, learning_rate(eta)=0.18, max_depth=8, min_child_weight=3-----")
    train_model2(train_xy, test_xy, 140, 0.18, 8, 3, 12)
    print("---n_estimators=90, learning_rate(eta)=0.17, max_depth=3, min_child_weight=1-----")
    train_model2(train_xy, test_xy, 90, 0.17, 3, 1, 12)
    print("---最主要7个特征----")
    train_model4(train_xy, test_xy, 90, 0.17, 3, 1, 12)
    print("---网格搜索最优参数(单个调试)----")
    train_model2(train_xy, test_xy, 40, 0.04, 4, 6, 12)
    print("---网格搜索最优参数(所有参数共同调试)----")
    train_model2(train_xy, test_xy, 20, 0.19, 3, 1, 12)
    '''
    ---默认值-----------
---默认值-----------
AUC Score (Train): 0.868519
             precision    recall  f1-score   support

          0       0.85      0.55      0.67        40
          1       0.81      0.95      0.87        81

avg / total       0.82      0.82      0.81       121

---n_estimators=140, learning_rate(eta)=0.2, max_depth=6, min_child_weight=4-----
AUC Score (Train): 0.857099
             precision    recall  f1-score   support

          0       0.70      0.53      0.60        40
          1       0.79      0.89      0.84        81

avg / total       0.76      0.77      0.76       121

---n_estimators=120, learning_rate(eta)=0.17, max_depth=8, min_child_weight=3-----
AUC Score (Train): 0.865432
             precision    recall  f1-score   support

          0       0.79      0.57      0.67        40
          1       0.82      0.93      0.87        81

avg / total       0.81      0.81      0.80       121

---n_estimators=140, learning_rate(eta)=0.18, max_depth=8, min_child_weight=3-----
AUC Score (Train): 0.879321
             precision    recall  f1-score   support

          0       0.79      0.55      0.65        40
          1       0.81      0.93      0.86        81

avg / total       0.80      0.80      0.79       121

---n_estimators=90, learning_rate(eta)=0.17, max_depth=3, min_child_weight=1-----
AUC Score (Train): 0.875309
             precision    recall  f1-score   support

          0       0.79      0.55      0.65        40
          1       0.81      0.93      0.86        81

avg / total       0.80      0.80      0.79       121

---最主要7个特征----
AUC Score (Train): 0.850309
             precision    recall  f1-score   support

          0       0.73      0.60      0.66        40
          1       0.82      0.89      0.85        81

avg / total       0.79      0.79      0.79       121

---网格搜索最优参数(单个调试)----
AUC Score (Train): 0.821605
             precision    recall  f1-score   support

          0       0.83      0.47      0.60        40
          1       0.79      0.95      0.86        81

avg / total       0.80      0.79      0.78       121

---网格搜索最优参数(所有参数共同调试)----
AUC Score (Train): 0.845988
             precision    recall  f1-score   support

          0       0.74      0.50      0.60        40
          1       0.79      0.91      0.85        81

avg / total       0.77      0.78      0.76       121

[Finished in 2.5s]
    '''

