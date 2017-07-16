# coding=utf-8
import pandas as pd
import xgboost as xgb
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier
from sklearn.utils import shuffle

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
    test_ID = list(test_ID)
    test_y = list(test_y)
    # print(test_ID)
    for i in range(len(y_pred_proba)):
        print(str(test_ID[i]) + "-----" + str(test_y[i]) + "------" + str(y_pred_proba[i]))
    print()
    # print(type(train_xy))
    print(train_xy.columns)
    print('feature importance:')
    print(xgb_model.feature_importances_)
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
    xy = pd.read_csv("Data/diabetes-all.csv")
    # 取出为0、1的数据
    xy_1 = xy[xy.Kind==1]
    xy_0 = xy[xy.Kind==0]
    # 各取70%作为训练
    len_1 = len(xy_1.index)
    len_divid_1 = int(len_1 * 0.7)
    len_0 = len(xy_0.index)
    len_divid_0 = int(len_0 * 0.7)
    

    train_0 = xy_0[0:len_divid_0] # 取 70%
    test_xy_0 = xy_0[len_divid_0:] # 取 30%

    train_1 = xy_1[0:len_divid_1] # 取 70%
    test_xy_1 = xy_1[len_divid_1:] # 取 30%

    train_xy = train_0.append(train_1)
    test_xy = test_xy_0.append(test_xy_1)
    train_xy = shuffle(train_xy)
    test_xy = shuffle(test_xy)

    # test_xy = pd.read_csv("Data/test-gao.csv")
    # n_estimators, learning_rate(eta), max_depth, min_child_weight
    print("---n_estimators=130, learning_rate(eta)=0.194, max_depth=5, min_child_weight=4-----")
    train_model2(train_xy, test_xy, 130, 0.194, 5, 4, 12)
    '''
    ---n_estimators=130, learning_rate(eta)=0.194, max_depth=5, min_child_weight=4-----
Index([u'calories', u'carbohydrate', u'fat', u'protein', u'vitamine', u'vta',
       u'vtc', u'vte', u'carotene', u'thiamine', u'riboflavin', u'yansuan',
       u'cholesterol', u'mg', u'ca', u'iron', u'zinc', u'copper', u'mn', u'k',
       u'p', u'na', u'se'],
      dtype='object')
feature importance:
[ 0.06578948  0.09210526  0.06140351  0.01754386  0.04385965  0.03947368
  0.05701754  0.03070175  0.02631579  0.01754386  0.01315789  0.03947368
  0.1008772   0.02631579  0.02192982  0.04824561  0.01754386  0.03947368
  0.02631579  0.01754386  0.03070175  0.12719299  0.03947368]
AUC Score (Train): 0.976431
             precision    recall  f1-score   support

          0       0.90      0.75      0.82        36
          1       0.93      0.98      0.96       132

avg / total       0.93      0.93      0.93       168

[Finished in 2.1s]
    '''

