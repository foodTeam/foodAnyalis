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
    test_ID = list(test_ID)
    test_y = list(test_y)
    # print(test_ID)
    for i in range(len(y_pred_proba)):
        print(str(test_ID[i]) + "-----" + str(test_y[i]) + "------" + str(y_pred_proba[i]))
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
    # print("---默认值-----------")
    # train_model3(train_xy, test_xy, 12)
    # print("---n_estimators=140, learning_rate(eta)=0.2, max_depth=6, min_child_weight=4-----")
    # train_model2(train_xy, test_xy, 140, 0.2, 6, 4, 12)
    # print("---n_estimators=120, learning_rate(eta)=0.17, max_depth=8, min_child_weight=3-----")
    # train_model2(train_xy, test_xy, 120, 0.17, 3, 2, 12)
    # print("---n_estimators=140, learning_rate(eta)=0.18, max_depth=8, min_child_weight=3-----")
    # train_model2(train_xy, test_xy, 140, 0.18, 8, 3, 12)
    print("---n_estimators=90, learning_rate(eta)=0.17, max_depth=3, min_child_weight=1-----")
    train_model2(train_xy, test_xy, 90, 0.17, 3, 1, 12)
    # print("---最主要7个特征----")
    # train_model4(train_xy, test_xy, 90, 0.17, 3, 1, 12)
    # print("---网格搜索最优参数(单个调试)----")
    # train_model2(train_xy, test_xy, 40, 0.04, 4, 6, 12)
    # print("---网格搜索最优参数(所有参数共同调试)----")
    # train_model2(train_xy, test_xy, 20, 0.19, 3, 1, 12)
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


预测结果：
98-----1------0.998003
689-----1------0.998325
2503-----1------0.997975
2504-----1------0.998531
2220-----1------0.668157
429-----1------0.998825
529-----0------0.816465
828-----0------0.983464
891-----0------0.985945
1008-----1------0.986193
1833-----0------0.924614
1939-----0------0.232714
1948-----0------0.0291316
2140-----0------0.976882
2862-----0------0.976565
2908-----1------0.968484
413-----0------0.958608
1073-----0------0.687499
1562-----0------0.530128
2745-----1------0.449399
3043-----1------0.963118
719-----0------0.879014
755-----0------0.822211
1644-----0------0.949377
466-----0------0.0960893
510-----0------0.166277
2217-----0------0.00808991
2228-----0------0.0933826
2260-----0------0.122411
2881-----1------0.980197
2118-----0------0.160222
2222-----0------0.926534
2223-----0------0.0235415
2265-----0------0.498323
2280-----0------0.510181
1922-----0------0.290511
1996-----0------0.0493159
2598-----0------0.993083
3370-----0------0.949139
255-----0------0.767919
1416-----0------0.804925
2531-----1------0.99879
41-----1------0.99937
42-----1------0.992624
430-----1------0.99132
3051-----1------0.983077
3052-----1------0.998723
3053-----1------0.991694
3054-----1------0.176636
3055-----1------0.547616
3056-----1------0.996518
3057-----1------0.995622
3058-----1------0.912091
3059-----1------0.97265
3060-----1------0.182681
3061-----1------0.664809
3062-----1------0.711885
3063-----1------0.996977
3064-----1------0.99654
3065-----1------0.874164
3066-----1------0.997377
3067-----1------0.999151
3068-----1------0.207194
3069-----1------0.998014
3070-----1------0.997545
3071-----1------0.918445
3072-----1------0.99937
3073-----1------0.994092
3074-----1------0.999732
3075-----1------0.940041
3076-----1------0.982219
3077-----1------0.968226
3078-----1------0.984111
3079-----1------0.935686
3080-----1------0.80088
3081-----1------0.272906
3082-----1------0.99233
3083-----1------0.993196
3084-----1------0.982427
3085-----1------0.994278
3086-----1------0.985838
3087-----1------0.977778
3088-----1------0.865399
3089-----1------0.887372
3090-----1------0.99745
3091-----1------0.739179
3092-----1------0.988629
3093-----1------0.994974
3094-----1------0.759544
3095-----1------0.762222
3096-----1------0.988391
3097-----1------0.997669
3098-----1------0.995348
3099-----1------0.954998
3100-----1------0.999373
3101-----1------0.964549
3102-----1------0.971174
3103-----1------0.995104
3104-----1------0.998767
3105-----1------0.993668
3106-----1------0.998774
3107-----1------0.982415
3108-----1------0.999433
3109-----1------0.991537
3110-----1------0.990368
3111-----1------0.986445
3112-----1------0.997518
3113-----1------0.828171
3114-----1------0.370767
3115-----1------0.825704
3116-----1------0.996371
2086-----0------0.0185344
2098-----0------0.0143269
2101-----0------0.0221694
2102-----0------0.00840599
2106-----0------0.00619877
2105-----0------0.0157911
2112-----0------0.00267297
2134-----0------0.00292362
2149-----0------0.0247183
2206-----0------0.248561
    '''

