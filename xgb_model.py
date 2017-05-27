# coding=utf-8
import pandas as pd
import xgboost as xgb
from sklearn import metrics
import matplotlib.pylab as plt

'''
AUC Score (Train): 0.847222
'''

def drawAucLine():
    pass


# 训练模型并预测出结果
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
        'early_stopping_rounds': 50,
        # 'scale_pos_weight': 0.63,  # 正样本权重
        'eval_metric': 'auc',
        'gamma': 0,
        'max_depth': 3,
        # 'lambda': 550,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'eta': 0.17,
        'seed': random_seed,
        'nthread': 3,
        'silent': 1
    }
    model = xgb.train(params, dtrain, num_boost_round=90)
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


if __name__ == '__main__':
    train_xy = pd.read_csv("Data/train-gao.csv")
    test_xy = pd.read_csv("Data/test-gao.csv")
    train_model(train_xy, test_xy, 12)