# coding=utf-8
import pandas as pd
import xgboost as xgb
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

'''
AUC Score (Train): 0.847222
'''

def drawAucLine():
    pass


# 训练模型并预测出结果
def train_model(train_xy, test_xy, random_seed):
    test_ID = test_xy.ID 
    test_y = test_xy.Kind
    test_xy = test_xy.drop(['ID'], axis=1)     # 去掉ID
    test_xy = test_xy.drop(['Kind'], axis=1)   # 去掉类标
    dtest = xgb.DMatrix(test_xy)


    train_y = train_xy.Kind
    train_xy = train_xy.drop(['ID'], axis=1)
    # train_xy = train_xy.drop(['Kind'], axis=1)
    # val用于验证，作为一个训练过程监视
    train, val = train_test_split(train_xy, test_size=0.2, random_state=random_seed)
    y = train.Kind
    X = train.drop(['Kind'], axis=1)
    val_y = val.Kind
    val_x = val.drop(['Kind'], axis=1)
    
    params = {
        'booster': 'gbtree',  # gbtree used
        'objective': 'binary:logistic',
        'early_stopping_rounds': 50,
        'scale_pos_weight': 0.63,  # 正样本权重
        'eval_metric': 'auc',
        'gamma': 0,
        'max_depth': 5,
        # 'lambda': 550,
        'subsample': 0.6,
        'colsample_bytree': 0.9,
        'min_child_weight': 1,
        'eta': 0.02,
        'seed': random_seed,
        'nthread': 3,
        'silent': 0
    }
    dtrain = xgb.DMatrix(X, label=y)
    dval = xgb.DMatrix(val_x, label=val_y)
    watchlist = [(dval, 'val'), (dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=50, evals=watchlist)
    
    # 对测试集进行预测（以上部分之所以划分成验证集，可以用来调参）
    predict_y = model.predict(dtest, ntree_limit=model.best_ntree_limit) # 预测结果为概率
    # print(predict_y)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(test_y, predict_y)

    # 输出特征重要性
    fea_importance = model.get_fscore()
    print(fea_importance)


if __name__ == '__main__':
    train_xy = pd.read_csv("Data/train-gao.csv")
    test_xy = pd.read_csv("Data/test-gao.csv")
    train_model(train_xy, test_xy, 12)