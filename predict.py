#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.externals import joblib

import matplotlib.pylab as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4#配置画图参数

#train = pd.read_csv('train_modified.csv')
test = pd.read_csv('Data/test-gao.csv')
test = test.drop(['Kind', 'ID'], axis=1)
target = 'Kind'
IDcol = 'ID'
modle = joblib.load('xgboost.model')
Gao_predict = modle.predict(test)
print(Gao_predict)
# you can wirte the result to file
print('done!')
