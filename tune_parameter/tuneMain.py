#coding=utf-8
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import random
import math
from sklearn import metrics

generations = 2   # 繁殖代数 100
pop_size = 5      # 种群数量  500
max_value = 10      # 基因中允许出现的最大值  
chrom_length = 8       # 染色体长度  
pc = 0.6            # 交配概率  
pm = 0.01           # 变异概率  
results = [[]]      # 存储每一代的最优解，N个二元组  
fit_value = []      # 个体适应度  
fit_mean = []       # 平均适应度 
pop = [[0, 1, 0, 1, 0, 1, 0, 1] for i in range(pop_size)] # 初始化种群中所有个体的基因初始序列


'''
n_estimators 取 {10、20、30、40、50、60、70、80、90、100、110、120、130、140、150}
max_depth 取 {1、2、3、4、5、6、7、8、9、10、11、12、13、14、15} 
（1111，1111）基因组8位长
'''
def randomForest(n_estimators_value, max_depth_value):

    print("n_estimators_value: " + str(n_estimators_value))
    print("max_depth_value: " + str(max_depth_value))

    train = loadFile("../Data/train-gao.csv")
    test = loadFile("../Data/test-gao.csv")

    train_y = train['Kind']  # 训练集类标
    test_y = test['Kind']  # 测试集类标

    train = train.drop('Kind', axis=1)  # 删除训练集的类标
    train = train.drop('ID', axis=1)  # 删除训练集的ID
    test = test.drop('Kind', axis=1)  # 删除测试集的类标
    test = test.drop('ID', axis=1)  # 删除测试集的ID

    rf = RandomForestClassifier(n_estimators=n_estimators_value,
                                max_depth=max_depth_value,
                                n_jobs=2)
    rf.fit(train, train_y)  # 训练分类器
    predict_test = rf.predict_proba(test)[:, 1]
    roc_auc = metrics.roc_auc_score(test_y, predict_test)
    return roc_auc

def loadFile(filePath):
    fileData = pd.read_csv(filePath)
    return fileData


# Step 1 : 对参数进行编码（用于初始化基因序列，可以选择初始化基因序列，本函数省略）
def geneEncoding(pop_size, chrom_length):  
    pop = [[]]
    for i in range(pop_size):
        temp = []
        for j in range(chrom_length):
            temp.append(random.randint(0, 1))
        pop.append(temp)
    return pop[1:]

# Step 2 : 计算个体的目标函数值
def cal_obj_value(pop):
    objvalue = [];
    variable = decodechrom(pop)
    for i in range(len(variable)):
        tempVar = variable[i]
        n_estimators_value = tempVar[0] * 10
        max_depth_value = tempVar[1]
        aucValue = randomForest(n_estimators_value, max_depth_value)
        objvalue.append(aucValue)
    return objvalue #目标函数值objvalue[m] 与个体基因 pop[m] 对应 


# 对每个个体进行解码，并拆分成单个变量，返回 n_estimators 和 max_depth
def decodechrom(pop):
    variable = []
    n_estimators_value = [];
    max_depth_value = [];
    for i in range(len(pop)):
        res = []
        
        # 计算第一个变量值，即 0101->10(逆转)
        temp1 = pop[i][0:4]
        preValue = 0;
        for pre in range(4):
            preValue += temp1[pre] * (math.pow(2, pre))
        res.append(int(preValue))
        
        # 计算第二个变量值
        temp2 = pop[i][4:8]
        aftValue = 0;
        for aft in range(4):
            aftValue += temp2[aft] * (math.pow(2, aft))
        res.append(int(aftValue))
        variable.append(res)
    return variable


# Step 3: 计算个体的适应值
def calfitvalue(objvalue):
    fitvalue = []
    temp = 0.0
    Cmin = 0;
    for i in range(len(objvalue)):
        if(objvalue[i] + Cmin > 0):
            temp = Cmin + objvalue[i]
        else:
            temp = 0.0
        fitvalue.append(temp)
    return fitvalue


if __name__ == '__main__':
    # pop = geneEncoding(pop_size, chrom_length)
    for i in range(generations):
        print("第 " + str(i) + " 代开始繁殖......")
        obj_value = cal_obj_value(pop) # 计算目标函数值
        print(obj_value)
        fitvalue = calfitvalue(objvalue); #计算个体的适应值