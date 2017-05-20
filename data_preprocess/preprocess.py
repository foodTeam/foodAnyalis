#coding=utf-8
import pylab as pl
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# 25列（包含类标和ID）
feature = ['Kind', 'ID', 'calories', 'carbohydrate', 'fat', 'protein', 'vitamine', 'vta', 'vtc', 'vte', 'carotene', 'thiamine',
           'riboflavin', 'yansuan', 'cholesterol', 'mg', 'ca', 'iron', 'zinc', 'copper', 'mn', 'k', 'p', 'na', 'se']

# 统计特征值的分布: count、mean、std、min、25%、50%、75%、max
def staticFeature(dataset):
    analysis_res = dataset.describe()
    analysis_res.to_csv("fea_static.csv", encoding='utf-8')

# 画每个特征的值分布直方图（全部画）
def drawFeaHistDistr(dataset):
    for i in range(len(feature)):
        column = dataset[feature[i]]
        column.hist(bins=100)
        plt.title(str(feature[i]) + " distribution histogram")
        plt.savefig("featureHist/" + str(feature[i]) + ".png")
        plt.close('all') # 必须要关闭，否则出现图形叠加
        # plt.show()

# 画每个特征的值分布直方图（手动画）
def drawFeaHistDistr2(dataset, fea_name, writeFile):
    column = dataset[fea_name]
    column.hist(bins=100)
    plt.title(str(fea_name) + " distribution histogram")
    plt.savefig(writeFile)
    # plt.show() 

def drawLineGraph(x, y, labelSet, title, xAxisLa, yAxisLa,
                  xStart, xLime, yStart, yLime):
    plotSet = []
    colorSet = ['r', 'b', 'k', 'g', 'm']
    for lx, ly, co, la in zip(x, y, colorSet, labelSet):
        plotSet.append(pl.plot(lx, ly, co, label=la))
    pl.title(title)
    pl.xlabel(xAxisLa)
    pl.ylabel(yAxisLa)
    pl.xlim(xStart, xLime)
    pl.ylim(yStart, yLime)
    pl.legend()
    pl.show()

if __name__ == '__main__':
    filePath = "../Data/train-gao.csv"
    dataset = pd.read_csv(filePath)
    # 统计特征值的分布(写入到当前csv文件)
    # staticFeature(dataset)
    # 特征分布统计图
    drawFeaHistDistr(dataset)
    # drawFeaHistDistr2(dataset, 'calories', 'featureHist/calories1.png')

