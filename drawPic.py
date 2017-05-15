import pylab as pl
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


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


def userAngleRate():
    x = []
    y = []
    xAxis = [10, 20, 30, 40, 50, 60, 70, 80]
    #  user 1
    y1 = [107, 107, 104, 99, 95, 76, 27, 2]
    #  user 2
    y2 = [140, 140, 140, 140, 135, 99, 24, 6]
    #  user 5
    # y3 = [49, 47, 46, 44, 29, 16, 6, 2]
    # #  user 6
    # y4 = [53, 53, 53, 52, 41, 14, 3, 0]
    # #  user 10
    # y5 = [77, 77, 77, 76, 73, 51, 8, 2]
    x.append(xAxis)
    y.append(y1)
    x.append(xAxis)
    y.append(y2)
    # x.append(xAxis)
    # y.append(y3)
    # x.append(xAxis)
    # y.append(y4)
    # x.append(xAxis)
    # y.append(y5)
    title = 'mode accuracy'
    drawLineGraph(x, y, ['train', 'test'],
                  title, 'epoch', 'accuracy',
                  0, 100, 0, 150)

if __name__ == '__main__':
	print('test')
    #userAngleRate()
