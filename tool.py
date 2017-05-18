import pylab as pl
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def readFile(filePath):


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
    x = []
    y = []
    #xAxis = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    xAxis = range(80, 110, 1)

    filePath = "score.txt"
    meanList,estimatorList = readFile(filePath)

    x.append(xAxis)
    y.append(meanList)

    title = 'mode accuracy'
    drawLineGraph(x, y, ['mean score', 'test'],
                  title, 'the number of tree', 'accuracy',
                  70, 110, 0.89, 0.905)
