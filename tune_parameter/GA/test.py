#coding=utf-8
from __future__ import division
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from random import randint


def test():
    pop = [[0, 1, 0, 1, 0, 1, 0, 1] for i in range(3)]
    print(pop)
    print(type(pop))
    print(pop[0][1])


if __name__ == '__main__':
    # test()
    # print(int(7.98))
    # s=Series(np.random.randn(10).cumsum(),index=np.arange(0,100,10))
    # s.plot()
    # df=DataFrame(np.random.randn(10,4).cumsum(0),columns=['A','B','C','D'],index=np.arange(0,100,10))
    # df.plot()  不需要
    # cons_value = (0.20-0.01）/ (32 - 0.01) # 0.19 / 31.99
    # print(cons_value)
    x = randint(1,10000)
    print(x)