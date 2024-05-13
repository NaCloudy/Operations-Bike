import numpy as np
import pandas as pd
import sympy as sp
from scipy import optimize as op

#曼哈顿距离
def distance(x,y):
    D = abs(x[0]-y[0]) + abs(x[1]-y[1])
    return D

def question_1(K, Loc, X_exp, X_act, w):

    Distance = []
    for i in range(K):
        for j in range(K):
            Distance.append(distance(Loc[i], Loc[j]))

    # 转换为 NumPy 数组
    Distance = np.array(Distance)
    
    restriction = np.zeros((2*K,K**2), dtype=int)
    #约束1
    for i in range(K):
        restriction[i, i*K:(i+1)*K] = 1
    #约束2
    for i in range(K):
        indices = np.arange(i, K**2, K)
        restriction[i+K,indices] = -1
        restriction[i+K,i*K:(i+1)*K] = 1
    X2 = X_act - X_exp
    
    X_restriction = np.concatenate((X_act,X2),axis=0)
    X_restriction = X_restriction.reshape(2*K,1)

    c = Distance #目标函数
    bounds = [(0, None) for var in range(len(c))]

    res = op.linprog(c, A_ub = restriction, b_ub = X_restriction, bounds = bounds, method='highs')
    return res

#站点数
K = 5
#站点位置
Loc = [(1,1),(2,2),(3,3),(4,4),(5,5)]
#第二天需求数
X_exp = np.array([100,60,50,0,90])
#前一晚的实际分布
X_act = np.array([10,90,90,50,60])
w = 1

res = question_1(K, Loc, X_exp, X_act, w)
print(res)