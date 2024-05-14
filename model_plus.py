import numpy as np
import pandas as pd
import sympy as sp
from scipy import optimize as op

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

def question_2(M, Loc_bike, K, Loc): 
    Distance_bike_with_site = []
    for i in range(M):
        for j in range(K):
            Distance_bike_with_site.append(distance(Loc_bike[i],Loc[j]))
    Distance_bike_with_site = np.array(Distance_bike_with_site)
    Distance_bike_with_site = Distance_bike_with_site.reshape(M,K)
    
    X_act = [0 for _ in range(K)]
    eta = X_act / (X_exp+0.1)
    Distance_total = 0
    for i in range(M):
        Distance_min = np.min(Distance_bike_with_site[i,:])
        site_min = np.where(Distance_bike_with_site[i,:] == Distance_min)
        Distance_total += Distance_min
        length = len(site_min[0])
        if length == 1:
            X_act[site_min[0][0]]+=1
        else:
            eta_saturation = [eta[j] for j in site_min[0]]
            min_eta_saturation = min(eta_saturation)
            min_eta_index = eta_saturation.index(min_eta_saturation)
            X_act[min_eta_index] += 1
        eta = X_act / (X_exp+0.1)
    return X_act, Distance_total


#自行车数
M = 10
#自行车位置
Loc_bike = [(1,2),(3,2),(1,4),(2,1),(1,2),(5,2),(1,5),(3,4),(6,5),(6,6)]
#站点数
K = 5
#站点位置
Loc = [(1,1),(2,2),(3,3),(4,4),(5,5)]
#第二天需求数
X_exp = np.array([3,1,4,0,2])
w = 1
X_act, Distance_total = question_2(M, Loc_bike, K, Loc)

res = question_1(K, Loc, X_exp, X_act, w)
print(res)

