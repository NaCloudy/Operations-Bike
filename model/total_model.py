import pulp
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

def question_3(M, K, Loc, Loc_bike, X_exp, w):

    Distance = np.zeros((M,K))
    for i in range(M):
        for j in range(K):
            Distance[i,j] = (distance(Loc_bike[i],Loc[j]))
   
    prob = pulp.LpProblem("Bike_Dispatch_Problem", pulp.LpMinimize)
    Pmk = pulp.LpVariable.dicts("Pmk", ((m, k) for m in range(M) for k in range(K)), cat='Binary')
    prob += pulp.lpSum(Pmk[(m,k)] * Distance[m,k] * w for m in range(M) for k in range(K))

    # 约束条件：每辆自行车只能移动到一个站点或不移动
    for m in range(M):
        prob += pulp.lpSum(Pmk[(m, k)] for k in range(K)) <= 1
    # 约束条件：每个站点移动后的自行车数目大于等于需求量
    for k in range(K):
        prob += pulp.lpSum(Pmk[(m, k)] for m in range(M)) >= X_exp[k]

    prob.solve()
    #状态
    status = pulp.LpStatus[prob.status]
    #决策变量的值
    variable_values = {v.name: v.varValue for v in prob.variables()}
    #目标函数的最优值
    objective_value = pulp.value(prob.objective)

    return status, variable_values, objective_value