#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 25

@author: yoichi_izunaga + amy langville + paul anderson
"""

import numpy as np
from gurobipy import *
import itertools

from .common import *

def optimize_weights(M,w_values,min_support,max_solutions=100000):
    k_values = {}
    P_values = {}
    p_values = {}
    D_values = {}
    completed = {}
    nunique = 0
    nskipped = 0
    for w in itertools.combinations_with_replacement(w_values,M.shape[1]):
        w = np.array(w)*1./sum(w)
        w_tuple = tuple(w)
        D = define_D(M,w,min_support)
        D_tuple = tuple(map(tuple, D))
        if D_tuple in completed:
            eq_w = completed[D_tuple]
            k_values[w_tuple] = k_values[eq_w]
            p_values[w_tuple] = p_values[eq_w]
            P_values[w_tuple] = P_values[eq_w]
            nskipped += 1
            continue
        k,p,P = bilp(D,max_solutions=max_solutions)

        k_values[w_tuple] = k
        p_values[w_tuple] = p
        P_values[w_tuple] = P
        D_values[w_tuple] = D

        completed[D_tuple] = w_tuple

        nunique += 1
    
    # Find the best
    best_k = np.Inf
    best_w = None
    for w, k in k_values.items():
        if k < best_k:
            best_k = k
            best_w = w
    
    return best_w, k_values, P_values, p_values, D_values

def bilp_from_data(M,w,min_support,max_solutions=1):
    D = define_D(M,w,min_support)
    return bilp(D,max_solutions=max_solutions)

def bilp(D,max_solutions=None,smartk=None,method=["orig","mos2","ref"][0]):
    if method == "mos2":
        return bilp_mos2(D,max_solutions=max_solutions,smartk=smartk)
    elif method == "orig":
        return bilp_orig(D,max_solutions=max_solutions,smartk=smartk)
    elif method == "ref":
        return bilp_ref(D,max_solutions=max_solutions,smartk=smartk)
    else:
        raise Exception("Invalid method "+str(method))

def bilp_orig(D,max_solutions=None,smartk=None):
    n = len(D[0])
    m = int(np.sum(D))

    AP = Model("rankability")
    
    AP.setParam( 'OutputFlag', False )

    x = {}
    y = {}
    for i in range(n):
        for j in range(n):
            x[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="x(%s,%s)"%(i,j)) #binary
            y[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="y(%s,%s)"%(i,j))  #binary
    AP.update()

    for i in range(n):
        for j in range(n):
            AP.addConstr(x[i,j] + y[i,j] <= 1)
            AP.addConstr(x[i,j] <= 1 - D[i,j])
            AP.addConstr(y[i,j] <= D[i,j])
    for j in range(1,n):
        for i in range(0,j):
            AP.addConstr(D[i,j] + x[i,j] - y[i,j] + D[j,i] + x[j,i] - y[j,i] == 1)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                if j != i and k != j and k != i:
                    AP.addConstr(D[i,j] + x[i,j] - y[i,j] + D[j,k] + x[j,k] - y[j,k] + D[k,i] + x[k,i] - y[k,i] <= 2)

    if smartk == None:
        smartk = (n*n-n)/2
    AP.addConstr(quicksum(x[i,j] for i in range(n) for j in range(n)) + quicksum(y[i,j] for i in range(n) for j in range(n)) <= smartk)

    AP.update()

    AP.setObjective(quicksum(x[i,j] for i in range(n) for j in range(n)) + quicksum(y[i,j] for i in range(n) for j in range(n)),GRB.MINIMIZE)
    AP.update()
    
    get_sol_x_func = get_sol_x_by_x(x,n)
    get_sol_y_func = get_sol_y_by_y(y,n)
       
    return _run(D,AP,get_sol_x_func,get_sol_y_func,max_solutions)

def _run(D,model,get_sol_x_func,get_sol_y_func,max_solutions,obj2k_func=lambda obj: int(obj)):
    # Limit how many solutions to collect
    if max_solutions != None:
        model.setParam(GRB.Param.PoolSolutions, max_solutions)
        # Limit the search space by setting a gap for the worst possible solution that will be accepted
        model.setParam(GRB.Param.PoolGap, .5)
        # do a systematic search for the k-best solutions
        if max_solutions != None:
            model.setParam(GRB.Param.PoolSearchMode, 2)
        model.update()

    model.optimize()

    k=obj2k_func(model.objVal)

    if max_solutions == None:
        return k,[]
    
    P = _get_P(D,model,get_sol_x_func,get_sol_y_func)
    return k, P

def get_sol_x_by_x(x,n):
    return lambda: np.reshape([round_Xn(x[i,j].Xn) for i in range(n) for j in range(n)],(n,n))

def get_sol_y_by_y(y,n):
    return lambda: np.reshape([round_Xn(y[i,j].Xn) for i in range(n) for j in range(n)],(n,n))

def get_sol_x_by_z(D,z,n):
    return lambda: np.reshape([1 if round_Xn(z[i,j].Xn) == 1 and D[i,j] == 0 and i != j else 0 for i in range(n) for j in range(n)],(n,n))

def get_sol_y_by_z(D,z,n):
    return lambda: np.reshape([1 if round_Xn(z[i,j].Xn) == 0 and D[i,j] == 1 and i != j else 0 for i in range(n) for j in range(n)],(n,n))

def get_sol_z_by_z(D,z,n):
    return lambda: np.reshape([round_Xn(z[i,j].Xn) for i in range(n) for j in range(n)],(n,n))

"""If zij = 0 and dij = 1, then set yij = 1. If zij = 1 and dij = 0,
292 then set xij = 1. Then k is the number of nonzeros in X plus the number of nonzeros
293 in Y, i.e., k = nnz(X) + nnz(Y)."""
    
def _get_P(D,model,get_sol_x_func,get_sol_y_func):
    # Print number of solutions stored
    nSolutions = model.SolCount

    # Print objective values of solutions and create a list of those that have the same objective value as the optimal
    alternative_solutions = []
    for e in range(nSolutions):
        model.setParam(GRB.Param.SolutionNumber, e)
        if model.PoolObjVal == model.objVal:
            alternative_solutions.append(e)
    
    # print all alternative solutions
    P = {}
    for e in alternative_solutions:
        model.setParam(GRB.Param.SolutionNumber, e)
        sol_x = get_sol_x_func()
        sol_y = get_sol_y_func()
        rowsum=np.sum(D+sol_x-sol_y,axis=0)
        ranking=np.argsort(rowsum)+1
        key = tuple([int(item) for item in ranking])
        if key not in P:
            P[key] = True
    P = [list(perm) for perm in P.keys()]
    return P

def bilp_mos2(A,max_solutions=None,smartk=None):
    n = len(A[0])
    m = int(np.sum(A))
    AP = Model("rankabilityAltFormReferee")
    AP.setParam( 'OutputFlag', False )
    z = {}
    for i in range(n):
        for j in range(n):
            z[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="z(%s,%s)"%(i,j)) #binary
    AP.update()

    for j in range(1,n):
        for i in range(0,j):
            AP.addConstr(z[i,j] + z[j,i] == 1)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                if j != i and k != j and k != i:
                    AP.addConstr(z[i,j] + z[j,k] + z[k,i] <= 2)

    AP.update()

    AP.setObjective(quicksum(A[i,j]*z[i,j] for i in range(n) for j in range(n)),GRB.MAXIMIZE)
    AP.update()
               
    get_sol_x_func = get_sol_x_by_z(A,z,n)
    get_sol_y_func = get_sol_y_by_z(A,z,n)
    get_sol_z_func = get_sol_z_by_z(A,z,n)
    
    obj2k_func=lambda obj: int(np.count_nonzero(get_sol_x_func())) + int(np.count_nonzero(get_sol_y_func()))
       
    results = _run(A,AP,get_sol_x_func,get_sol_y_func,max_solutions,obj2k_func=obj2k_func)
    #print("D:")
    #print(A)
    #print("z:")
    #print(get_sol_z_func())
    #print("x:")
    #print(get_sol_x_func())
    #print("y:")
    #print(get_sol_y_func())
    
    return results
               
def bilp_ref(A,max_solutions=None,smartk=None):
    n = len(A[0])
    m = int(np.sum(A))

    AP = Model("rankabilityAltFormReferee")
    AP.setParam( 'OutputFlag', False )
    z = {}
    for i in range(n):
        for j in range(n):
            z[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="z(%s,%s)"%(i,j)) #binary
    AP.update()

    for j in range(1,n):
        for i in range(0,j):
            AP.addConstr(z[i,j] + z[j,i] == 1)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                if j != i and k != j and k != i:
                    AP.addConstr(z[i,j] + z[j,k] + z[k,i] <= 2)

    AP.update()

    AP.setObjective(quicksum(A[i,j]*z[i,j] for i in range(n) for j in range(n)),GRB.MAXIMIZE)
    AP.update()
    
    get_sol_x_func = get_sol_x_by_z(A,z,n)
    get_sol_y_func = get_sol_y_by_z(A,z,n)
    
    obj2k_func=lambda obj: np.count_nonzero(get_sol_x_func()) + np.count_nonzero(get_sol_y_func())
       
    return _run(A,AP,get_sol_x_func,get_sol_y_func,max_solutions,obj2k_func=obj2k_func)