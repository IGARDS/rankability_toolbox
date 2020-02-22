# -*- coding: utf-8 -*-
import itertools
import copy
import multiprocessing

import numpy as np
from gurobipy import *
from joblib import Parallel, delayed

from .common import *

def bilp(D):
    n = D.shape[0]

    AP = Model('lop')
   
    AP.setParam( 'OutputFlag', False )

    x = {}
   
    for i in range(n):
        for j in range(n):
            x[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="x(%s,%s)"%(i,j)) #binary
 
    AP.update()

    for i in range(n):
        for j in range(n):
            if j!=i:
                AP.addConstr(x[i,j] + x[j,i] == 1)
    AP._x = x
    AP._constraints = []
   
    AP.update()
    AP.setObjective(quicksum(D[i,j]*x[i,j] for i in range(n) for j in range(n)),GRB.MAXIMIZE)
    AP.update()
   
    I = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if j != i and k != j and k != i:
                    idx = (i,j,k)                    
                    I.append((i,j,k))
                    AP.addConstr(AP._x[idx[0],idx[1]] + AP._x[idx[1],idx[2]] + AP._x[idx[2],idx[0]] <= 2)
                      
    AP.update()
    AP.optimize()
    
    def get_sol_x_by_x(x,n):
        def myfunc():
            values = []
            for i in range(n):
                for j in range(n):
                    values.append(x[i,j].X)
            return np.reshape(values,(n,n))
        return myfunc
    k=AP.objVal
    details = {"P":[],"x": get_sol_x_by_x(x,n)()}
        
    return k,details



def lp(D,relaxation_method=None,level=2):
    n = D.shape[0]

    AP = Model('lop')
   
    AP.setParam('OutputFlag', False )

    x = {}
        
    for i in range(n):
        for j in range(n):
            x[i,j] = AP.addVar(lb=0,vtype="C",ub=1,name="x(%s,%s)"%(i,j)) #continuous
                                  
    AP.update()

    for i in range(n):
        for j in range(n):
            if j!=i:
                AP.addConstr(x[i,j] + x[j,i] == 1)
    AP._x = x
    AP._constraints = []
   
    AP.update()
    AP.setObjective(quicksum(D[i,j]*x[i,j] for i in range(n) for j in range(n)),GRB.MAXIMIZE)
    AP.update()
    
    #orig = copy.deepcopy(AP)
   
    I = []
    values = set()
    for i,j in x.keys():
        values.add(i)
        values.add(j)
    for i in values:#range(n):
        for j in values:#range(n):
            for k in values:#range(n):
                if j != i and k != j and k != i:
                    idx = (i,j,k)
                    if relaxation_method == "constraints":
                        if (idx[0],idx[1]) in x and (idx[1],idx[2]) in x and (idx[2],idx[0]) in x:
                            AP.addConstr(AP._x[idx[0],idx[1]] + AP._x[idx[1],idx[2]] + AP._x[idx[2],idx[0]] <= 2).setAttr(GRB.Attr.Lazy,level)
                    else:
                        AP.addConstr(AP._x[idx[0],idx[1]] + AP._x[idx[1],idx[2]] + AP._x[idx[2],idx[0]] <= 2)

    AP.Params.Method = 2
    AP.Params.Crossover = 0    
    AP.update()
    AP.optimize()
        
    def get_sol_x_by_x(x,n):
        def myfunc():
            values = []
            for i in range(n):
                for j in range(n):
                    values.append(x[i,j].X)
            return np.reshape(values,(n,n))
        return myfunc
    k=AP.objVal
    details = {"P":[],"x": get_sol_x_by_x(x,n)()}
        
    return k,details

def threshold_x(x,lower_cut=1e-3,upper_cut=1-1e-3):
    x = x.copy()
    cut_ixs = np.where(x < lower_cut)
    x[cut_ixs] = 0.
    cut_ixs = np.where(x > upper_cut)
    x[cut_ixs] = 1.
    return x
