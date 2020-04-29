# -*- coding: utf-8 -*-
import itertools
import copy
import multiprocessing
import tempfile
import os
import shutil
import time

import numpy as np
from gurobipy import *
from joblib import Parallel, delayed

from .common import *

def compute_C(D):
    c = np.zeros(D.shape)
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            c[i,j] = np.count_nonzero(D[:,j]-D[:,i]<0) + np.count_nonzero(D[i,:]-D[j,:]<0) 
           
    return c

def bilp(D_orig,max_solutions=None,num_random_restarts=0,lazy=False,verbose=False):
    n = D_orig.shape[0]
    
    temp_dir = tempfile.mkdtemp()
    
    Pfirst = []
    Pfinal = []
    objs = []
    xs = []
    for ix in range(num_random_restarts+1):
        if ix > 0:
            perm_inxs = np.random.permutation(range(D_orig.shape[0]))
            D = D_orig[perm_inxs,:][:,perm_inxs]
        else:
            perm_inxs = np.arange(n)
            D = copy.deepcopy(D_orig)
        model_file = os.path.join(temp_dir,"model.mps")
        if os.path.isfile(model_file):
            AP = read(model_file)
            x = {}
            for i in range(n):
                for j in range(n):
                    x[i,j] = AP.getVarByName("x(%s,%s)"%(i,j))
        else:
            AP = Model('lop')

            x = {}

            for i in range(n):
                for j in range(n):
                    x[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="x(%s,%s)"%(i,j)) #binary

            AP.update()

            for i in range(n):
                for j in range(n):
                    if j!=i:
                        AP.addConstr(x[i,j] + x[j,i] == 1)
                    else:
                        AP.addConstr(x[i,i] == 0)

            AP.update()
            I = []
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        if j != i and k != j and k != i:
                            idx = (i,j,k)                    
                            if lazy:
                                AP.addConstr(x[idx[0],idx[1]] + x[idx[1],idx[2]] + x[idx[2],idx[0]] <= 2).setAttr(GRB.Attr.Lazy,1)
                            else:
                                AP.addConstr(x[idx[0],idx[1]] + x[idx[1],idx[2]] + x[idx[2],idx[0]] <= 2)

            if max_solutions is not None and max_solutions > 1:
                AP.setParam(GRB.Param.PoolSolutions, max_solutions)
                # Limit the search space by setting a gap for the worst possible solution that will be accepted
                AP.setParam(GRB.Param.PoolGap, .5)
                # do a systematic search for the k-best solutions
                AP.setParam(GRB.Param.PoolSearchMode, 2)
            AP.update()
            AP.write(model_file)
        
        tic = time.perf_counter()
        c = compute_C(D)
        AP.setObjective(quicksum(c[i,j]*x[i,j] for i in range(n) for j in range(n)),GRB.MINIMIZE)
        AP.setParam( 'OutputFlag', False )
        AP.update()
        toc = time.perf_counter()
        if verbose:
            print(f"Updating opjective in {toc - tic:0.4f} seconds")
        
        if verbose:
            print('Start optimization %d'%ix)
        tic = time.perf_counter()
        AP.optimize()
        toc = time.perf_counter()
        if verbose:
            print(f"Optimization in {toc - tic:0.4f} seconds")
            print('End optimization %d'%ix)

        def get_sol_x_by_x(x,n):
            def myfunc():
                values = []
                for i in range(n):
                    for j in range(n):
                        values.append(int(x[i,j].X))
                return np.reshape(values,(n,n))
            return myfunc
        k=int(AP.objVal)

        P = []
        if max_solutions is not None and max_solutions > 1:
            if verbose:
                print('Running pool search')
            for perm in _get_P(D,AP,get_sol_x_by_x):
                P.append(tuple(perm_inxs[np.array(perm)]))
        elif max_solutions is not None and max_solutions == 1:
            sol_x = get_sol_x_by_x(x,n)()
            r = np.sum(sol_x,axis=0)
            ranking = np.argsort(r)
            key = tuple([int(item) for item in ranking])
            perm = tuple(perm_inxs[np.array(key)])
            P.append(perm)
            reorder = np.argsort(perm_inxs)
            xs.append(sol_x[np.ix_(reorder,reorder)])
        
        if ix == 0:
            Pfirst = P
            xfirst = get_sol_x_by_x(x,n)()
        
        Pfinal.extend(P)
        objs.append(k)
    
    details = {"Pfirst": Pfirst, "P":list(set(Pfinal)),"x": xfirst,"objs":objs,"xs":xs}
    
    shutil.rmtree(temp_dir)
        
    return k,details

