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

def bilp_two_most_distant(D,lazy=False,verbose=True):
    first_k, first_details = bilp(D,lazy=lazy,verbose=verbose)
    if verbose:
        print('Finished first optimization. Obj:',first_k)
        
    c = compute_C(D)
    
    n = D.shape[0]
        
    AP = Model('hillside')

    x = {}
    y = {}
    u = {}
    v = {}
    b = {}
    for i in range(n-1):
        for j in range(i+1,n):
            x[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="x(%s,%s)"%(i,j)) #binary
            y[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="y(%s,%s)"%(i,j)) #binary
            u[i,j] = AP.addVar(name="u(%s,%s)"%(i,j),vtype=GRB.BINARY,lb=0,ub=1) #nonnegative
            v[i,j] = AP.addVar(name="v(%s,%s)"%(i,j),vtype=GRB.BINARY,lb=0,ub=1) #nonnegative
    AP.update()
    
    for i in range(n-1):
        for j in range(i+1,n):
            for k in range(j+1,n):
                trans_cons = []
                trans_cons.append(AP.addConstr(x[i,j] + x[j,k] - x[i,k] <= 1))
                trans_cons.append(AP.addConstr(x[i,j] + x[j,k] - x[i,k] >= 0))
                trans_cons.append(AP.addConstr(y[i,j] + y[j,k] - y[i,k] <= 1))
                trans_cons.append(AP.addConstr(y[i,j] + y[j,k] - y[i,k] >= 0))
                if lazy:
                    for cons in trans_cons:
                        cons.setAttr(GRB.Attr.Lazy,1)
    AP.update()
    
    AP.addConstr(quicksum((c[i,j]-c[j,i])*x[i,j]+c[j,i] for i in range(n-1) for j in range(i+1,n)) == first_k)
    AP.addConstr(quicksum((c[i,j]-c[j,i])*y[i,j]+c[j,i] for i in range(n-1) for j in range(i+1,n)) == first_k)

    AP.update()
    for i in range(n-1):
        for j in range(i+1,n):
            AP.addConstr(u[i,j] - v[i,j] == x[i,j] - y[i,j])
            AP.addConstr(u[i,j] + v[i,j] <= 1)
    AP.update()

    AP.setObjective(quicksum((u[i,j]+v[i,j]) for i in range(n-1) for j in range(i+1,n)),GRB.MAXIMIZE)
    AP.setParam( 'OutputFlag', verbose )
    AP.update()
        
    if verbose:
        print('Start optimization')
    tic = time.perf_counter()
    AP.update()
    AP.optimize()
    toc = time.perf_counter()
    if verbose:
        print(f"Optimization in {toc - tic:0.4f} seconds")
        print('End optimization')
    
    sol_x = get_sol_x_by_x(x,n)()
    sol_y = get_sol_x_by_x(y,n)()
    sol_v = get_sol_uv_by_x(v,n)()
    sol_u = get_sol_uv_by_x(u,n)()
    r = np.sum(sol_x,axis=0)
    ranking = np.argsort(r)
    perm_x = tuple([int(item) for item in ranking])
    
    r = np.sum(sol_y,axis=0)
    ranking = np.argsort(r)
    perm_y = tuple([int(item) for item in ranking])
    
    k_x = np.sum(np.sum(c*sol_x))
    k_y = np.sum(np.sum(c*sol_y))
    
    details = {"obj":AP.objVal,"k_x": k_x, "k_y":k_y, "perm_x":perm_x,"perm_y":perm_y, "x": sol_x,"y":sol_y,"u":sol_u,"v":sol_v}
            
    return first_k,details
    
def compute_C(D):
    c = np.zeros(D.shape)
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            c[i,j] = np.count_nonzero(D[:,j]-D[:,i]<0) + np.count_nonzero(D[i,:]-D[j,:]<0) 
           
    return c

def bilp(D_orig,num_random_restarts=0,lazy=False,verbose=False,find_pair=False):
    n = D_orig.shape[0]
    
    temp_dir = tempfile.mkdtemp()
    
    Pfirst = []
    Pfinal = []
    objs = []
    xs = []
    
    pair_Pfirst = []
    pair_Pfinal = []
    pair_objs = []
    pair_xs = []
    first_k = None
    
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

            AP.update()
            AP.write(model_file)
            
        tic = time.perf_counter()
        c = compute_C(D)
        if first_k is not None:
            AP.addConstr(quicksum(c[i,j]*x[i,j] for i in range(n) for j in range(n))==first_k)
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

        k=int(AP.objVal)
        if first_k is None:
            first_k = k

        P = []
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
        
        if find_pair:
            AP = read(model_file)
            x = {}
            for i in range(n):
                for j in range(n):
                    x[i,j] = AP.getVarByName("x(%s,%s)"%(i,j))
            AP.addConstr(quicksum(c[i,j]*x[i,j] for i in range(n) for j in range(n))==k)
            AP.update()
            u={}
            v={}
            b={}
            for i in range(n):
                for j in range(i+1,n):
                    u[i,j] = AP.addVar(name="u(%s,%s)"%(i,j),lb=0)
                    v[i,j] = AP.addVar(name="v(%s,%s)"%(i,j),lb=0)
                    b[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="b(%s,%s)"%(i,j))
            AP.update()
            for i in range(n):
                for j in range(i+1,n):
                    AP.addConstr(u[i,j] - v[i,j] == x[i,j] - sol_x[i,j])
                    AP.addConstr(u[i,j] >= b[i,j])
                    AP.addConstr(v[i,j] <= 1 - b[i,j])
            AP.update()
            
            #AP.setObjective(quicksum(u[i,j]-v[i,j] for i in range(n-1) for j in range(i+1,n)),GRB.MAXIMIZE)
            AP.setObjective(quicksum(u[i,j]+v[i,j] for i in range(n-1) for j in range(i+1,n)),GRB.MAXIMIZE)
            AP.setParam( 'OutputFlag', False )
            AP.update()
            
            if verbose:
                print('Start pair optimization %d'%ix)
            tic = time.perf_counter()
            AP.optimize()
            toc = time.perf_counter()
            if verbose:
                print(f"Optimization in {toc - tic:0.4f} seconds")
                print('End optimization %d'%ix)
            
            #k=int(AP.objVal)

            P = []
            sol_x = get_sol_x_by_x(x,n)()
            sol_u = get_sol_x_by_x(u,n)()
            sol_v = get_sol_x_by_x(v,n)()
            r = np.sum(sol_x,axis=0)
            ranking = np.argsort(r)
            key = tuple([int(item) for item in ranking])
            perm = tuple(perm_inxs[np.array(key)])
            P.append(perm)
            reorder = np.argsort(perm_inxs)
            pair_xs.append(sol_x[np.ix_(reorder,reorder)])
            k = np.sum(np.sum(c*sol_x))

            if ix == 0:
                pair_Pfirst = P
                pair_xfirst = get_sol_x_by_x(x,n)() 
            
            pair_Pfinal.extend(P)
            pair_objs.append(k)
    
    details = {"Pfirst": Pfirst, "P":Pfinal,"x": xfirst,"objs":objs,"xs":xs}
    pair_details = None
    if find_pair:
        pair_details = {"Pfirst": pair_Pfirst, "P":pair_Pfinal,"x": pair_xfirst,"objs":pair_objs,"xs":pair_xs}
    details["pair_details"] = pair_details
    
    shutil.rmtree(temp_dir)
        
    return k,details

