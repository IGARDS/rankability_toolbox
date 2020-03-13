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

def generate_perms(perm_iters):
    if len(perm_iters) == 0:
        return [[]]
    rem_perms = generate_perms(perm_iters[1:])
    perms = []
    offset = perm_iters[0][0]
    for perm in perm_iters[0][1]:
        for rem_perm in rem_perms:
            new_perm = list(np.array(perm)+offset)+rem_perm
            if len(new_perm) > 0:
                perms.append(new_perm)
    return perms

def threshold_x(x,lower_cut=1e-3,upper_cut=1-1e-3):
    x = x.copy()
    cut_ixs = np.where(x < lower_cut)
    x[cut_ixs] = 0.
    cut_ixs = np.where(x > upper_cut)
    x[cut_ixs] = 1.
    return x

def compare_objective_values(o1,o2,tol=1**-6):
    if abs(o1-o2) <= tol:
        return True
    return False

def find_P_from_x(D,k,details,lower_cut=1e-3,upper_cut=1-1e-3):
    n = details['x'].shape[0]
    r = np.sum(details['x'],axis=1)
    ixs = np.argsort(-1*r)
    Xstar = details['x'][np.ix_(ixs,ixs)]
    Xstar = threshold_x(Xstar,lower_cut=lower_cut,upper_cut=upper_cut)
    Dordered = D[np.ix_(ixs,ixs)]
    Xstar[range(n),range(n)] = 0
    # for loop to look for binary cross and then try to extend it
    n = Xstar.shape[0]
    fixed_positions = np.zeros((n,)).astype(bool)
    fixed_positions[0] = np.array_equal(np.zeros(n-1,),Xstar[1:,0])
    fixed_positions[-1] = np.array_equal(np.ones(n-1,),Xstar[:-1,-1])
    for i in range(1,n-1):
        fixed_positions[i] = np.array_equal(np.ones(i,),Xstar[:i,i]) and np.array_equal(np.zeros(n-i-1,),Xstar[np.arange(i+1,n),i])
    # construct groups
    groups = []
    is_variable = []
    start_var = False
    group = []
    if fixed_positions[0] == False:
        start_var = True
    group.append(0)
    for i in range(1,n):
        # check to see if this is a change and we need to save the results
        if fixed_positions[i] == True and start_var:
            groups.append(group)
            is_variable.append(True)
            group = []
            start_var = False
        elif fixed_positions[i] == False and not start_var:
            groups.append(group)
            is_variable.append(False)
            group = []
            start_var = True
        group.append(i)

    if len(group) > 0:
        if start_var:
            is_variable.append(True)
        else:
            is_variable.append(False)
        groups.append(group)
        
    new_k,new_details = lp(Dordered,relaxation_method='constraints')
    perm_iters = []
    for i,group in enumerate(groups):
        if is_variable[i] == False:
            perm_iters.append((0,[group]))
        else:
            Xsub = np.round(new_details['x'])
            Xsub[np.ix_(group,group)] = Xstar[np.ix_(group,group)]
            obj_sub, permutations_sub = objective_count(Dordered,Dordered[np.ix_(group,group)],Xsub,k,group)
            perm_iters.append((group[0],permutations_sub))
    permutations = generate_perms(perm_iters)
    details["P"] = [list(np.array(ixs)[perm])[::-1] for perm in permutations]
    P = [tuple(entry) for entry in details['P']]
    details["P"] = list(set(P))
    new_P = []
    num_removed = 0
    for perm in details["P"]:
        obj = objective_count_perm(D,perm[::-1])
        if compare_objective_values(obj,k): # or (k == 0 and obj == 0): #remove and figure out +1
            new_P.append(list(perm))
        else:
            num_removed += 1
            continue
            #import pdb; pdb.set_trace()
            #print("Is this ever going to happen?")
    details["P"] = new_P
    details["Xstar_cut"] = Xstar
        
    info = {"Xstar":Xstar,"fixed_positions":fixed_positions,"groups":groups,"Dordered":Dordered,"ixs":ixs, "num_removed":num_removed}
    return new_P,info

def objective_count(Dordered,D,Xstar,min_value,group):
    frac_ixs = np.where((Xstar < 1) & (Xstar > 0))
    if len(frac_ixs[0]) == 0:
        return sum(sum(Xstar*Dordered)), [list(range(D.shape[0]))]
    new_rows = []
    new_cols = []
    for i in range(len(frac_ixs[0])):
        if frac_ixs[0][i] < frac_ixs[1][i]:
            new_rows.append(frac_ixs[0][i])
            new_cols.append(frac_ixs[1][i])
    frac_ixs = (np.array(new_rows),np.array(new_cols))
            
    cpu_count = multiprocessing.cpu_count()
    init_digits = int(np.log2(cpu_count))
    init_seqs = list(itertools.product([0.,1.], repeat=init_digits))
    if len(frac_ixs[0])-len(init_seqs) < 1:
        init_seqs = [[]]

    print("Going to loop for",2**len(frac_ixs[0])*1./len(init_seqs))

    def compute(init_seq,min_value=min_value,Xstar=Xstar):
        #min_value = np.Inf
        Xsub_min_value = []
        frac_ixs_min_value = []
        max_trials = np.Inf
        c = 0
        for rem_seq in itertools.product([0.,1.], repeat=len(frac_ixs[0])-len(init_seq)):
            seq = list(init_seq) + list(rem_seq)
            if c > max_trials:
                break
            c+=1
            Xsub = copy.copy(Xstar)
            Xsub[frac_ixs] = seq
            Xsub[(frac_ixs[1],frac_ixs[0])] = 1.-np.array(seq)
            obj = np.sum(np.sum(Xsub*Dordered))
            if compare_objective_values(obj,min_value):
                Xsub_min_value.append(Xsub)
                frac_ixs_min_value.append(frac_ixs)
        
        return min_value,Xsub_min_value,frac_ixs_min_value
    
    results = Parallel(n_jobs=cpu_count)(delayed(compute)(init_seq) for init_seq in init_seqs)
    # combine
    #min_value = np.Inf
    Xsub_min_value = []
    frac_ixs_min_value = []
    for result in results:
        if compare_objective_values(result[0],min_value):
            Xsub_min_value.extend(result[1])
            frac_ixs_min_value.extend(result[2])

    permutations = set()
    prev_n_permutations = 0
    for i,Xsub in enumerate(Xsub_min_value):
        r = np.sum(Xsub[np.ix_(group,group)],axis=1)
        ixs = np.argsort(-1*r,kind='stable')
        sets = {}
        for j,v in enumerate(r):
            if v not in sets:
                sets[v] = [j]
            else:
                sets[v].append(j)
        perm_iters = []
        last_v = None
        for ix in ixs:
            v = r[ix]
            if last_v is None or last_v != v:
                perm_iters.append((0,itertools.permutations(sets[v])))
                last_v = v
        new_permutations = [tuple(perm) for perm in generate_perms(perm_iters)]
        permutations = permutations.union(set(new_permutations))
        prev_n_permutations = len(permutations)
        
    return min_value, permutations

def objective_count_perm(Dorig,perm):
    D = Dorig[perm,:][:,perm]
    return np.sum(np.sum(np.triu(D)))

def objective_count_exhaustive(D,X):
    n = D.shape[0]
    min_value = -np.Inf
    solutions = []
    for perm in itertools.permutations(range(n)):
        obj = objective_count_perm(D,perm)
        if not compare_objective_values(obj,min_value) and obj > min_value:
            min_value = obj
            solutions = [perm]
        elif compare_objective_values(obj,min_value):
            solutions.append(perm)
    details = {}
    details["P"] = solutions
    return min_value,details