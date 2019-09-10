import itertools
import copy

import numpy as np
from gurobipy import *

from .common import *

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

def objective_count(C,Xstar):
    frac_ixs = np.where((Xstar < 1) & (Xstar > 0))
    if len(frac_ixs[0]) == 0:
        return sum(sum(Xstar*C)), [list(range(C.shape[0]))]
    new_rows = []
    new_cols = []
    for i in range(len(frac_ixs[0])):
        if frac_ixs[0][i] < frac_ixs[1][i]:
            new_rows.append(frac_ixs[0][i])
            new_cols.append(frac_ixs[1][i])
    frac_ixs = (np.array(new_rows),np.array(new_cols))
    min_value = np.Inf
    Xsub_min_value = []
    frac_ixs_min_value = []
    for seq in itertools.product([0.,1.], repeat=len(frac_ixs[0])):
        Xsub = copy.copy(Xstar)
        Xsub[frac_ixs] = seq
        Xsub[(frac_ixs[1],frac_ixs[0])] = 1.-np.array(seq)
        obj = sum(sum(Xsub*C))
        if obj < min_value:
            min_value = obj
            Xsub_min_value = [Xsub]
            frac_ixs_min_value = [frac_ixs]
        elif obj == min_value:
            Xsub_min_value.append(Xsub)
            frac_ixs_min_value.append(frac_ixs)
    permutations = []
    for i,Xsub in enumerate(Xsub_min_value):
        r = np.round(np.sum(Xsub,axis=1)*1000)/1000
        ixs = np.argsort(-1*r,kind='stable')
        #print(r,ixs)
        #import pdb; pdb.set_trace()
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
        new_permutations = generate_perms(perm_iters)
        permutations.extend(new_permutations)
    return min_value, permutations

def objective_count_exhaustive(D):
    n = D.shape[0]
    min_value = np.Inf
    solutions = []
    for perm in itertools.permutations(range(n)):
        obj = objective_count_perm(D,perm)
        if obj < min_value:
            min_value = obj
            solutions = [perm]
        elif obj == min_value:
            solutions.append(perm)
    details = {}
    details["P"] = solutions
    return min_value,details

def objective_count_perm(Dorig,perm):
    D = Dorig[perm,:][:,perm]
    score = 0
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            score += np.count_nonzero(D[i,j]>D[i+1:,j])
            score += np.count_nonzero(D[i,j]<D[i,j+1:])
    return score

def compute_C(D):
    c = np.zeros(D.shape)
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            c[i,j] = np.count_nonzero(D[:,j]-D[:,i]<0) + np.count_nonzero(D[i,:]-D[j,:]<0)
           
    return c

def count(D,obj2k_func=lambda obj: round(obj)):
    n = len(D[0])
    m = int(np.sum(D))
   
    c = compute_C(D)

    AP = Model('rankability_hillside_count')
   
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
           # else x[i,j] = 0   # don’t know if we need this. Just want x[i,i]=0 for all i
    AP._x = x
    AP._constraints = []
   
    AP.update()
    AP.setObjective(quicksum(-c[i,j]*x[i,j] for i in range(n) for j in range(n)),GRB.MINIMIZE)
    AP.update()
   
    I = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if j != i and k != j and k != i:
                    I.append((i,j,k))
       
    def sublim(model, where):
        if where == GRB.callback.MIPSOL:
            zeros = [] # make a list zeros
            ones = []
            sol = []
            for i in range(n):
                sol.append(model.cbGetSolution([model._x[i,j] for j in range(n)]))
                       
            for idx in I:
                val = sol[idx[0]][idx[1]] + sol[idx[1]][idx[2]] + sol[idx[2]][idx[0]]

                if val>2:
                    I.remove(idx)

                    #add a violated constraint
                    model.cbLazy(model._x[idx[0],idx[1]] + model._x[idx[1],idx[2]] + model._x[idx[2],idx[0]] <= 2)
                    model._constraints.append(idx)
                       
    AP.params.LazyConstraints = 1
    AP.update()
    AP.optimize(sublim)

    k=obj2k_func(AP.objVal)
               
    details = {"x": get_sol_x_by_x(x,n)(),"c":c,"model": AP}
   
    return k, details


def count_lp(D,obj2k_func=lambda obj: int(obj)):
    n = len(D[0])
    m = int(np.sum(D))
   
    c = compute_C(D)

    AP = Model('rankability_hillside_count')
   
    AP.setParam( 'OutputFlag', False )

    x = {}
   
    for i in range(n):
        for j in range(n):
            x[i,j] = AP.addVar(lb=0,vtype="C",ub=1,name="x(%s,%s)"%(i,j)) #continuous
 
    AP.update()

    for i in range(n):
        for j in range(n):
            if j!=i:
                AP.addConstr(x[i,j] + x[j,i] == 1)
           # else x[i,j] = 0   # don’t know if we need this. Just want x[i,i]=0 for all i
    AP._x = x
    AP._constraints = []
   
    AP.update()
    AP.setObjective(quicksum(c[i,j]*x[i,j] for i in range(n) for j in range(n)),GRB.MINIMIZE)
    AP.update()
   
    I = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if j != i and k != j and k != i:
                    idx = (i,j,k)
                    AP.addConstr(AP._x[idx[0],idx[1]] + AP._x[idx[1],idx[2]] + AP._x[idx[2],idx[0]] <= 2)
                    #I.append((i,j,k))
       
    def sublim(model, where):
        if where == GRB.callback.MIPSOL:
            zeros = [] # make a list zeros
            ones = []
            sol = []
            for i in range(n):
                sol.append(model.cbGetSolution([model._x[i,j] for j in range(n)]))
                       
            for idx in I:
                val = sol[idx[0]][idx[1]] + sol[idx[1]][idx[2]] + sol[idx[2]][idx[0]]

                if val>2:
                    I.remove(idx)

                    #add a violated constraint
                    model.cbLazy(model._x[idx[0],idx[1]] + model._x[idx[1],idx[2]] + model._x[idx[2],idx[0]] <= 2)
                    model._constraints.append(idx)
                       
    #AP.params.LazyConstraints = 1
    AP.Params.Method = 2
    AP.Params.Crossover = 0    
    AP.update()
    AP.optimize()
    #AP.optimize(sublim)

    k=obj2k_func(AP.objVal)
               
    def get_sol_x_by_x(x,n):
        return lambda: np.reshape([x[i,j].X for i in range(n) for j in range(n)],(n,n))
    details = {"x": get_sol_x_by_x(x,n)(),"c":c,"model": AP}
    
    n = details['x'].shape[0]
    r = np.sum(details['x'],axis=1)
    ixs = np.argsort(-1*r)
    mult = 10000
    Xstar_rounded = np.round(mult*details['x'][np.ix_(ixs,ixs)])/mult
    Xstar = details['x'][np.ix_(ixs,ixs)]
    C = details['c'][np.ix_(ixs,ixs)]
    Xstar[range(n),range(n)] = 0
    # for loop to look for binary cross and then try to extend it
    n = Xstar.shape[0]
    fixed_positions = np.zeros((n,)).astype(bool)
    fixed_positions[0] = np.array_equal(np.zeros(n-1,),Xstar_rounded[1:,0])
    # redundant - start_arrow1 = np.array_equal(np.ones(n-1,),Xstar[0,1:])
    fixed_positions[-1] = np.array_equal(np.ones(n-1,),Xstar_rounded[:-1,-1])
    # redundant end_arrow1 = np.array_equal(np.zeros(n-1,),Xstar[-1,:-1])
    #print(start_arrow0,start_arrow1,end_arrow0,end_arrow1)
    for i in range(1,n-1):
        fixed_positions[i] = np.array_equal(np.ones(i,),Xstar_rounded[:i,i]) and np.array_equal(np.zeros(n-i-1,),Xstar_rounded[np.arange(i+1,n),i])
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

    perm_iters = []
    for i,group in enumerate(groups):
        if is_variable[i] == False:
            perm_iters.append((0,[group]))
        else:
            Xsub = Xstar[np.ix_(group,group)]
            Csub = C[np.ix_(group,group)]
            obj_sub, permutations_sub = objective_count(Csub,Xsub)
            perm_iters.append((group[0],permutations_sub))
    permutations = generate_perms(perm_iters)
    details["P"] = [list(np.array(ixs)[perm])[::-1] for perm in permutations]
    P = [tuple(entry) for entry in details['P']]
    details["P"] = list(set(P))
    new_P = []
    for perm in details["P"]:
        obj = objective_count_perm(D,perm)
        if obj == k+1: #remove and figure out +1
            new_P.append(list(perm))
    details["P"] = new_P
    return k, details