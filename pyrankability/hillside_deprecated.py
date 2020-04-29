# -*- coding: utf-8 -*-
import itertools
import copy
import multiprocessing

import numpy as np
from gurobipy import *
from joblib import Parallel, delayed

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

def objective_count(D,C,Xstar,min_value,group):
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
            obj = sum(sum(Xsub*C))
            #if obj < min_value:
            #    min_value = obj
            #    Xsub_min_value = [Xsub]
            #    frac_ixs_min_value = [frac_ixs]
            if obj == min_value:
                Xsub_min_value.append(Xsub)
                frac_ixs_min_value.append(frac_ixs)
        
        return min_value,Xsub_min_value,frac_ixs_min_value
    
    results = Parallel(n_jobs=cpu_count)(delayed(compute)(init_seq) for init_seq in init_seqs)
    # combine
    #min_value = np.Inf
    Xsub_min_value = []
    frac_ixs_min_value = []
    for result in results:
        #if result[0] < min_value:
        #    Xsub_min_value = result[1]
        #    frac_ixs_min_value = result[2]
        #    min_value = result[0]
        if result[0] == min_value:
            Xsub_min_value.extend(result[1])
            frac_ixs_min_value.extend(result[2])

    permutations = set()
    prev_n_permutations = 0
    for i,Xsub in enumerate(Xsub_min_value):
        r = np.sum(Xsub[np.ix_(group,group)],axis=1) #np.round(np.sum(Xsub,axis=1)*1000)/1000
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
        new_permutations = [tuple(perm) for perm in generate_perms(perm_iters)]
        permutations = permutations.union(set(new_permutations))
        #print("# of new permutations added",len(permutations)-prev_n_permutations)
        prev_n_permutations = len(permutations)
        
    return min_value, permutations

def objective_count_ga(D,C,Xstar):
    k,details = count(D,max_solutions=1)
    
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
    max_trials = np.Inf
    
    def ga_objective(seq):
        Xsub = copy.copy(Xstar)
        Xsub[frac_ixs] = seq
        Xsub[(frac_ixs[1],frac_ixs[0])] = 1.-np.array(seq)
        return -1*sum(sum(Xsub*C)),
    
    import random

    from deap import base
    from deap import creator
    from deap import tools
    from deap import algorithms
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    # Attribute generator 
    toolbox.register("attr_bool", random.randint, 0, 1)
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(frac_ixs[0]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", ga_objective)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=2)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    def initIndividual(icls, content):
        return icls(content)

    def initPopulation(pcls, ind_init, filename):
        with open(filename, "r") as pop_file:
            contents = json.load(pop_file)
        return pcls(ind_init(c) for c in contents)

    toolbox = base.Toolbox()

    toolbox.register("individual_guess", initIndividual, creator.Individual)
    toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, "my_guess.json")

    pop = toolbox.population_guess()
    
    #pop = toolbox.population(n=1000)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=500, 
                                   stats=stats, verbose=True)
    
    c = 0
    print("Going to loop for",2**len(frac_ixs[0]))
    for seq in itertools.product([0.,1.], repeat=len(frac_ixs[0])):
        if c > max_trials:
            break
        c+=1
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
            
    print("1")
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
    print("2")
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
            #+ np.count_nonzero((D[:,j]==D[:,i]))
            #np.count_nonzero((D[:,j]==D[:,i]) & (D[:,j]>0))
           
    return c

def _count(D,c=None,relaxation_method="constraints"):
    n = D.shape[0]

    if c is None:
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
    AP.setObjective(quicksum(c[i,j]*x[i,j] for i in range(n) for j in range(n)),GRB.MINIMIZE)
    AP.update()
   
    I = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if j != i and k != j and k != i:
                    idx = (i,j,k)                    
                    if relaxation_method != "cut":
                        AP.addConstr(AP._x[idx[0],idx[1]] + AP._x[idx[1],idx[2]] + AP._x[idx[2],idx[0]] <= 2).setAttr(GRB.Attr.Lazy,1)
                    else:
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
                      
    if relaxation_method == "cut":
        AP.params.LazyConstraints = 1
    AP.update()
        
    return AP,sublim

def count(D,max_solutions=None,iterations=1,c=None,relaxation_method="constraints"):
    k,details = _run(D,c=c,max_solutions=max_solutions,iterations=iterations,relaxation_method=relaxation_method)
                  
    return k, details

def _run(D_orig,c=None,model_func=_count,iterations=1,max_solutions=None,obj2k_func=lambda obj: int(obj),relaxation_method="constraints"):
    def get_sol_x_by_x(x,n):
        return lambda: np.reshape([x[i,j].X for i in range(n) for j in range(n)],(n,n))

    P = {}
    prev_size_P = 0
    for iteration in range(iterations):
        perm_inxs = np.random.permutation(range(D_orig.shape[0]))
        D = permute_D(D_orig,perm_inxs)
        #print("Iteration",iteration+1)
        model,sublim = model_func(D,relaxation_method=relaxation_method,c=c)
        # Limit how many solutions to collect
        if max_solutions is not None:
            model.setParam(GRB.Param.PoolSolutions, max_solutions)
            # Limit the search space by setting a gap for the worst possible solution that will be accepted
            model.setParam(GRB.Param.PoolGap, .5)
            # do a systematic search for the k-best solutions
            model.setParam(GRB.Param.PoolSearchMode, 2)
            model.update()

        if relaxation_method == "cut":
            model.optimize(sublim)
        else:
            model.optimize()

        k=obj2k_func(model.objVal)
        
        details = {"P":[],'x':get_sol_x_by_x(model._x,D_orig.shape[0])()}

        if max_solutions is None:
            return k,details

        for perm in _get_P(D,model,get_sol_x_by_x):
            P[tuple(perm_inxs[perm])] = True
        print("Number of new solutions",len(P.keys())-prev_size_P)
        prev_size_P = len(P.keys())
    details = {"P":list(P.keys()),"x":get_sol_x_by_x(model._x,D.shape[0])()}
    return k, details

def _get_P(D,model,get_sol_x_func):
    # Print number of solutions stored
    nSolutions = model.SolCount

    # Print objective values of solutions and create a list of those that have the same objective value as the optimal
    alternative_solutions = []
    for e in range(nSolutions):
        model.setParam(GRB.Param.SolutionNumber, e)
        if model.PoolObjVal == model.objVal:
            alternative_solutions.append(e)
    
    # get all alternative solutions
    P = {}
    for e in alternative_solutions:
        model.setParam(GRB.Param.SolutionNumber, e)
        sol_x = get_sol_x_func(model._x,D.shape[0])()
        r = np.sum(sol_x,axis=0)
        ranking = np.argsort(-1*r)
        #rowsum=np.sum(D+sol_x,axis=0)
        #ranking=np.argsort(rowsum)+1
        key = tuple([int(item) for item in ranking])
        if key not in P:
            P[key] = True
    P = [list(perm) for perm in P.keys()]
    return P


def count_lp(D,obj2k_func=lambda obj: round(obj),c=None,relaxation_method="constraints",level=1,constraints=[],x_eq_1_constraints=[]):
    if relaxation_method == "cut":
        raise Exception("relaxation method of cut is not valid here")
    n = D.shape[0]#len(D[0])
    #m = int(np.sum(D))
   
    if c is None:
        c = compute_C(D)

    AP = Model('rankability_hillside_count')
   
    AP.setParam( 'OutputFlag', False )

    x = {}
            
    for idx in x_eq_1_constraints:
        x[idx[0],idx[1]]=1
        if idx[0] != idx[1]:
            x[idx[1],idx[0]]=0  
        
    for i in range(n):
        for j in range(n):
            if not ((i,j) in x_eq_1_constraints or (j,i) in x_eq_1_constraints):
                x[i,j] = AP.addVar(lb=0,vtype="C",ub=1,name="x(%s,%s)"%(i,j)) #continuous
                                  
    AP.update()

    for i in range(n):
        for j in range(n):
            if j!=i:
                if (i,j) in x_eq_1_constraints or (j,i) in x_eq_1_constraints:
                    continue
                else:
                    AP.addConstr(x[i,j] + x[j,i] == 1)
           # else x[i,j] = 0   # don’t know if we need this. Just want x[i,i]=0 for all i
    AP._x = x
    AP._constraints = []
   
    AP.update()
    AP.setObjective(quicksum(c[i,j]*x[i,j] for i in range(n) for j in range(n)),GRB.MINIMIZE)
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
                        #else:
                        #    print("skipping",idx)
                    elif relaxation_method == "cut":
                        I.append((i,j,k))
                    else:
                        AP.addConstr(AP._x[idx[0],idx[1]] + AP._x[idx[1],idx[2]] + AP._x[idx[2],idx[0]] <= 2)
               
    for idx in constraints:
        AP.addConstr(AP._x[idx[0],idx[1]] + AP._x[idx[1],idx[2]] + AP._x[idx[2],idx[0]] <= 2)
       
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
                       
    if relaxation_method == "cut":
        AP.params.LazyConstraints = 1
    AP.Params.Method = 2
    AP.Params.Crossover = 0    
    AP.update()
    if relaxation_method == "constraints":
        AP.optimize()
    elif relaxation_method == "cut":
        AP.optimize(sublim)
    else:
        AP.optimize()
        
    def get_sol_x_by_x(x,n):
        def myfunc():
            values = []
            for i in range(n):
                for j in range(n):
                    try:
                        values.append(x[i,j].X)
                    except:
                        values.append(round(x[i,j]))
            return np.reshape(values,(n,n))
        return myfunc
    k=obj2k_func(AP.objVal)
    details = {"P":[],"x": get_sol_x_by_x(x,n)(),"c":c}
        
    return k,details

def threshold_x(x,lower_cut=1e-3,upper_cut=1-1e-3):
    x = x.copy()
    cut_ixs = np.where(x < lower_cut)
    x[cut_ixs] = 0.
    cut_ixs = np.where(x > upper_cut)
    x[cut_ixs] = 1.
    return x

def find_P_from_x(D,k,details,lower_cut=1e-3,upper_cut=1-1e-3):
    n = details['x'].shape[0]
    r = np.sum(details['x'],axis=1)
    ixs = np.argsort(-1*r)
    Xstar = details['x'][np.ix_(ixs,ixs)]
    Xstar = threshold_x(Xstar,lower_cut=lower_cut,upper_cut=upper_cut)
    C = details['c'][np.ix_(ixs,ixs)]
    Dordered = D[np.ix_(ixs,ixs)]
    Xstar[range(n),range(n)] = 0
    # for loop to look for binary cross and then try to extend it
    n = Xstar.shape[0]
    fixed_positions = np.zeros((n,)).astype(bool)
    fixed_positions[0] = np.array_equal(np.zeros(n-1,),Xstar[1:,0])
    fixed_positions[-1] = np.array_equal(np.ones(n-1,),Xstar[:-1,-1])
    for i in range(1,n-1):
        fixed_positions[i] = np.array_equal(np.ones(i,),Xstar[:i,i]) and np.array_equal(np.zeros(n-i-1,),Xstar[np.arange(i+1,n),i])
    #print("Fixed items",ixs[fixed_positions])
    #print("Fixed positions mask",fixed_positions)
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
        
    info = {"Xstar":Xstar,"fixed_positions":fixed_positions,"groups":groups,"C":C,"Dordered":Dordered,"ixs":ixs}
    return None,info

def find_P_from_x_old(D,k,details,lower_cut=1e-3,upper_cut=1-1e-3):         
    n = details['x'].shape[0]
    r = np.sum(details['x'],axis=1)
    ixs = np.argsort(-1*r)
    Xstar = details['x'][np.ix_(ixs,ixs)]
    Xstar = Xstar.copy()
    cut_ixs = np.where(Xstar < lower_cut)
    Xstar[cut_ixs] = 0.
    cut_ixs = np.where(Xstar > upper_cut)
    Xstar[cut_ixs] = 1.
    #Xstar_rounded = np.round(mult*Xstar)/mult
    C = details['c'][np.ix_(ixs,ixs)]
    Dordered = D[np.ix_(ixs,ixs)]
    Xstar[range(n),range(n)] = 0
    # for loop to look for binary cross and then try to extend it
    n = Xstar.shape[0]
    fixed_positions = np.zeros((n,)).astype(bool)
    fixed_positions[0] = np.array_equal(np.zeros(n-1,),Xstar[1:,0])
    # redundant - start_arrow1 = np.array_equal(np.ones(n-1,),Xstar[0,1:])
    fixed_positions[-1] = np.array_equal(np.ones(n-1,),Xstar[:-1,-1])
    # redundant end_arrow1 = np.array_equal(np.zeros(n-1,),Xstar[-1,:-1])
    #print(start_arrow0,start_arrow1,end_arrow0,end_arrow1)
    for i in range(1,n-1):
        fixed_positions[i] = np.array_equal(np.ones(i,),Xstar[:i,i]) and np.array_equal(np.zeros(n-i-1,),Xstar[np.arange(i+1,n),i])
    print("Fixed items",ixs[fixed_positions])
    print("Fixed positions mask",fixed_positions)
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

    new_k,new_details = count(Dordered,c=C,max_solutions = 1)
    if k+1 != new_k:
        raise Exception("ERROR: k and new_k do not match",k,new_k)
    
    perm_iters = []
    for i,group in enumerate(groups):
        if is_variable[i] == False:
            perm_iters.append((0,[group]))
        else:
            Xsub = np.round(new_details['x'])
            Xsub[np.ix_(group,group)] = Xstar[np.ix_(group,group)]
            Csub = C#[np.ix_(group,group)]
            obj_sub, permutations_sub = objective_count(Dordered[np.ix_(group,group)],Csub,Xsub,k+1,group)
            perm_iters.append((group[0],permutations_sub))
    print("3")
    permutations = generate_perms(perm_iters)
    print("4")
    details["P"] = [list(np.array(ixs)[perm])[::-1] for perm in permutations]
    P = [tuple(entry) for entry in details['P']]
    details["P"] = list(set(P))
    new_P = []
    for perm in details["P"]:
        obj = objective_count_perm(D,perm)
        if obj == k+1:# or (k == 0 and obj == 0): #remove and figure out +1
            new_P.append(list(perm))
        else:
            import pdb; pdb.set_trace()
            #print("Is this ever going to happen?")
    details["P"] = new_P
    details["Xstar_cut"] = Xstar
    return k+1, details # TODO: why k + 1?
