import itertools
import numpy as np
import pandas as pd
import copy

from . import common
from . import bilp
from . import pruning

def bilp_compute(D,inxs,max_solutions):
    return bilp.bilp(common.permute_D(D,inxs),max_solutions=max_solutions,method="mos2")

def find_P(D, max_solutions=1000, bilp_method="mos2"):
    k,P = bilp.bilp(D,method=bilp_method,max_solutions=max_solutions)
    return k, P

class LargeProblemSolver(common.Search):
    def __init__(self,D,num_iterations=10,max_num_starting_solutions=10,client=None,print_flag=True):
        self.num_iterations = num_iterations
        self.max_num_starting_solutions = max_num_starting_solutions
        self.D = D
        
        permutation_inxs = [np.random.permutation(D.shape[0]) for i in range(num_iterations)]
        if client != None:
            values = client.map(bilp_compute,
                                list(itertools.repeat(D,num_iterations)), 
                                permutation_inxs,
                                list(itertools.repeat(max_num_starting_solutions,num_iterations))
                               )
            results = client.gather(values)
        else:
            results = list(map(bilp_compute,
                          list(itertools.repeat(D,num_iterations)), 
                          permutation_inxs,
                          list(itertools.repeat(max_num_starting_solutions,num_iterations))
                         ))
        self.starting_results = results
        self.print_flag = print_flag
        # Now setup the P_to_search
        k = None
        P = []
        c = 0
        for ki,Pi in results:
            if k == None:
                k = ki
            if k != ki:
                raise "Error: k != ki"
            Pi_list = permutation_inxs[c][np.array(Pi)-1].tolist()
            P.extend(Pi_list)
            c+=1
        self.k = k
        self.P = []
        num_new_solutions = self.add_P(P)
        if self.print_flag:
            print("Number of new solutions added:",num_new_solutions)
        self.P_to_search = copy.deepcopy(self.P)
        self.record = {}
        self.P_searched = {}
        
    def add_P(self,P,check_k=False):
        if check_k:
            for Psolution in P:
                sol_k = common.calc_k(common.permute_D(self.D,Psolution))
                if sol_k != self.k:
                    raise Exception("k value is not correct when trying to add solution")
                    
        orig_num_solutions = len(self.P)
        self.P = pd.DataFrame(self.P+P).drop_duplicates().values.tolist()
        new_num_solutions = len(self.P)
        return new_num_solutions - orig_num_solutions
        
    def check_record(self,Psolution,global_inxs):
        key = (tuple(Psolution),tuple(global_inxs))
        if key in self.record:
            return True
        else:
            return False
    
    def add_record(self,Psolution,global_inxs,new_solutions):
        key = (tuple(Psolution),tuple(global_inxs))
        self.record[key] = new_solutions
        
    def find_P_pop(self,Psolution,size_of_chunk=8,client=None):
        num_new_solutions_added = 0
        if size_of_chunk > self.D.shape[0]:
            size_of_chunk = self.D.shape[0]
        chunks = common.chunks(np.arange(self.D.shape[0]),size_of_chunk)
        chunks_to_search = []
        for global_inxs in chunks:
            if self.check_record(Psolution,global_inxs) == False: # Not done yet
                chunks_to_search.append(global_inxs)
                
        if client == None:
            new_solutions_per_chunk = list(map(expand_solution,
                                               itertools.repeat(self.D,len(chunks_to_search)),
                                               itertools.repeat(Psolution,len(chunks_to_search)),
                                               chunks_to_search))
            for i,new_solutions_tuple in enumerate(new_solutions_per_chunk):
                k,new_solutions,skipped,searched,leaf,exact_solution_found = new_solutions_tuple
                global_inxs = chunks_to_search[i]
                self.add_record(Psolution,global_inxs,new_solutions)
                num_added = self.add_P(new_solutions,check_k=True)
                if self.print_flag:
                    print("Number of new solutions added:",num_added)
                    print("Number of skipped solutions:",skipped)
                    print("Number of searched solutions:",searched)
                num_new_solutions_added += num_added
        
        if tuple(Psolution) not in self.P_searched:
            self.P_searched[tuple(Psolution)] = 0
        self.P_searched[tuple(Psolution)] += 1
                
        return num_new_solutions_added
    
    def find_P(self,size_of_chunk=8,client=None):
        while len(self.P_to_search) > 0:
            Psolution = np.array(self.P_to_search.pop())
            if tuple(Psolution) not in self.P_searched:
                if self.print_flag:
                    print("Performing local search for",tuple(Psolution))
                num_new_solutions = self.find_P_pop(Psolution,size_of_chunk=size_of_chunk,client=client)
                #if self.print_flag:
                #    print("Total number of new solutions:",num_new_solutions)
                
    def other_to_dict(self):
        return {"exact_solution_found": False}

def expand_solution(D,Psolution,global_inxs,target_search_space_at_leaf=1000,leave_out=4,client=None):
    subset = Psolution[global_inxs]
    Dspec = common.permute_D(D,subset)
    k_spec = common.calc_k(Dspec)
    global_D = common.permute_D(D,Psolution)
    global_sol_k = common.calc_k(global_D)
    
    calc_k_spec = pruning.generate_calc_k_spec(global_D,global_inxs)
    k, P, skipped, searched, leaf, exact_solution_found = pruning.find_P(
        Dspec,target_search_space_at_leaf,leave_out,k_spec,Dspec.shape[0],(),calc_k_spec,client=client)
    new_solutions = []
    for local_solution in P:
        new_solution = copy.copy(Psolution)
        new_solution[global_inxs] = subset[np.array(local_solution)]
        Dperm = common.permute_D(D,new_solution)
        new_solutions.append(list(new_solution))
    return k,new_solutions,skipped,searched,leaf,exact_solution_found


