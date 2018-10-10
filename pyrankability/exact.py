import itertools
import numpy as np
from scipy.misc import comb
import math

from dask import delayed
#import dask.multiprocessing
#import dask.threaded
#from dask.distributed import Client
#client = Client("cluster-address:8786")

from . import common
from . import bilp
from . import pruning

#n=5
#k=15

class ExhaustiveSearch(common.Search):
    def __init__(self,n):
        self.n = n
        self.search_space_size = math.factorial(n)
        self.solution_size = n
        self.max_leave_out = self.solution_size - 1
    
    def print_solution_space_summary(self,leave_out):
        print('Total search space of the problem:',self.search_space_size)
        print('Number of workers',self.calc_num_workers(leave_out))
        print('Work per worker',self.calc_work_per_worker(leave_out))
        #print('Number of subproblems/worker',total_size//work_per_worker)
        print('Achieved by leaving',leave_out,'items out')
        print('Check num_workers x work_per_worker equals total search space:',
             self.calc_num_workers(leave_out)*self.calc_work_per_worker(leave_out) == self.search_space_size)
    
    def calc_work_per_worker(self,leave_out):
        return math.factorial(self.n-leave_out)
    
    def calc_num_workers(self,leave_out):
        return nPr(self.n,leave_out) #math.factorial(leave_out)
    
    def determine_leave_out(self,num_parallel,min_work_per_worker):
        # Decide if we should devide the problem into subtasks and 
        # if so how many and with how much work
        single_thread = True
        leave_out = 1
        while leave_out < self.max_leave_out:
            work_per_worker = self.calc_work_per_worker(leave_out) #nCr(self.total_num_indices-leave_out,k)
            num_subtasks = self.calc_num_workers(leave_out) #nCr(self.solution_size,leave_out)
            if num_subtasks > num_parallel and work_per_worker > min_work_per_worker:
                single_thread = False # Don't run as a single thread
                break
            leave_out += 1
            
        return leave_out, single_thread
    
    def generate_src_inxs(self):
        return list(range(self.solution_size))
    
    def generate_leave_out_iter(self):
        return itertools.permutations(self.generate_src_inxs(),self.leave_out)
    
    def generate_iter(self,subproblem):
        return itertools.permutations(subproblem)
    
    def autoselect_leaveout(self,min_num_workers=2,min_work_per_worker=1):
        leave_out, single_thread = self.determine_leave_out(min_num_workers,min_work_per_worker)
        self.leave_out = leave_out
        self.single_thread = single_thread
    
    def prepare_iterators(self,print_summary=False):
        if print_summary:
            self.print_solution_space_summary(self.leave_out)
        
        # Now create the iterators that can run in parallel
        src_inxs = self.generate_src_inxs()
        leave_out_iter = self.generate_leave_out_iter()#itertools.permutations(src_inxs,leave_out)
        subproblems = []
        iters = []
        entries = []
        for entry in leave_out_iter:
            src_inxs_set = set(src_inxs)
            entry_set = set(entry)
            subproblems.append(src_inxs_set-entry_set)
            entries.append(entry)
            iters.append(self.generate_iter(subproblems[-1]))        
        
        self.solutions_iters = iters
        self.subproblems = subproblems
        self.entries = entries
    
    @staticmethod
    def find_P_parallel(D,solutions_iter,entry):
        k = np.Inf
        P = []
        for j,solution in enumerate(solutions_iter):
            perm = entry+solution
            
            sol_k = common.calc_k(common.permute_D(D,perm))
            if sol_k < k:
                k = sol_k
                P = []
            if sol_k == k:
                P.append(tuple(np.array(perm)))
        return int(k),P
               
    def get_find_P_parallel_func(self):
        return lambda D,solutions_iter,entry: ExhaustiveSearch.find_P_parallel(D,solutions_iter,entry)

    def find_P(self,D,num_parallel):
        """find_P finds the solutions to the rankability problem.
        """
        opt_P = []
        opt_k = np.Inf
        #values = [delayed(self.get_find_P_parallel_func())(D,solutions_iter,self.entries[i]) for i,solutions_iter in enumerate(self.solutions_iters)]
        results = []
        for i,solutions_iter in enumerate(self.solutions_iters):
            results.append(self.get_find_P_parallel_func()(D,solutions_iter,self.entries[i]))
        #results = compute(*values, get=dask.multiprocessing.get, num_workers=num_parallel)
        #results = compute(*values, scheduler='single-threaded')#, num_workers=num_parallel)
        for k,P in results:
            if k < opt_k:
                opt_P = []
                opt_k = k
            if k == opt_k:
                opt_P.extend(P)
        self.k = opt_k
        self.P = opt_P
        
class RecusiveKBoundedSearch(common.Search):
    def __init__(self,D,target_search_space_at_leaf=1000,leave_out=4):
        self.target_search_space_at_leaf = target_search_space_at_leaf
        self.leave_out = leave_out
        self.D = D
        bilp_res = bilp.bilp(self.D)
        self.k = bilp_res[0]
        
    def find_P(self,client=None):
        calc_k_spec = pruning.generate_calc_k_spec(self.D,np.arange(self.D.shape[0]))
        self.k,self.P,self.skipped,self.searched,leaf,self.exact_solution_found = pruning.find_P(
            self.D,self.target_search_space_at_leaf,self.leave_out,self.k,self.D.shape[0],(),calc_k_spec,num_parallel=18,client=client)
    