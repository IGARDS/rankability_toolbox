import numpy as np
from numpy import ix_

from . import bilp
from . import common
from . import exact

import copy

# Generator allows us to perform local optimization in the context of a global solution
def generate_calc_k_spec(global_D,global_inxs):
    return lambda D,rem_inxs,partial_perm,prior_partial_perm: calc_k_local(D,rem_inxs,partial_perm,prior_partial_perm,global_D,global_inxs) 

def calc_k_local(D,rem_inxs,partial_perm,prior_partial_perm,global_D,global_inxs):
    rem_inxs_partial = tuple(rem_inxs[list(partial_perm)])
    rem_inxs_rest = np.delete(rem_inxs,partial_perm)
    Dspecial = common.permute_D(D,prior_partial_perm+rem_inxs_partial+tuple(rem_inxs_rest))
    # need to make the remaining end of matrix perfect
    # this currently is done with a loop, but needs to be done with numpy arrays
    # it also could be calculated with bilp for a tigther bound?
    for i1 in range(len(prior_partial_perm)+len(rem_inxs_partial),D.shape[0]):
        for j1 in range(len(prior_partial_perm)+len(rem_inxs_partial),D.shape[0]):
            if i1 == j1:
                Dspecial[i1,j1] = 0
            elif i1 > j1:
                Dspecial[i1,j1] = 0
            else:
                Dspecial[i1,j1] = 1

    global_D_copy = copy.copy(global_D)
    global_D_copy[ix_( global_inxs, global_inxs )] = Dspecial
    #return common.calc_k(global_D_copy),rem_inxs_partial,rem_inxs_rest
    return common.calc_k(Dspecial),rem_inxs_partial,rem_inxs_rest

def multi_compute_func(D, final_k, leave_out,target_search_space_at_leaf, entries, n, prior_partial_perm,rem_inxs,calc_k_spec):
    return [compute_func(D,final_k,leave_out,target_search_space_at_leaf,n,prior_partial_perm,partial_perm,rem_inxs,calc_k_spec) for partial_perm in entries]

def compute_func(D,final_k,leave_out,target_search_space_at_leaf,n,prior_partial_perm,partial_perm,rem_inxs,calc_k_spec):
    # local versions of variables
    res_k = None
    res_P = []
    skipped = 0
    searched = 0
    leaf = False

    sol_k,rem_inxs_partial,rem_inxs_rest = calc_k_spec(D,rem_inxs,partial_perm,prior_partial_perm)
    if sol_k <= final_k:
        rem_sol_k, rem_sol_P, rem_skipped, rem_searched, rem_leaf, exact_solution_found = find_P(D,target_search_space_at_leaf,leave_out,final_k-sol_k,
                                                                                                 n-leave_out,
                                                                                                 prior_partial_perm+rem_inxs_partial,
                                                                                                 calc_k_spec)
        skipped += rem_skipped
        searched += rem_searched
        if rem_sol_k != None:
            res_P_j = []
            for i,rem_perm in enumerate(rem_sol_P):
                if rem_leaf: # if this was a leaf, then do the conversion
                    rem_perm = tuple(rem_inxs_rest[list(rem_perm)])
                res_P_j.append(rem_inxs_partial+rem_perm)
            res_P = res_P_j # res_P + res_P_j
            res_k = sol_k + rem_sol_k
    else:
        skipped_exhaustive_search = exact.ExhaustiveSearch(n-leave_out)
        skipped += skipped_exhaustive_search.search_space_size
        exact_solution_found = True
    return res_k, res_P, skipped, searched, leaf, exact_solution_found

def find_P(D,target_search_space_at_leaf,leave_out,k,n,prior_partial_perm,calc_k_spec,num_parallel=1,client=None): 
    leaf = False
    skipped = 0
    searched = 0
    exact_solution_found = True

    exhaustive_search = exact.ExhaustiveSearch(n)
    # Base case 1: not large enough to leave out the number specified in leave_out
    # Base case 2: the search space is less than the target search space for a leaf node
    if n <= leave_out or exhaustive_search.search_space_size <= target_search_space_at_leaf:
        leaf = True
        exhaustive_search.leave_out = 0
        exhaustive_search.prepare_iterators(print_summary=False)
        D_copy = copy.copy(D)
        D = np.delete(D,prior_partial_perm,axis=0)
        D = np.delete(D,prior_partial_perm,axis=1)
        # We can solve this with an exact solution
        if exhaustive_search.search_space_size <= target_search_space_at_leaf:
            searched += exhaustive_search.search_space_size
            exhaustive_search.find_P(D,4)
            perm = tuple(exhaustive_search.P[0]) # pick one optimal permutation
            rem_inxs = np.arange(len(perm))+len(prior_partial_perm)
            k_global,rem_inxs_partial,rem_inxs_rest = calc_k_spec(D_copy,rem_inxs,perm,prior_partial_perm)
            if len(prior_partial_perm) > 0:
                prior_k = common.calc_k(common.permute_D(D_copy,prior_partial_perm))
            else:
                prior_k = 0
            k_adjusted = exhaustive_search.k #k_global-prior_k
            if k_adjusted > k:
                return None, [], skipped, searched, leaf, exact_solution_found
            elif k_adjusted == k:
                return exhaustive_search.k, exhaustive_search.P, skipped, searched, leaf, exact_solution_found
            else:
                #import pdb; pdb.set_trace()
                raise Exception("ERROR: Found k less than lower bound. Found k: " + str(int(exhaustive_search.k)) + " Lower bound: "+str(int(k)))
        else:
            solution = bilp.bilp(D,max_solutions=target_search_space_at_leaf)
            bilp_k = solution[0]
            bilp_P = (bilp_P-1).aslist()
            exact_solution_found = False
            return bilp_k, bilp_P, skipped, searched, leaf, exact_solution_found

    # Do the recursive step
    exhaustive_search.leave_out = leave_out
    exhaustive_search.prepare_iterators(print_summary=False)
    res_P = []
    rem_inxs = np.delete(np.arange(D.shape[0]),prior_partial_perm)
    res_k = None

    if client == None: 
        lambda_multi_compute_func = lambda entries: multi_compute_func(
            D, k, leave_out,target_search_space_at_leaf, entries, n, prior_partial_perm,rem_inxs,calc_k_spec)
        results = lambda_multi_compute_func(exhaustive_search.entries)
    else: # Right now we are only parallezing the outermost loop
        def len_chunks(l,n):
            return len(range(0,len(l),n))
        def chunks(l, n):
            chunks_ = []
            for i in range(0, len(l), n):
                #yield l[i:i + n]
                chunks_.append(l[i:i + n])
            return chunks_
        desired_num_chunks = 100
        size_of_chunk = int(len(exhaustive_search.entries)*1./desired_num_chunks)
        #size_of_chunk = int(1.*len(exhaustive_search.entries)/(num_parallel*1000))
        print('A. Begin breaking up search space into chunks of size',size_of_chunk)
        #chunks_ = chunks(exhaustive_search.entries,int(1.*len(exhaustive_search.entries)/num_parallel))
        chunks_ = chunks(exhaustive_search.entries,size_of_chunk)
        len_of_chunks = len_chunks(exhaustive_search.entries,size_of_chunk)
        print('A. Done')
        print('B. Begin map')
        values = client.map(multi_compute_func,
                            list(itertools.repeat(D,len_of_chunks)), 
                            list(itertools.repeat(k,len_of_chunks)), 
                            list(itertools.repeat(leave_out,len_of_chunks)),
                            list(itertools.repeat(target_search_space_at_leaf,len_of_chunks)),
                            chunks_,
                            list(itertools.repeat(n,len_of_chunks)),
                            list(itertools.repeat(prior_partial_perm,len_of_chunks)),
                            list(itertools.repeat(rem_inxs,len_of_chunks)),
                            list(itertools.repeat(calc_k_spec,len_of_chunks))
                           )
        print('B. Done')
        print('C. Begin gather')
        gather_results = client.gather(values)
        print('C. Done')
        results = []
        for gresults in gather_results:
            results.extend(gresults)

    # Now combine the results
    for ind_results in results:
        res_k_j, res_P_j, skipped_j, searched_j, leaf_j, exact_solution_found_j = ind_results
        if exact_solution_found_j == False:
            exact_solution_found = False
        if res_k_j != None:
            res_k = res_k_j
            res_P = res_P + res_P_j
        skipped += skipped_j
        searched += searched_j
    return res_k, res_P, skipped, searched, leaf, exact_solution_found