import itertools
import numpy as np
from numpy import ix_
import math
import tempfile
from dask import delayed
from dask.distributed import as_completed
import joblib
import copy
import uuid
import os
import math

from operator import itemgetter

from . import bilp
from . import common

import functools
print = functools.partial(print, flush=True)

def reindex(orig_inxs, order_inxs):
    return list(np.array(orig_inxs)[order_inxs])

def find_P(path_D, o, t, bilp_test=False, bilp_method="mos2", client=None, max_search_space=np.Inf, prune_history=False, uuid1=common.random_generator(),expand=True,verbose=False):
    def load_D():
        return np.genfromtxt(path_D, delimiter=",")
        
    orig_right_perm = list(range(load_D().shape[0]))
    bilp_res = bilp.bilp(load_D(),method=bilp_method)
    k_optimal = bilp_res[0]
    if verbose:
        print("k",k_optimal)
    #n = len(orig_right_perm)
    a = 0.5
    b = -1.0
    c = -k_optimal
    d = (b**2) - (4*a*c)
    n_worst = (-b+math.sqrt(d))/(2*a)
    o = min([round(n_worst),len(orig_right_perm)])
    o = 4
    if verbose:
        print('setting o to',o)

    add_left_perms_grouped_file = "/dev/shm/add_left_perms_grouped"+uuid1+".joblib"
    if verbose:
        print(add_left_perms_grouped_file)
    
    def _no_left_calc_k(add_left_perms):
        D = load_D()
        chunk_results = []
        #i = 0
        for add_left_perm in add_left_perms:
            chunk_results.append((add_left_perm,bilp.bilp(D[ix_(add_left_perm, add_left_perm)],method=bilp_method,max_solutions=1)))
            #if i % 10 == 0:
            #    print(i)
            #i += 1
            
        add_left_perms_grouped = []
        for add_left_perm,bilp_results in chunk_results:
            kvalue,P = bilp_results
            if kvalue > k_optimal:
                continue
            order_ixs = np.array(P[0])-1
            key = tuple(np.array(add_left_perm)[order_ixs].tolist())
            add_left_perms_grouped.append(key)
        return add_left_perms_grouped
    
    def _exhaustive_search(left_perm, search_perm, right_perm):
        all_search_perms = [list(perm) for perm in itertools.permutations(
            search_perm, len(search_perm))]

        def local_calc_k(local_search_perm): return _calc_k(
            left_perm + local_search_perm + right_perm)
        results = list(map(local_calc_k, all_search_perms))
        k_values = []
        for k_value, Dperm_right in results:
            k_values.append(k_value)
        k = np.min(k_values)
        optimal_inxs = np.where(k_values == k)[0]
        P = [left_perm + opt_perm + right_perm for opt_perm in np.array(
            all_search_perms)[optimal_inxs].tolist()]
        return k, P, [list(search_perm) for search_perm in np.array(all_search_perms)[optimal_inxs]]

    def _calc_k(left_perm, right_perm=[]):
        perm = left_perm + right_perm
        D = load_D()
        Dperm = D[ix_(perm, perm)]
        # Left perm is optimal, but right perm is not, so we'll set it to the perfect submatrix.
        # This could be replaced to a call to bilp for a tighter bound
        if len(right_perm) > 0:
            rn = len(right_perm)
            right_inxs = np.arange(len(left_perm),Dperm.shape[0])
            Dperm_right = Dperm[ix_(right_inxs, right_inxs)]
            Dperm[ix_(right_inxs, right_inxs)] = np.triu(np.ones((rn, rn)), 1)
        else:
            Dperm_right = None
        Dperfect = np.triu(np.ones((Dperm.shape[0], Dperm.shape[0])), 1)
        k = int(np.sum(np.abs(Dperfect-Dperm)))
        return k, Dperm_right
    
    def generate_right_perm(calc_k_left_perm):
        calc_k_right_perm = list(set(orig_right_perm) - set(calc_k_left_perm))
        _dict = {}
        for key in list(itertools.permutations(calc_k_right_perm, o)):
            _dict[tuple(sorted(key))] = True
        return list(_dict.keys())
    
    def generate_right_perm2(right_perm, result_calc_k_left_perm):
        next_right_perm = []
        result_calc_k_left_perm = set(result_calc_k_left_perm)
        for key in right_perm:
            if result_calc_k_left_perm.isdisjoint(key):
                next_right_perm.append(key)
        return next_right_perm
    
    def _check(left_perm,add_left_perm):
        def _convert(perm):
            actual_perm = []
            for key in perm:
                actual_perm += key
            return actual_perm  
    
        if type(add_left_perm) == tuple:
            add_left_perm = [add_left_perm]
        
        # move the contents of add_left_perm out of prev_right_perm, this wrinkle is done to
        # save memory
        if len(left_perm) == 0:
            calc_k_left_perm = _convert(add_left_perm)
        else:
            calc_k_left_perm = _convert(left_perm+add_left_perm)
        calc_k_right_perm = list(set(orig_right_perm) - set(calc_k_left_perm)) #list(set(prev_right_perm) - set(add_left_perm))
            
        perm_k, Dperm_right = _calc_k(calc_k_left_perm, right_perm=calc_k_right_perm)
        if perm_k <= k_optimal:
            if bilp_test: # extra bilp test
                kbilp_res = bilp.bilp(Dperm_right,method=bilp_method)
                kbilp = kbilp_res[0]
                if kbilp+perm_k <= k_optimal: # still need to continue
                    return True, 0, left_perm, add_left_perm, calc_k_left_perm, calc_k_right_perm
            else:
                return True, 0, left_perm, add_left_perm, calc_k_left_perm, calc_k_right_perm
            
        skipped = 0
        # pruned
        #if prune_history:
        #    skipped = math.factorial(len(calc_k_right_perm))
        #else:
        #    skipped = 0
        return False, skipped, left_perm, add_left_perm, calc_k_left_perm, calc_k_right_perm
    
    def _get_groups(D):
        add_left_perms = itertools.combinations(range(D.shape[0]), o)
        desired_number_chunks = 20
        num_add_left_perms = common.nCr(D.shape[0],o) #len(add_left_perms)
        chunk_length = round(num_add_left_perms*1./desired_number_chunks)
        add_left_perms_chunks = []
        for i in range(0, num_add_left_perms, chunk_length):
            add_left_perms_chunks.append(itertools.islice(add_left_perms,i,i+chunk_length)) #l[i:i + n])
        #add_left_perms_chunks = common.chunks(add_left_perms,chunk_length)
        if verbose:
            print('mapping',len(add_left_perms_chunks))
        #_no_left_calc_k(add_left_perms_chunks[0])
        futures = client.map(_no_left_calc_k,add_left_perms_chunks)
        if verbose:
            print('gathering')
        results = client.gather(futures)
        if verbose:
            print('done gathering')
        add_left_perms_grouped = []
        for chunk_add_left_perms_grouped in results:
            add_left_perms_grouped.extend(chunk_add_left_perms_grouped)
        """
        for chunk_add_left_perms_grouped, chunk_add_left_perms_grouped_best_ks in results:
            for key in chunk_add_left_perms_grouped.keys():
                if key not in add_left_perms_grouped or chunk_add_left_perms_grouped_best_ks[key] < add_left_perms_grouped_best_ks[key]:
                    add_left_perms_grouped[key] = chunk_add_left_perms_grouped[key]
                    add_left_perms_grouped_best_ks[key] = chunk_add_left_perms_grouped_best_ks[key]
                elif chunk_add_left_perms_grouped_best_ks[key] == add_left_perms_grouped_best_ks[key]:
                    add_left_perms_grouped[key].extend(chunk_add_left_perms_grouped[key])
        """
        return add_left_perms_grouped
    
    def _multi_check(add_left_perms,left_perm):
        results = []
        add_left_perms_grouped = joblib.load(add_left_perms_grouped_file)
        for add_left_perm in add_left_perms:
            results.append(_check(left_perm,add_left_perm))
        return results
    
    def _get_futures(left_perm,right_perm):
        add_left_perms = right_perm # list(itertools.permutations(right_perm, D.shape[0]/o))
        num_add_left_perms = len(right_perm) #common.nPr(len(right_perm),o)
        desired_number_chunks = 1000
        chunk_length = round(num_add_left_perms*1./desired_number_chunks)
        if chunk_length == 0:
            chunk_length = num_add_left_perms
        #chunk_length = 4000
        add_left_perms_chunks = common.chunks(add_left_perms,chunk_length)
        #import pdb; pdb.set_trace()
        futures = client.map(_multi_check,
                             add_left_perms_chunks,
                             itertools.repeat(left_perm,len(add_left_perms_chunks)))
        return futures
    
    def _iterative_find_P():
        D = load_D()
        
        #try:
        #add_left_perms_grouped = joblib.load("/dev/shm/add_left_perms_grouped.joblib")
        #add_left_perms_grouped_best_ks = joblib.load("/dev/shm/add_left_perms_grouped_best_ks.joblib")
        #import pdb; pdb.set_trace()
        #for key in add_left_perms_grouped.keys():
        #    if len(add_left_perms_grouped[key]) == 0:
        #        import pdb; pdb.set_trace();
        #import pdb; pdb.set_trace()
        #except:
        add_left_perms_grouped = _get_groups(D)
        joblib.dump(add_left_perms_grouped,add_left_perms_grouped_file)
        
        right_perm = add_left_perms_grouped # [(key) for key in add_left_perms_grouped.keys()]
        P = []
        futures = _get_futures([],right_perm)
        seq = as_completed(futures)
        num_completed = 0
        for future in seq:
            if num_completed >= max_search_space:
                print('Reached maximum search space. Stopping.')
                break
            if verbose:
                print("Futures remaining:",seq.count())
                print("Futures completed:",num_completed)
            j = 0
            try:
                for result_check, result_skipped, result_left_perm, result_add_left_perm, result_calc_k_left_perm, result_calc_k_right_perm in future.result():
                    num_completed += 1
                    if result_check:
                        if len(result_calc_k_right_perm) < o or math.factorial(len(result_calc_k_right_perm)) <= t:
                            k, Pnew, search_perms = _exhaustive_search(result_calc_k_left_perm, result_calc_k_right_perm, [])
                            z = 0
                            result_right_perm = generate_right_perm2(right_perm, result_calc_k_left_perm)
                            for i, search_perm in enumerate(search_perms):
                                perm = result_left_perm + result_add_left_perm + search_perm
                                # need to update Pnew at the right location
                                if k == k_optimal:
                                    P.append(perm)
                                elif k < k_optimal:
                                    raise Exception(
                                        "Error! The k value that you found was less than the optimal, which should never happen.")
                        else:
                            result_right_perm = generate_right_perm2(right_perm, result_calc_k_left_perm)
                            #(result_calc_k_left_perm)
                            new_futures = _get_futures(result_left_perm + result_add_left_perm,result_right_perm)
                            for i, new_future in enumerate(new_futures):
                                seq.add(new_future)
                    elif prune_history:
                        if verbose:
                            print('pruned prefix:',result_left_perm+result_add_left_perm)
                    j += 1
            except:
                client.recreate_error_locally(future)
               
        def _find_numbers(perms):
            solution_history = {}
            numbers = []
            Pexpanded = []
            for perm in perms:
                new_perms = [[]]
                numbers.append([])
                for i,item in enumerate(perm):
                    if type(item) == tuple:
                        new_new_perms = []
                        if item not in solution_history:
                            k, Pnew, search_perms = _exhaustive_search([], item, [])
                            solution_history[item] = Pnew
                        else:
                            Pnew = solution_history[item]
                        numbers[-1].append(len(Pnew))
                        if expand:            
                            for perm_item in Pnew:
                                #if type(perm_item) == list:
                                #    perm_item = perm_item[0]
                                for old_perm in new_perms:                                
                                    old_perm = copy.copy(old_perm)
                                    old_perm.extend(perm_item)
                                    new_new_perms.append(old_perm)
                            new_perms = new_new_perms
                    else:
                        break
                # handle remainder
                item = perm[i:]
                if len(item) > 0:
                    numbers[-1].append(1)
                    if expand:            
                        new_new_perms = []
                        for old_perm in new_perms:
                            old_perm = copy.copy(old_perm)
                            old_perm.extend(item)
                            new_new_perms.append(old_perm)
                        new_perms = new_new_perms
                Pexpanded.extend(new_perms)
            return numbers,Pexpanded
            
        desired_number_chunks = 12
        chunk_length = round(len(P)*1./desired_number_chunks)
        if chunk_length == 0:
            chunk_length = len(P)
        perms_chunks = common.chunks(P,chunk_length)
        if verbose:
            print('Finding final numbers')
        futures = client.map(_find_numbers,perms_chunks)
        results = client.gather(futures)
        numbers = []
        Pexpanded = []
        for chunk_numbers,chunk_Pexpanded in results:
            numbers.extend(chunk_numbers)
            Pexpanded.extend(chunk_Pexpanded)
        final_numbers = []
        p = 0
        for row in numbers:
            final_numbers.append(np.prod(row))
        p = np.sum(final_numbers)
        return p, Pexpanded
    
    p, P = _iterative_find_P()
    
    return k_optimal, p, P
