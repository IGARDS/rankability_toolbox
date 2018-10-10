import itertools
import numpy as np
from numpy import ix_
import math
import tempfile
from dask import delayed
from dask.distributed import as_completed
from sklearn.externals import joblib
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

def find_P(path_D, o, t, bilp_test=False, bilp_method="mos2", client=None, max_search_space=100000, prune_history=False, uuid1=""):
    def load_D():
        return np.genfromtxt(path_D, delimiter=",")
    orig_right_perm = list(range(load_D().shape[0]))
    bilp_res = bilp.bilp(load_D(),method=bilp_method)
    k_optimal = bilp_res[0]
    print("k",k_optimal)
    add_left_perms_grouped_file = "/dev/shm/add_left_perms_grouped"+uuid1+".joblib"
    add_left_perms_grouped_best_ks_file = "/dev/shm/add_left_perms_grouped_best_ks"+uuid1+".joblib"
    print(add_left_perms_grouped_file,add_left_perms_grouped_best_ks_file)
        
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
    
    def _check(left_perm,add_left_perm,add_left_perms_grouped):
        def _convert(perm):
            actual_perm = []
            for key in perm:
                actual_perm += add_left_perms_grouped[key][0]
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
        #right_perm = []
        #partial_add_left_perm = set(add_left_perm[0])
        #for key in prev_right_perm:
        #    if partial_add_left_perm.isdisjoint(key):
        #        right_perm.append(key)
            
        perm_k, Dperm_right = _calc_k(calc_k_left_perm, right_perm=calc_k_right_perm)
        if perm_k <= k_optimal:
            if bilp_test: # extra bilp test
                kbilp_res = bilp.bilp(Dperm_right,method=bilp_method)
                kbilp = kbilp_res[0]
                if kbilp+perm_k <= k_optimal: # still need to continue
                    return True, 0, left_perm, add_left_perm, calc_k_left_perm, calc_k_right_perm
            else:
                return True, 0, left_perm, add_left_perm, calc_k_left_perm, calc_k_right_perm
            
        # pruned
        if prune_history:
            skipped = math.factorial(len(calc_k_right_perm))
        else:
            skipped = 0
        return False, skipped, left_perm, add_left_perm, calc_k_left_perm, calc_k_right_perm
    
    def _get_groups(D):
        def _no_left_calc_k(add_left_perms):
            chunk_results = [_calc_k(list(add_left_perm)) for add_left_perm in add_left_perms]
            i = 0
            add_left_perms_grouped = {}
            add_left_perms_grouped_best_ks = {}
            for kvalue,Dperm_right in chunk_results:
                sorted_items = sorted(add_left_perms[i])
                key = tuple(sorted_items)
                # initialize
                if key not in add_left_perms_grouped_best_ks:
                    add_left_perms_grouped_best_ks[key] = kvalue
                    add_left_perms_grouped[key] = []
                # re-initialize if we found a smaller k
                elif kvalue < add_left_perms_grouped_best_ks[key]:
                    add_left_perms_grouped_best_ks[key] = kvalue
                    add_left_perms_grouped[key] = []
                    
                # one of the best we've so far, so add to the group
                if kvalue == add_left_perms_grouped_best_ks[key]:
                    add_left_perms_grouped[key].append(add_left_perms[i])
                i+=1
            return add_left_perms_grouped, add_left_perms_grouped_best_ks
        
        add_left_perms = list(itertools.permutations(range(D.shape[0]), o))
        num_add_left_perms = len(add_left_perms)
        desired_number_chunks = 16
        chunk_length = round(num_add_left_perms*1./desired_number_chunks)
        add_left_perms_chunks = common.chunks(add_left_perms,chunk_length)
        futures = client.map(_no_left_calc_k,add_left_perms_chunks)
        results = client.gather(futures)
        add_left_perms_grouped = {}
        add_left_perms_grouped_best_ks = {}
        for chunk_add_left_perms_grouped, chunk_add_left_perms_grouped_best_ks in results:
            for key in chunk_add_left_perms_grouped.keys():
                if key not in add_left_perms_grouped or chunk_add_left_perms_grouped_best_ks[key] < add_left_perms_grouped_best_ks[key]:
                    add_left_perms_grouped[key] = chunk_add_left_perms_grouped[key]
                    add_left_perms_grouped_best_ks[key] = chunk_add_left_perms_grouped_best_ks[key]
                elif chunk_add_left_perms_grouped_best_ks[key] == add_left_perms_grouped_best_ks[key]:
                    add_left_perms_grouped[key].extend(chunk_add_left_perms_grouped[key])
        return add_left_perms_grouped, add_left_perms_grouped_best_ks
    
    def _multi_check(add_left_perms,left_perm):
        results = []
        add_left_perms_grouped = joblib.load(add_left_perms_grouped_file)
        for add_left_perm in add_left_perms:
            results.append(_check(left_perm,add_left_perm,add_left_perms_grouped))
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
        add_left_perms_grouped, add_left_perms_grouped_best_ks = _get_groups(D)
        print('1')
        joblib.dump(add_left_perms_grouped,add_left_perms_grouped_file)
        joblib.dump(add_left_perms_grouped_best_ks,add_left_perms_grouped_best_ks_file)
        print(2)
        
        right_perm = [(key) for key in add_left_perms_grouped.keys()]
        print(3)
        P = []
        futures = _get_futures([],right_perm)
        print(4)
        seq = as_completed(futures)
        num_completed = 0
        for future in seq:
            if num_completed >= max_search_space:
                break
            print("Futures remaining:",seq.count())
            print("Futures completed:",num_completed)
            j = 0
            try:
                for result_check, result_skipped, result_left_perm, result_add_left_perm, result_calc_k_left_perm, result_calc_k_right_perm in future.result():
                    num_completed += 1
                    if result_check:
                        if len(result_calc_k_right_perm) <= o or math.factorial(len(result_calc_k_right_perm)) <= t:
                            k, Pnew, search_perms = _exhaustive_search(result_calc_k_left_perm, result_calc_k_right_perm, [])
                            z = 0
                            result_right_perm = generate_right_perm(result_calc_k_left_perm)
                            for i, search_perm in enumerate(search_perms):
                                perm = result_left_perm + result_add_left_perm + result_right_perm + search_perm
                                # need to update Pnew at the right location
                                if k == k_optimal:
                                    P.append(perm)
                                else:
                                    raise Exception(
                                        "Error! The k value that you found was less than the optimal, which should never happen.")
                        else:
                            result_right_perm = generate_right_perm(result_calc_k_left_perm)
                            new_futures = _get_futures(result_left_perm + result_add_left_perm,result_right_perm)
                            for i, new_future in enumerate(new_futures):
                                seq.add(new_future)
                    elif prune_history:
                        print('pruned:',result_skipped)
                    j += 1
            except:
                client.recreate_error_locally(future)
               
        Pexpanded = []
        for perm in P:
            new_perms = [[]]
            for i,item in enumerate(perm):
                if type(item) == tuple:
                    new_new_perms = []
                    for perm_item in add_left_perms_grouped[item]:
                        if type(perm_item) == list:
                            perm_item = perm_item[0]
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
                new_new_perms = []
                for old_perm in new_perms:
                    old_perm = copy.copy(old_perm)
                    old_perm.extend(item)
                    new_new_perms.append(old_perm)
                new_perms = new_new_perms
            Pexpanded.extend(new_perms)

        return Pexpanded
    
    P = _iterative_find_P()
    # cleanup
    try:
        os.remove(add_left_perms_grouped_best_ks_file)
    except:
        pass
    try:
        os.remove(add_left_perms_grouped_file)
    except:
        pass
    
    return k_optimal, P
