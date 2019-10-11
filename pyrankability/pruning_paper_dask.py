import itertools
import numpy as np
from numpy import ix_
import math
import tempfile
from dask import delayed
from dask.distributed import as_completed
import joblib

from . import bilp
from . import common

import functools
print = functools.partial(print, flush=True)

def reindex(orig_inxs, order_inxs):
    return list(np.array(orig_inxs)[order_inxs])

def find_P(path_D, o, t, bilp_test=False, prune_history=None, check_and_recurse=True, bilp_method="mos2", client=None):
    def load_D():
        return np.genfromtxt(path_D, delimiter=",")

    bilp_res = bilp.bilp(load_D(),method=bilp_method)
    k_optimal = bilp_res[0]
    print("k",k_optimal)

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
        P = [left_perm + opt_perm + right_perm for opt_perm in np.array(
            all_search_perms)[np.where(k_values == k)[0]].tolist()]
        return k, P

    def _calc_k(left_perm, right_perm=[]):
        D = load_D()
        perm = left_perm + right_perm
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
        Dperfect = np.triu(np.ones((D.shape[0], D.shape[0])), 1)
        k = int(np.sum(np.abs(Dperfect-Dperm)))
        return k, Dperm_right
    
    def _check(left_perm,add_left_perm,prev_right_perm):
        if type(add_left_perm) == tuple:
            add_left_perm = list(add_left_perm)
        
        # move the contents of add_left_perm out of prev_right_perm, this wrinkle is done to
        # save memory
        right_perm = list(set(prev_right_perm) - set(add_left_perm))
        
        perm_k, Dperm_right = _calc_k(left_perm+add_left_perm, right_perm=right_perm)
        if perm_k <= k_optimal:
            if bilp_test: # extra bilp test
                kbilp_res = bilp.bilp(Dperm_right,method=bilp_method)
                kbilp = kbilp_res[0]
                if kbilp+perm_k <= k_optimal: # still need to continue
                    return True, 0, add_left_perm, right_perm
            else:
                return True, 0, left_perm, add_left_perm, right_perm
            
        # pruned
        skipped = math.factorial(len(right_perm))
        return False, skipped, left_perm, add_left_perm, right_perm
            
    
    def _get_futures(left_perm,right_perm):
        def _multi_check(add_left_perms):
            results = []
            for add_left_perm in add_left_perms:
                results.append(_check(left_perm,add_left_perm,right_perm))
            return results
        
        add_left_perms = list(itertools.permutations(right_perm, o))
        num_add_left_perms = common.nPr(len(right_perm),o)
        desired_number_chunks = 1000
        chunk_length = round(num_add_left_perms*1./desired_number_chunks)
        if chunk_length == 0:
            chunk_length = len(add_left_perms)
        add_left_perms_chunks = common.chunks(add_left_perms,chunk_length)
        futures = client.map(_multi_check,add_left_perms_chunks)
        return futures
    
    def _iterative_find_P():
        D = load_D()
        right_perm = list(range(D.shape[0]))
        P = []
        skipped = 0
        futures = _get_futures([],right_perm)
        prune_history_record = {}
        seq = as_completed(futures)
        for future in seq:
            print(seq.count())
            for result_check, result_skipped, result_left_perm, result_add_left_perm, result_right_perm in future.result():
                if result_check:
                    if len(result_right_perm) <= o or math.factorial(len(result_right_perm)) <= t:
                        k, Pnew = _exhaustive_search(result_left_perm + result_add_left_perm, result_right_perm, [])
                        if k == k_optimal:
                            P.extend(Pnew)
                        elif k < k_optimal:
                            print(k,k_optimal,Pnew)
                            raise Exception(
                                "Error! The k value that you found was less than the optimal, which should never happen.")
                    else:
                        new_futures = _get_futures(result_left_perm + result_add_left_perm,result_right_perm)
                        for new_future in new_futures:
                            seq.add(new_future)
                #else:
                #    print('pruned')
                #    skipped += result_skipped
                #    if prune_history:
                #        prune_history_record["skipped"] = skipped
                #        prune_history_record[tuple(result_left_perm+result_add_left_perm)] = True
                #        joblib.dump(prune_history_record,"/dev/shm/prune_history_record.joblib.z")

        return P, skipped
    
    P, skipped = _iterative_find_P()
    return k_optimal, P, skipped
