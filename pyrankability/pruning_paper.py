import itertools
import numpy as np
from numpy import ix_
from joblib import Parallel, delayed
import math
from joblib import parallel_backend
import tempfile

from . import bilp
from . import common

def reindex(orig_inxs, order_inxs):
    return list(np.array(orig_inxs)[order_inxs])

def find_k(D,bilp_method="mos2"):
    bilp_res = bilp.bilp(D,method=bilp_method)
    k_optimal = bilp_res[0]
    return k_optimal

def find_P(D, o, t, bilp_test=True, prune_history=None, check_and_recurse=True, bilp_method="mos2"):
    bilp_res = bilp.bilp(D,method=bilp_method)
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

    def _check_and_recurse(left_perm, add_left_perm, prev_right_perm):
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
                    return _find_P(left_perm+add_left_perm, right_perm)
            else:
                return _find_P(left_perm+add_left_perm, right_perm)
            
        # pruned
        skipped = math.factorial(len(right_perm))
        #if prune_history:
        #    skipped_perm = left_perm + add_left_perm
        #    print("prune",skipped_perm)
        return [], skipped
    
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
                return True, 0, add_left_perm, right_perm
            
        # pruned
        skipped = math.factorial(len(right_perm))
        #if prune_history:
        #    skipped_perm = left_perm + add_left_perm
        #    print("prune",skipped_perm)
        return False, skipped, add_left_perm, right_perm

    def _find_P(left_perm, right_perm):
        if len(right_perm) <= o or math.factorial(len(right_perm)) <= t:
            k, P = _exhaustive_search(left_perm, right_perm, [])
            # This could be collapsed, but I like to include it as a check of assumptions
            if k > k_optimal:
                return [], 0
            elif k == k_optimal:
                return P, 0
            else:
                raise Exception(
                    "Error! The k value that you found was less than the optimal, which should never happen.")

        # Recursive step
        skipped = 0
        if check_and_recurse: # Depth
            add_left_perms = [
                list(perm) for perm in itertools.permutations(list(right_perm), o)]
            right_perms = [list(set(right_perm) - set(add_left_perm))
                           for add_left_perm in add_left_perms]
            results = Parallel()(delayed(_check_and_recurse)(left_perm, add_left_perm, right_perm) for add_left_perm in add_left_perms)
        else: # Breadth
            add_left_perms = itertools.permutations(right_perm, o)
            num_add_left_perms = common.nPr(len(right_perm),o)
            results = Parallel()(delayed(_check_and_recurse)(left_perm, add_left_perm, right_perm) for add_left_perm in add_left_perms)
            results = map(_check,itertools.repeat(left_perm, num_add_left_perms), add_left_perms, itertools.repeat(right_perm, num_add_left_perms))
            not_skipped_add_left_perms = []
            not_skipped_right_perm = []
            i = 0                
            if prune_history:
                summary = {}
            for result_check,result_skipped, result_add_left_perm, result_right_perm in results:
                skipped += result_skipped
                if result_check:
                    not_skipped_add_left_perms.append(result_add_left_perm)
                    not_skipped_right_perm.append(result_right_perm)
                    # fill in a summary of what was not pruned
                    if prune_history:
                        for item in result_add_left_perm:
                            if item not in summary:
                                summary[item] = 0
                            summary[item] += 1
                i+=1
            if prune_history:
                print(summary)
                for item in right_perm:
                    if item not in summary:
                        print("prune",left_perm, -item, right_perm)
            results = Parallel()(delayed(_find_P)
                                 (left_perm + not_skipped_add_left_perms[i], not_skipped_right_perm[i]) for i in range(len(not_skipped_add_left_perms)))
        P = []
        for result_P, result_skipped in results:
            skipped += result_skipped
            P.extend(result_P)

        return P, skipped
    
    with parallel_backend("loky",n_jobs=14):
        P, skipped = _find_P([], list(range(D.shape[0])))
        return k_optimal, P, skipped
