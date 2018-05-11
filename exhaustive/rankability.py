import numpy as np
import itertools

def calc_k(D,max_value=None):
    if not max_value:
        max_value = np.max(D)
    perfectRG=np.triu(max_value*np.ones((D.shape[0],D.shape[0])),1).astype(int)
    k = np.sum(np.abs(perfectRG-D))
    return k

def exhaustive(D,max_value=None):
    best_k = np.Inf
    best_P = []
    for perm in itertools.permutations(range(D.shape[0])):
        k = calc_k(D[perm,:][:,perm],max_value=max_value)
        if k < best_k:
            best_k = k
            best_P = [perm]
        elif k == best_k:
            best_P.append(perm)
    return int(best_k), len(best_P), best_P
    
def exhaustive_index_1(D,max_value=None):
    k,p,P = exhaustive(D,max_value=max_value)
    newP = [[int(item) + 1 for item in entry] for entry in P]
    return k,p,newP

if __name__ == "__main__":
    # Random Graph with Answer: k = 9, |P|=12
    D = np.array([[0,1,1,0,0,1],
                [0,0,1,1,0,0],
                [0,1,0,0,0,0],
                [1,0,0,0,0,1],
                [1,0,0,0,0,0],
                [0,1,1,0,1,0]])
    
    k, p, P = exhaustive(D)
    print(k,p,P)