import math
import numpy as np
import json
import sys

np.set_printoptions(threshold=sys.maxsize)

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)
    #return comb(n,r)
    
def nPr(n,r):
    f = math.factorial
    return f(n) // f(n-r)

def permute_D(D,perm):
    return D[perm,:][:,perm]

def calc_k(D,max_value=None):
    if not max_value:
        max_value = np.max(D)
        if max_value == 0:
            max_value = 1
    perfectRG=np.triu(max_value*np.ones((D.shape[0],D.shape[0])),1).astype(int)
    k = np.sum(np.abs(perfectRG-D))
    return k

def define_D(M,w,min_support):
    n = M.shape[0]
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = 0
            for k in range(len(w)):
                c += w[k]*1.0*int(M[i,k] > M[j,k])
            D[i,j] = int(c > min_support)
            
    return D

def len_chunks(l,n):
    return len(range(0,len(l),n))

def chunks(l, n):
    chunks_ = []
    for i in range(0, len(l), n):
        #yield l[i:i + n]
        chunks_.append(l[i:i + n])
    return chunks_

def chunks_generator(l,n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def as_json(k,P,other={},p=None):
    if len(P) > 0 and np.min(P) == 0:
        P = (np.array(P,dtype=int)+1).tolist()
    if p == None:
        p = len(P)
    return json.dumps({"k":k,"p": p, "P":P,"other":other})

def json_string(k,P,other={},p=None):
    p = len(P)
    if len(P) > 0 and np.min(P) == 0:
        P = (np.array(P,dtype=int)+1).tolist()
    if p == None:
        p = len(P)
    k = int(k)
    indent = "    "
    instance_as_string = "{\n"
    instance_as_string += indent + json.dumps({"k": k}).replace("{","").replace("}","")+",\n"
    instance_as_string += indent + json.dumps({"p": p}).replace("{","").replace("}","")+",\n"
    instance_as_string += indent + '"P": \n'
    P_as_string = np.array2string(np.array(P,dtype=int),separator=",",max_line_width=np.Inf).replace("[","[\n",1)
    lines = P_as_string.split("\n")
    for i,line in enumerate(lines):
        add_indent = indent+indent
        if i > 0:
            add_indent += indent
        instance_as_string += add_indent + line + "\n"
    instance_as_string = instance_as_string[:-2] + "],\n"
    instance_as_string += indent + json.dumps({"other": other})[1:-1]+"\n"
    instance_as_string += "}"
    return instance_as_string
    
        
class Search:
    def to_json(self):
        solution = {}
        solution["k"] = int(self.k)
        solution["p"] = len(self.P)
        solution["P"] = [[int(v) + 1 for v in perm] for perm in self.P]
        solution["other"] = self.other_to_dict()
        return solution
    
    # Override this if the search subclass does not keep track of thse things
    def other_to_dict(self):
        return {"skipped": self.skipped, "searched": self.searched, "exact_solution_found": self.exact_solution_found}
        
