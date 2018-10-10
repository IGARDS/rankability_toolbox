

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 25

@author: amy langville
"""

#---packages---#
import numpy as np
from gurobipy import *
#--------------#


"""
#---read data---#
A_org: adjacency matrix of the original graph
A: adjacency matrix of the graph G with the source node(0)
n: the number of nodes of G
m: the number of directed-edges of G
"""
D = np.array([[0,1,1,0,0,1],
                [0,0,1,0,0,0],
                [0,1,0,0,0,0],
                [1,0,0,0,0,1],
                [1,0,1,0,0,0],
                [0,1,1,0,1,0]])


# NEED PAUL’S HELP HERE TO CREATE K matrix from D matrix
#########################################################
# K matrix is a strictly upper triangular matrix (0s in lower triangular part)
#              that tells the Type (Type 1 or Type 2) for each variable.
#   K[i,j]=1 if [i,j]-element is a Type 1 variable which means D[i,j] and D[j,i] are
#              symmetric (both have the same value, either both are 0 or both are 1). This means
#              exactly one of these two elements must be changed, the tie between i and j must be 
#              broken.
#   K[i,j]=2 if [i,j]-element is a Type 2 variable which means D[i,j] and D[j,i] are
#              anti-symmetric (i.e, one has value 0 and the other has value 1). 
#              This means either the dominance ordering between these two elements stays the same
#              or it flips, meaning there are two changes.
#   In Matlab, the following command creates the K matrix from the D matrix.
#      K=2*triu(ones(n,n),1)-triu(~(triu(D,1)-(tril(D,-1))'),1)
#########################################################   

n = len(A[0])
m = int(np.sum(A))

print("n = {0:}".format(n))
print("m = {0:}".format(m))

AP = Model("rankabilityAltFormMOS2”)
z = {}
for i in range(n):
    for j in range(n):
        z[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name=“z(%s,%s)”%(i,j)) #binary
AP.update()


# In[59]:


for j in range(1,n):
    for i in range(0,j):
        AP.addConstr(z[i,j] + (3-2*K[i,j])*z[j,i] == 2-K[i,j])

for i in range(n):
    for j in range(n):
        for k in range(n):
            if j != i and k != j and k != i:
                AP.addConstr((1-2*D[i,j])*z[i,j] + (1-2*D[j,k])*z[j,k] + (1-2*D[k,i])*z[k,i] <= 2-D[i,j]-D[j,k]-D[k,i])
                

AP.update()

AP.setObjective(quicksum((2*K[i,j]-1)*z[i,j] for i in range(n) for j in range(n)),GRB.MINIMIZE)
AP.update()

AP.write('rankabilityAltFormMOS2.bilp')

# Limit how many solutions to collect
model.setParam(GRB.Param.PoolSolutions, 1024)
# do a systematic search for the k-best solutions
model.setParam(GRB.Param.PoolSearchMode, 2)
AP.update()

AP.optimize()

print(AP.Status)

# Print best selected set
    print('Selected elements in best solution:')
    print('\t', end='')
    for e in Groundset:
        if Elem[e].X > .9:
            print(' El%d' % e, end='')
    print('')

# Print number of solutions stored
nSolutions = model.SolCount
print('Number of solutions found: ' + str(nSolutions))

# Print objective values of solutions
    for e in range(nSolutions):
        model.setParam(GRB.Param.SolutionNumber, e)
        print('%g ' % model.PoolObjVal, end='')
        if e % 15 == 14:
            print('')
    print('')


# In[61]:


sol_z = [round(100*z[i,j].X)/100 for i in range(n) for j in range(n)]
sol_z = np.reshape(sol_z,(n,n))
print(sol_z)


#########################################################
# NEED PAUL’S HELP HERE TO TRANSFORM z[i,j] solution into 
#   X and Y matrices in order to compare with original formulation.
#   First, initialize X and Y matrices as all zeros, then follow rules below.
#   Rule for Y: If z[i,j]=1 and D[i,j]=1, then make y[i,j]=1.
#   Rule for X: If z[i,j]=1 and D[i,j]=0, then make x[i,j]=1.
#   Then set k=n1+objectivevalue, where n1=number of 1s in K (in Matlab, n1=nnz(find(K==1))  )


#########################################################




# In[54]:


dir(z[0,0].Xn)

