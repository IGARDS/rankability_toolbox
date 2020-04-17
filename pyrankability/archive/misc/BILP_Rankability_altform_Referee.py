

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
A: agjacency matrix of the graph G with the source node(0)
n: the number of nodes of G
m: the number of directed-edges of G
"""
D = np.array([[0,1,1,0,0,1],
                [0,0,1,0,0,0],
                [0,1,0,0,0,0],
                [1,0,0,0,0,1],
                [1,0,1,0,0,0],
                [0,1,1,0,1,0]])

n = len(A[0])
m = int(np.sum(A))

print("n = {0:}".format(n))
print("m = {0:}".format(m))

AP = Model("rankabilityAltFormReferee”)
z = {}
for i in range(n):
    for j in range(n):
        z[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name=“z(%s,%s)”%(i,j)) #binary
AP.update()


# In[59]:


for j in range(1,n):
    for i in range(0,j):
        AP.addConstr(z[i,j] + z[j,i] == 1)

for i in range(n):
    for j in range(n):
        for k in range(n):
            if j != i and k != j and k != i:
                AP.addConstr(z[i,j] + z[j,k] + z[k,i] <= 2)
                

AP.update()

AP.setObjective(quicksum(D[i,j]*z[i,j] for i in range(n) for j in range(n)),GRB.MAXIMIZE)
AP.update()

AP.write('rankabilityAltFormReferee.bilp')

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
#   Rule for Y: If z[i,j]=0 and D[i,j]=1, then make y[i,j]=1.
#   Rule for X: If z[i,j]=1 and D[i,j]=0, then make x[i,j]=1.
#   Then set k=nnz(X)+nnz(Y).

#########################################################





# In[54]:


dir(z[0,0].Xn)

