import networkx as nx
import numpy as np

def get_graph(P,names=None):
    mdg = nx.MultiDiGraph()
    
    dg = nx.DiGraph()
        
    #now add in the perm_i
    edges = {}
    for j in range(len(P)):
        perm_j = P[j,:]
        for i in range(len(perm_j)-1):
            i_t = perm_j[i]
            j_t = perm_j[i+1]
            if (i_t,j_t) not in edges:
                edges[(i_t,j_t)] = 0.0
            edges[(i_t,j_t)] += 1.0
            mdg.add_weighted_edges_from([(i_t,j_t, 1.0)],index=j)
            #colors.append("blue")
    for key in edges.keys():
        weight = edges[key]*1./len(P)
        i_t,j_t = key
        dg.add_weighted_edges_from([(i_t,j_t,weight)])

    #g.es["color"] = colors
    
    return mdg,dg
    
    