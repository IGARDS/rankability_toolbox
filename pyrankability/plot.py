import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from pylab import rcParams

# Given something like:
# A = [4, 10, 1, 12, 3, 9, 0, 6, 5, 11, 2, 8, 7]
# B = [5, 4, 10, 1, 7, 6, 12, 3, 9, 0, 11, 2, 8]
def AB_to_P2(A,B):
    P2 = pd.DataFrame(np.array([A,B]))
    return P2

def spider(P2,file=None,fig_format="PNG",width=5,height=10):
    """
    from pyrankability.plot import spider, AB_to_P2

    A = [4, 10, 1, 12, 3, 9, 0, 6, 5, 11, 2, 8, 7]
    B = [5, 4, 10, 1, 7, 6, 12, 3, 9, 0, 11, 2, 8]
    spider(AB_to_P2(A,B))
    """
    rcParams['figure.figsize'] = width, height

    G = nx.Graph()

    pos = {}
    buffer = 0.25
    step = (2-2*buffer)/P2.shape[1]
    labels={}
    y1 = []
    y2 = []
    y = []
    index = []
    for i in range(P2.shape[1]):
        name1 = "A%d:%d"%(i+1,P2.iloc[0,i])
        name2 = "B%d:%d"%(i+1,P2.iloc[1,i])
        G.add_node(name1)
        G.add_node(name2)
        loc = 1-buffer-(i*step)
        pos[name1] = np.array([-1,loc])
        pos[name2] = np.array([1,loc])
        labels[name1] = P2.iloc[0,i]
        labels[name2] = P2.iloc[1,i]
        y1.append(name1)
        y2.append(name2)
        y.append("A")
        y.append("B")
        index.append(name1)
        index.append(name2)
    y=pd.Series(y,index=index)

    for i in range(P2.shape[1]):
        name1 = "A%d:%d"%(i+1,P2.iloc[0,i])
        ix = np.where(P2.iloc[1,:] == P2.iloc[0,i])[0]
        name2 = "B%d:%d"%(ix+1,P2.iloc[0,i])
        G.add_edge(name1, name2)
    edges = G.edges()

    nx.draw_networkx_labels(G,pos=pos,labels=labels)

    color_map = y.map({"A":"blue","B":"red"})
    nx.draw(G, pos, edges=edges, node_color=color_map)
    if file is not None:
        plt.savefig(file, fig_format="PNG")
    plt.show()