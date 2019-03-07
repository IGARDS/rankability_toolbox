import numpy as np

from . import lp

def greedy(D,l,verbose=False):
    output_lines = []
    D = np.copy(D) # Leave the original untouched
    for niter in range(l):
        n=D.shape[0]
        
        k,P,X,Y,k2 = lp.lp(D)

        mult = 100
        X = np.round(X*mult)/mult
        Y = np.round(Y*mult)/mult

        T0 = np.zeros((n,n))
        T1 = np.zeros((n,n))
        inxs = np.where(D + D.transpose() == 0)
        T0[inxs] = 1
        inxs = np.where(D + D.transpose() == 2)
        T1[inxs] = 1
        T0[np.arange(n),np.arange(n)]= 0
        T1[np.arange(n),np.arange(n)] = 0

        DOM = D + X - Y

        Madd=T0*DOM # note: DOM = P_> in paper
        M1 = Madd # Copy Madd into M, % Madd identifies values >0 in P_> that have 0-tied values in D
        M1[Madd<=0] = np.nan # Set anything <= 0 to NaN
        min_inx = np.nanargmin(M1) # Find min value and index
        bestlinktoadd_i, bestlinktoadd_j = np.unravel_index(min_inx,M1.shape) # adding (i,j) link associated with
        # smallest nonzero value in Madd is likely to produce greatest improvement in rankability
        minMadd = M1[bestlinktoadd_i, bestlinktoadd_j]

        Mdelete=T1*DOM # note: DOM = P_> in paper
        Mdelete=Mdelete*(Mdelete<1) # Mdelete identifies values <1 in P_> that have 1-tied values in D
        bestlinktodelete_i, bestlinktodelete_j=np.unravel_index(np.nanargmax(Mdelete), Mdelete.shape) # deleting (i,j) link associated with
        # largest non-unit (less than 1) value in Mdelete is likely to produce greatest improvement in rankability
        maxMdelete = Mdelete[bestlinktodelete_i, bestlinktodelete_j]

        # This next section modifies D to create Dtilde
        Dtilde = np.copy(D) # initialize Dtilde
        # choose whether to add or remove a link depending on which will have the biggest
        # impact on reducing the size of the set P
        # PAUL: Or if we only want to do link addition, you don't need to form
        # Mdelete and find the largest non-unit value in it. And vice versa, if
        # only link removal is desired, don't form Madd.
        if (1-minMadd)>maxMdelete:
            Dtilde[bestlinktoadd_i,bestlinktoadd_j]=1 # adds this link, creating one-mod Dtilde
            formatSpec = 'The best one-link way to improve rankability is by adding a link from %d to %d.\nThis one modification removes about %.10f percent of the rankings in P.'%(bestlinktoadd_i,bestlinktoadd_j,(1-minMadd)*100)
            if verbose:
                print(formatSpec)
            output_lines.append(formatSpec)
        elif 1-minMadd<maxMdelete:
            Dtilde[bestlinktodelete_i,bestlinktodelete_j] = 0 # removes this link, creating one-mod Dtilde
            formatSpec = 'The best one-link way to improve rankability is by deleting the link from %d to %d.\nThis one modification removes about %.10f percent of the rankings in P.' % (bestlinktodelete_i,bestlinktodelete_j,maxMdelete*100)
            if verbose:
                print(formatSpec)
            output_lines.append(formatSpec)
        
        D = Dtilde
    return D, "\n".join(output_lines)