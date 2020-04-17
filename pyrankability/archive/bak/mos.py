class MOSSearch(Search):
    def __init__(self,D):
        self.D = D
        bilp_res = bilp(self.D)
        self.k = int(bilp_res[0])
        self.n = int(D.shape[0])
        self.solution_size = self.n**2-self.n
        self.search_space_size = nCr(self.solution_size,self.k)
        self.max_leave_out = self.solution_size - self.k
        
    def print_solution_space_summary(self):
        print('Total search space of the problem:',self.search_space_size)
           
    def prepare_iterators(self,print_summary=False):
        if print_summary:
            self.print_solution_space_summary()
        
        # Now create the iterators that can run in parallel        
        self.solutions_iter = itertools.combinations(range(self.solution_size),self.k)
 
    def find_P(self):
        k = np.Inf
        P = []
        n = self.n
        D = self.D
        for j,solution in enumerate(self.solutions_iter):
            Z=np.zeros((n,n)) # For big n, it would help to have all matrices (D, X, Y, Z, DXY) in sparse format.
            X=np.zeros((n,n))
            Y=np.zeros((n,n))
            for b in range(self.k):
                index = solution[b]+1 # start index at 1
                i=int(np.ceil(index/(self.n-1)))  # row index i for bth element with value 1 of ath Z matrix
                rem1=int(np.remainder(index,(self.n-1)))
                # 4 cases for column index j
                if rem1 >= i: # case 1
                    j=rem1+1
                if rem1>0 and rem1<i: # case 2
                    j=rem1
                if rem1==0 and i<n: # case 3
                    j=n
                if rem1==0 and i==n: # case 4
                    j=n-1
                Z[i-1,j-1]=1
                if D[i-1,j-1]==0:
                    X[i-1,j-1]=1
                if D[i-1,j-1]==1:
                    Y[i-1,j-1]=1
                
                DXY=np.remainder(D+Z,2)
                sumDXY=np.sum(DXY,1)
                ranking = np.argsort(-1*sumDXY)
                descendsort = sumDXY[ranking]
                sumDXY=np.sum(DXY,0)
                ranking = np.argsort(sumDXY)
                ascendsort = sumDXY[ranking]
                if sum(abs(descendsort-np.arange(self.n-1,-1,-1)))==0 and sum(abs(ascendsort-np.arange(0,n)))==0:
                    sol_k = calc_k(permute_D(D,ranking))
                    if sol_k < k:
                        k = sol_k
                        P = []
                    if sol_k == k:
                        P.append(tuple(np.array(ranking)))
        self.k = int(k)
        self.P = P