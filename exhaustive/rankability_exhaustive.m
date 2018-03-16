function [k,p,P] = brute_force(D)
% INPUT:  D = n by n data matrix of dominance information, uses only 
%             zeros and ones; D(i,j)=1 if i beat j
% OUTPUT: k = minimum number of changes (links added or removed) to
%             transform the input graph to a dominance graph, which can be 
%             reordered to strictly upper triangular form.
% OUTPUT: p = cardinality of set P of rankings (dominance graphs) that can
%             be created if k perturbations are allowed. NOTE: For most
%             problems P will be a partial set of rankings, not the full
%             set of rankings.

n = size(D,1);
X=perms(1:n);
X=X';
fitness = zeros(1,size(X,2));
for l=1:size(X,2)
   perm=X(:,l);
   fitness(l)=nnz(tril(D(perm,perm)))+(n*(n-1)/2 - nnz(triu(D(perm,perm))));
end
k=min(fitness);
indexk=find(fitness==k);
p=length(indexk);
P=X(:,indexk);