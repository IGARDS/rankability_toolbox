function [k,p,P] = rankability_exhaustive(D)
% INPUT:  D = n by n data matrix of dominance information, uses only 
%             zeros and ones; D(i,j)=1 if i beat j
% OUTPUT: k = minimum number of changes (links added or removed) to
%             transform the input graph to a dominance graph, which can be 
%             reordered to strictly upper triangular form.
% OUTPUT: p = cardinality of set P of rankings (dominance graphs) that can
%             be created if k perturbations are allowed. NOTE: For most
%             problems P will be a partial set of rankings, not the full
%             set of rankings.

test_inxs = find(D < 1 & D > 0);
unweighted = true;
if ~isempty(test_inxs) % for unweighted graph 
    unweighted = false;
end

n = size(D,1);
X=perms(1:n);
X=X';
fitness = zeros(1,size(X,2));
perfectRG=triu(ones(size(D,1)),1);
for l=1:size(X,2)
    perm=X(:,l);
    Dperm = D(perm,perm);
    if unweighted
        fitness(l)=sum(sum(abs(perfectRG-Dperm)));
    else
        Dperm_triu = ceil(triu(Dperm));
        Dperm_tril = tril(Dperm);
        Dperm = Dperm_triu+Dperm_tril;
        fitness(l)=sum(sum(abs(perfectRG-Dperm)));
    end
end
k=min(fitness);
indexk=find(fitness==k);
p=length(indexk);
P=X(:,indexk);