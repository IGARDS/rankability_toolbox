function [k,p,P,stats] = rankability_exhaustive(D,varargin)
%% Usage:
% INPUT:  D = n by n data matrix of dominance information. 
%             Can be unweighted (binary) or weighted. e.g., D(i,j)=1 if i beat j. 
%             The diagonal terms must be 0. i.e., D(i,i) = 0
%             For weighted D matrices, 0 <= D(i,j) <= 1
% OUTPUT: k = minimum number of changes (links added or removed) to
%             transform the input graph to a dominance graph, which can be 
%             reordered to strictly upper triangular form.
% OUTPUT: p = cardinality of set P of rankings (dominance graphs) that can
%             be created if k perturbations are allowed. 
% OUTPUT: P = Set of rankings (dominance graphs) that can
%             be created if k perturbations are allowed. 

% set defaults for optional inputs
optargs = struct('normalize', false);
ix = find(strcmp(varargin,'normalize'),1);
if ~isempty(ix)
    optargs.normalize = varargin{ix+1};
end

test_inxs = find(D ~= 1 & D ~= 0);
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
        fitness(l)=calc_k(Dperm);
    end
end
k=min(fitness);
indexk=find(fitness==k);
p=length(indexk);
P=X(:,indexk);
r = k*p;

stats = struct('r',r);
if optargs.normalize
    kavg = mean(fitness);
    kvalues = unique(fitness);
    pvalues = zeros(1,length(kvalues));
    rvalues = zeros(1,length(kvalues));
    for i = 1:length(kvalues)
        kv = kvalues(i);
        indexkv = find(fitness == kv);
        pvalues(i) = length(indexkv);
        rvalues(i) = kv*length(indexkv);
    end
    pavg = mean(pvalues);
    ravg = mean(rvalues);
    pnorm = p/pavg;
    knorm = k/kavg;
    rnorm = 1-r/ravg;
    stats.pnorm = pnorm;
    stats.knorm = knorm;
    stats.rnorm = rnorm;
end