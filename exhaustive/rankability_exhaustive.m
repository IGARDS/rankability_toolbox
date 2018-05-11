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

proceed = check_d(D);
if ~proceed
    msgID = 'rankability:invalid_D';
    msg = 'D matrix is invalid. Make sure it is a square matrix and contains only integers.';
    baseException = MException(msgID,msg);
    throw(baseException)
end

% set defaults for optional inputs
optargs = struct('transform', false);
ix = find(strcmp(varargin,'transform'),1);
if ~isempty(ix)
    optargs.transform = varargin{ix+1};
end

n = size(D,1);
X=perms(1:n);
X=X';
fitness = zeros(1,size(X,2));
for l=1:size(X,2)
    perm=X(:,l);
    Dperm = D(perm,perm);
    fitness(l)=calc_k(Dperm);
end
k=min(fitness);
indexk=find(fitness==k);
p=length(indexk);
P=X(:,indexk);

if ~exist('max_value')
    max_value = max(max(D));
end
kmax = max_value*(n^2-n)/2;
r = 1-k*p/(kmax*factorial(n));

stats = struct('r',r);
if optargs.transform
    stats = compute_rtransformed(k,p,P,kmax,stats);
end
