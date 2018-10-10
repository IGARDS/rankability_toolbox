% List of examples in no particular order

% Unweighted D matrix
fprintf('Loading the unweighted example data\n');
D = csvread('data/unweighted1.csv');

% non-parallel exhaustive method
[k,p,P] = rankability_exhaustive(D);
fprintf('Exhaustive k=%f\n',k);

[k,p,P] = rankability_exhaustive_parallel(D,4);
fprintf('Exhaustive Parallel k=%f\n',k);

% lp method
[k,p,X,Y] = rankability_lp(D);
fprintf('LP k=%f\n',k);

% Weighted
fprintf('Loading the weighted example data\n');
D = csvread('data/weighted1.csv');

% non-parallel exhaustive method
[k,p,P] = rankability_exhaustive(D);
fprintf('Exhaustive k=%f\n',k);

[k,p,P] = rankability_exhaustive_parallel(D,4);
fprintf('Exhaustive Parallel k=%f\n',k);

% lp method
[k,p,X,Y] = rankability_lp(D);
fprintf('LP k=%f\n',k);

% normalization
D=[0 1 1 1 1 1 1 1; 0 0 1 1 1 1 1 1; 0 0 0 1 1 1 1 1; 0 0 0 0 1 1 1 1; 0 0 0 0 0 1 1 1; 0 0 0 0 0 0 1 1; ; 0 0 0 0 0 0 0 1; ; 0 0 0 0 0 0 0 0];
[k,p,P,stats] = rankability_exhaustive(D,'normalize',true);
%[k,p,P,stats] = rankability_exhaustive_parallel(D,100,'normalize',true);
stats

D=ones(7,7);
[k,p,P,stats] = rankability_exhaustive(D,'normalize',true);
stats

ntimes = 100;
rnorm = zeros(1,ntimes);
for j = 1:ntimes
    D=round(rand(7,7));
    for i = 1:size(D,1)
        D(i,i) = 0;
    end
    [k,p,P,stats] = rankability_exhaustive(D,'normalize',true);
    rnorm(j) = stats.rnorm;
end
hist(rnorm);

D=zeros(7,7);
for i = 1:size(D,1)-1
    D(i,i+1) = 1;
end
[k,p,P,stats] = rankability_exhaustive(D,'normalize',true);
stats
