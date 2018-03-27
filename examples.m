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
