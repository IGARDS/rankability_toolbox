addpath('exhaustive')
addpath('lp')

% Unweighted D matrix
D = csvread('data/unweighted1.csv');
true_k = 9;
true_p = 12;

% non-parallel exhaustive method
[k,p] = rankability_exhaustive(D,'transform',true);
if k ~= true_k || p ~= true_p
    fprintf('Test 1 failed for rankability_exhaustive!\n');
else
    fprintf('Test 1 passed for rankability_exhaustive!\n');
end

% parallel exhaustive method
[k,p,P,stats] = rankability_exhaustive_parallel(D,10,'transform',true);
if k ~= true_k || p ~= true_p
    fprintf('Test 1 failed for rankability_exhaustive_parallel!\n');
else
    fprintf('Test 1 passed for rankability_exhaustive_parallel!\n');
end

% lp method
k = rankability_lp(D);
if k ~= true_k
    fprintf('Test 1 failed for rankability_lp (p ignored)!\n');
else
    fprintf('Test 1 passed for rankability_lp (p ignored)!\n');
end

% Weighted D matrix
D = csvread('data/weighted1.csv');
true_k = 1523;
true_p = 2;

% non-parallel exhaustive method
[k,p] = rankability_exhaustive(D,'transform',true);
if k ~= true_k || p ~= true_p
    fprintf('Test 1 failed for rankability_exhaustive!\n');
else
    fprintf('Test 1 passed for rankability_exhaustive!\n');
end

% parallel exhaustive method
[k,p] = rankability_exhaustive_parallel(D,10,'transform',true);
if k ~= true_k || p ~= true_p
    fprintf('Test 1 failed for rankability_exhaustive_parallel!\n');
else
    fprintf('Test 1 passed for rankability_exhaustive_parallel!\n');
end

% lp method
k = rankability_lp(D);
if k ~= true_k
    fprintf('Test 1 failed for rankability_lp (p ignored)!\n');
else
    fprintf('Test 1 passed for rankability_lp (p ignored)!\n');
end
