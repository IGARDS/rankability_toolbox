function k = fitness(Dperm)
% Find any 0's in the upper triangular section and add those to the k
Dperm_triu = ceil(triu(Dperm));
k = length(find(Dperm_triu == 0));

% Find any in the lower that to remove and remove them based on their
% weight
Dperm_tril = tril(Dperm);
inxs = length(find(Dperm_tril ~= 0));
k = k + Dperm_tril(inxs);
