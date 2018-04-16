function k = calc_k(Dperm,max_value)
if ~exist('max_value')
    max_value = max(max(Dperm)) - min(min(Dperm));
end
perfectRG=triu(max_value*ones(size(Dperm,1)),1);
k = sum(sum(abs(perfectRG-Dperm)));

% % Find any 0's in the upper triangular section and add those to the k
% Dperm_triu = ceil(triu(Dperm));
% k = length(find(Dperm_triu == 0));
% 
% % Find any in the lower that can be removed and remove them based on their
% % weight
% Dperm_tril = tril(Dperm);
% %inxs = find(Dperm_tril ~= 0);
% k = k + sum(sum(Dperm_tril));
