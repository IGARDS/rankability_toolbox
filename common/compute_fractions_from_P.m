function [counts,F] = compute_fractions_from_P(P)
%% Computes the number of times and the fractional equivalent of how many
%  times item j shows up in position i.
% counts(i,j) = number of times that item j shows up at position i
counts = zeros(size(P,1),size(P,1));
for i = 1:size(P,2)
    perm = P(:,i);
    for j = 1:length(perm)
        counts(j,perm(j)) = counts(j,perm(j))+1;
    end
end
F = counts/size(P,2);
