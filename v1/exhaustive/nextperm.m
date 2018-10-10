function y = nextperm(x,n)
%   NEXTPERM next permutation
%   NEXTPERM(X,N) provides the next permutation of the ordered tuple X,
%   using lexical order, on the set {1, 2, 3, ..., N-1}. For example
%   nextperm([1 5 4], 5) returns [2 1 3]
%   nextperm([2 1 3], 5) returns [2 1 4]
%   nextperm([5 4 3], 5) returns [5 4 3]

d = length(x);
i = d;
u = 1:n;
while i > 0
    r = setdiff(u,x(1:i));
    next = r(r>x(i));
    if isempty(next)
        i = i - 1;
    else
       x(i) = min(next);
       if i < d
        r = setdiff(u,x(1:i));
        x((i+1):d) = r(1:(d-i));
       end
       i = 0;
    end
end
y = x;