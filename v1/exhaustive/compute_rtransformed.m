function stats = compute_rtransformed(k,p,P,kmax,stats)
if p == 1 % special case
    tau = 0;
    pval = NaN;
    rho = NaN;
    pval_flattened = NaN;
else
    [rho,pval] = corr(P,'type','Kendall');
    pval_flattened = NaN*ones(1,(size(pval,1)^2-size(pval,1))/2);
    c = 1;
    for i = 1:size(pval,1)
        for j = (i+1):size(pval,1)
            pval_flattened(c) = pval(i,j);
            c = c + 1;
        end
    end
    tau = mean(pval_flattened);
end
rtransformed = (kmax - k)/kmax/(p*(1-tau));
stats.rtransformed = rtransformed;
stats.pval = pval;
stats.rho = rho;
stats.pval_flattened = pval_flattened;
