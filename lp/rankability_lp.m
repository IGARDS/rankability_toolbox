function [k,p,r,ranking,X,Y] = rankability_lp(D)

n = size(D,1);
smartk=(n^2-n)/2; 
intcon=[1:2*n*n]';
f=ones(2*n*n,1); % for unweighted graph
%f=[ones(n*n,1);reshape(D,n*n,1)]; % for weighted graph D
lb=[zeros(2*n*n,1)];
ub=[ones(n*n,1)-reshape(D,n*n,1);reshape(D,n*n,1)];
% force xii=0 and yii=0 for all i by setting lb=0 and ub=0 for these indices
indicesxii=1:n+1:n^2;
lb(indicesxii)=0;
ub(indicesxii)=0;
indicesyii=n^2+1:n+1:2*n^2;
lb(indicesyii)=0;
ub(indicesyii)=0;


% inequality 1: xij + yij <= 1; for all i,j
% number of contraints of this type: n^2
A1=[eye(n*n) eye(n*n)];
b1=ones(n*n,1);




% inequality 2: transitivity triplets for all j ~= i, k~=j, k~= i
% number of contraints of this type: n*(n-1)*(n-2)
% dij + xij - yij + djk + xjk - yjk + dki + xki - yki <= 2
% or
% xij - yij + xjk - yjk + xki - yki <= 2 - dij - djk - dki
A2=spalloc(n*(n-1)*(n-2),n^2,6*n*(n-1)*(n-2));
count=1;
for i=1:n
    for j=1:n 
        if j~=i
            for k=1:n
                if (k~=j & k~=i)
                   A2(count,n*(j-1)+i) = 1;
                   A2(count,n*(k-1)+j) = 1;
                   A2(count,n*(i-1)+k) = 1;
                   d(count)=D(i,j)+D(j,k)+D(k,i);
                   count=count+1;
                end
            end
        end
    end
end
A2=[A2 -A2];
b2=[2*ones(n*(n-1)*(n-2),1)]-d';
clear d;

% inequality 3: smart initialization 
% number of contraints of this type: 1
A3=ones(1,2*n*n);
b3=smartk;


% concatenate all inequalities into A matrix and b vector
A=[A1;A2;A3];
b=[b1;b2;b3];
clear A1 A2 A3 b1 b2 b3;


% equality constraint: antisymmetry for all i<j 
% number of contraints of this type: n*(n-1)/2
% dij + xij - yij + dji + xji - yji = 1
% or
% xij - yij + xji - yji = 1 - dij - dji
Aeq=spalloc(n*(n-1)/2,n*n,n*(n-1));
count=1;
for i=1:n
    for j=i+1:n
        Aeq(count,(i-1)*n+j)=1;
        Aeq(count,(j-1)*n+i)=1;
        d(count)=D(i,j)+D(j,i);
        count=count+1;
    end
end
Aeq=[Aeq -Aeq];
beq=ones(n*(n-1)/2,1)-d';



% solve LP
%tic;
options=optimoptions('linprog','Display','none','Algorithm','interior-point');
[x,fval,exitflag,output] = linprog(f,A,b,Aeq,beq,lb,ub,options);
%fval
%output
%LPtime=toc

k=fval;
X=reshape(x(1:n^2),n,n);
Y=reshape(x(n^2+1:2*n^2),n,n);
D+X-Y;
rowsum=sum(D+X-Y,2);
[r,ranking]=sort(rowsum,'descend');
nfracX=nnz(.001<X&X<.999);
nfracY=nnz(.001<Y&Y<.999);
p=max([nfracX+nfracY,1]);


