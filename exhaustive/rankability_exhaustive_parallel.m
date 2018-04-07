function [k,p,P] = rankability_exhaustive_parallel(D,num_start_positions)
%% Usage:
% INPUT:  D = n by n data matrix of dominance information. 
%             Can be unweighted (binary) or weighted. e.g., D(i,j)=1 if i beat j. 
%             The diagonal terms must be 0. i.e., D(i,i) = 0
%             For weighted D matrices, 0 <= D(i,j) <= 1
% INPUT:  num_start_positions = number of subtasks. Larger numbers provide
%                               more incremental progress updates but more
%                               overhead. Example values: 10, 100, or 1000.
% OUTPUT: k = minimum number of changes (links added or removed) to
%             transform the input graph to a dominance graph, which can be 
%             reordered to strictly upper triangular form.
% OUTPUT: p = cardinality of set P of rankings (dominance graphs) that can
%             be created if k perturbations are allowed. 
% OUTPUT: P = Set of rankings (dominance graphs) that can
%             be created if k perturbations are allowed. 

n=size(D,1);

if n <= 7
    [k,p,P] = rankability_exhaustive(D);
    return
end

test_inxs = find(D ~= 1 & D ~= 0);
unweighted = true;
if ~isempty(test_inxs) % for unweighted graph 
    unweighted = false;
end

perm=[1:n]; % first permutation input to nextperm.m function is [1:n]
minfitness=Inf; %initialize minfitness at upperbound
P=[];

p = gcp(); % If no pool, do not create new one.
num_parallel = p.NumWorkers;
%end
%num_parallel = poolsize;

%num_start_positions = 100;

convert_func = @(x)char(typecast(x,'uint8'));
invert_func = @(x) typecast(uint8(x),'double');

start_positions_map = containers.Map({convert_func(1:n)},[0]);
for i = 2:num_start_positions
    while true
        start_position = convert_func(randperm(n));        
        if ~isKey(start_positions_map,start_position)
            start_positions_map(start_position) = 0;
            break;
        end
    end
end
start_positions = cellfun(invert_func,keys(start_positions_map),'unif',0);

max_pc = factorial(n);
pc = 0; % permutations completed
fprintf('Starting new parallel brute force run\n');

start_pos = 1;
end_pos = min([start_pos + num_parallel,length(start_positions)]);

while pc < max_pc
    parallel_fitness = Inf*ones(1,num_parallel);
    parallel_P = cell(1,num_parallel);
    number_calculated = zeros(1,num_parallel);
    num_actual_parallel = end_pos-start_pos;
    if num_actual_parallel == 0 % Last one
        num_actual_parallel = 1;
    end
    parfor i = 1:num_actual_parallel
        perfectRG=triu(ones(size(D,1)),1);
        start_position_ix = start_pos+i-1;
        perm = start_positions{start_position_ix};
                
        % Inner loop that is executed independently
        proceed = true;
        while proceed
            Dperm = D(perm,perm);
            if unweighted
                fitness=sum(sum(abs(perfectRG-Dperm)));
            else
                fitness=calc_k(Dperm);
            end

            if fitness < parallel_fitness(i)
                parallel_fitness(i) = fitness;
                parallel_P{i} = [perm'];
            elseif fitness == parallel_fitness(i)
                parallel_P{i}=[parallel_P{i} perm'];  % add perm to P
            end

            number_calculated(i) = number_calculated(i) + 1;

            next_perm = nextperm(perm,n);
            
            % Check to see if we've reached the last permutation
            if sum(abs(next_perm - perm)) == 0
                proceed = false;
            end
            perm = next_perm;
            % Check to see if this next permutation is another key
            perm_str = convert_func(perm);
            if isKey(start_positions_map,perm_str)
                proceed = false;
            end
        end
    end
    
    fitness = min(parallel_fitness);
    if fitness < minfitness
        P = [];
    end
    if fitness <= minfitness
        minfitness = fitness;
        inxs = find(parallel_fitness == minfitness);     
        for ix = inxs
            P = [P parallel_P{ix}];
        end
    end
    
    start_pos = min([start_pos + num_parallel,length(start_positions)]);
    end_pos = min([start_pos + num_parallel,length(start_positions)]);
    
    pc = pc + sum(number_calculated);
    
    fprintf('Percent Complete: %.2f\n',pc/max_pc*100);
end 
fprintf('Finished parallel brute force run\n');

k = minfitness;
p = size(P,2);


return


% Analyze the set P by creating four matrices R, Padd (P_+ in paper), 
%      Pdelete (P_- in paper), and DOM (P_> in paper)

% Ranking matrix R; R(i,j)=percentage of ranking vectors in P that have
% item i in jth rank position
R=zeros(n,n);
for i=1:n
    for j=1:p
        R(P(i,j),i)=R(P(i,j),i)+1;
    end
end
R=R./p
%figure(1)
%spy(R)

% DOM matrix where DOM(i,j)=percentage of ranking vectors in P that have
% i>j
% Padd matrix; Padd(i,j)=percentage of rankings in P that add link from i to j
% Pdelete matrix; Pdelete(i,j)=percentage of rankings in P that delete link from i to j
DOM=zeros(n,n);
Padd=zeros(n,n);
Pdelete=zeros(n,n);
for h=1:p
    ranking=P(:,h);
    for i=1:n
        for j=i+1:n
            DOM(ranking(i),ranking(j))=DOM(ranking(i),ranking(j))+1;
            if D(ranking(i),ranking(j))==0
                Padd(ranking(i),ranking(j))=Padd(ranking(i),ranking(j))+1;
            end
            if D(ranking(j),ranking(i))==1
                Pdelete(ranking(j),ranking(i))=Pdelete(ranking(j),ranking(i))+1;
            end
        end
    end
end
DOM=DOM./p
%figure(2)
%spy(DOM)
 
Padd=Padd./p
Pdelete=Pdelete./p
[row,col]=find(Pdelete==1);
% each row of noiselinks contains a link that appears to be noise, since
% every ranking in P contains the dominance relation in the opposite direction
noiselinks=[row col]



imagesc(1-R)
ax = gca
ax.Visible = 'off'
colormap(gray)
caxis([0 1])

    

