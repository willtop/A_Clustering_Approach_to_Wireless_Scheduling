clear;

s = what; %look in current directory
%s=what('dir') %change dir for your directory name 
matfiles=s.mat;
nSeeds = numel(matfiles);

[rateProposed,rateDirect,rateFP] = deal(0);

for a=1:nSeeds
    load(char(matfiles(a)));
    
    rateProposed = rateProposed + sum(rate{1});
    rateDirect = rateDirect + sum(rate{2});
    rateFP = rateFP + sum(rate{3});
end

rateProposed/nSeeds
rateDirect/nSeeds
rateFP/nSeeds

save('60mimo', 'rateProposed','rateDirect');