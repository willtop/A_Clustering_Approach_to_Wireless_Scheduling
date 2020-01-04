clear;

s = what; %look in current directory
%s=what('dir') %change dir for your directory name 
matfiles=s.mat;
nSeeds = numel(matfiles);

Converge = zeros(51,2);

for a=1:nSeeds
    load(char(matfiles(a)));
    Converge = Converge + converge;
end

Converge = Converge/nSeeds;

% figure; hold on
plot(0:30,Converge(1:31,1),'--b')
plot(0:30,Converge(1:31,2),'--r')