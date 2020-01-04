clc; clear;
% cvx_solver SDPT3
% set random seeds (for older version matlab)
ctime = datestr(now,30);
tseed = str2double(ctime((end-5):end));
rand('seed',tseed)
randn('seed',tseed)
% rng(tseed)
rand()
randn()

% cvx_solver SDPT3

bandwidth = 5; % MHz
numBS = 20; % num BS % 30 for cdf
numUser = numBS; % num user
noise = 10^((-169-30)/10)*bandwidth*1e6;
numSlot = 20;%1;%200;%1000;
maxPower = ones(1,numBS)*10^((40-30)/10);
[ G ] = generateNetwork( numBS );

disp('debug: replace G here')
load('channel_losses.mat')
G = g_mat;

algorithms = [2];
algorithm = cell(9,1);
algorithm{1} = 'FP';
algorithm{2} = 'FP2';
algorithm{3} = 'FP3';
algorithm{4} = 'full';
algorithm{5} = 'FlashLinQ';
algorithm{6} = 'TIN';
algorithm{7} = 'ITLinQ';
algorithm{8} = 'ITLinQP';
algorithm{9} = 'ITLinQP_pc';

obj = cell(9);
rate = cell(9);
numSchedule = cell(9);

global converge
converge = nan(51,1);

for alg = algorithms   
    obj{alg} = MyClass(bandwidth, numBS, numUser, noise, numSlot,...
        G, maxPower, algorithm{alg});
    [ numSchedule{alg}, rate{alg} ]= obj{alg}.computeAvgRate();
end

% [numSchedule{1} numSchedule{2} numSchedule{3} numSchedule{4} numSchedule{5} numSchedule{6} numSchedule{7} numSchedule{8} numSchedule{9}]

[sum(rate{1}) sum(rate{2}) sum(rate{3}) sum(rate{4}) sum(rate{5}) sum(rate{6}) sum(rate{7}) sum(rate{8}) sum(rate{9})]

save(strrep(strrep(num2str(clock),' ',''),'.','_'), 'rate','numSchedule','converge');

figure; hold on
[x, y] = ecdf(rate{1}); plot(y,x,'b') % FP
[x, y] = ecdf(rate{2}); plot(y,x,'r') % FP2
[x, y] = ecdf(rate{3}); plot(y,x,'g') % FP3
[x, y] = ecdf(rate{4}); plot(y,x,'y') % full
[x, y] = ecdf(rate{5}); plot(y,x,'m') % FlashLinQ
[x, y] = ecdf(rate{6}); plot(y,x,'k') % TIN
[x, y] = ecdf(rate{7}); plot(y,x,'--b') % ITLinQ
[x, y] = ecdf(rate{8}); plot(y,x,'c') % ITLinQ+
[x, y] = ecdf(rate{9}); plot(y,x,'--y') % ITLinQ+ & pc

% % 
% 
% % utiltiy1 = sum(log(rate{1}))
% utiltiy2 = sum(log(rate{2}))
% utiltiy3 = sum(log(rate{3}))
% % utiltiy4 = sum(log(rate{4}))
% utiltiy5 = sum(log(rate{5}))
% utiltiy6 = sum(log(rate{6}))
% utiltiy7 = sum(log(rate{7}))
% utiltiy8 = sum(log(rate{8}))
% utiltiy9 = sum(log(rate{9}))
