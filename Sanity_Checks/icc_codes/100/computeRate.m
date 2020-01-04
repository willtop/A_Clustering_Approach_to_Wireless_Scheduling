clc; clear;

[rateFP, rateFP2, rateFP3, rateFull, rateTIN, rateITLinQ, rateITLinQP, rateFlashLinQ, rateITLinQP_pc] = deal(0);
% [scheduleFP2 scheduleFP3 scheduleFull scheduleTIN scheduleITLinQ scheduleITLinQP scheduleFlashLinQ] = deal(0);
s = what; %look in current directory
%s=what('dir') %change dir for your directory name 
matfiles=s.mat;
nSeeds = numel(matfiles);

for a=1:nSeeds
    load(char(matfiles(a)));
    
    rateFP = rateFP + sum(rate{1});
    rateFP2 = rateFP2 + sum(rate{2});
    rateFP3 = rateFP3 + sum(rate{3});
    rateFull = rateFull + sum(rate{4});
    rateFlashLinQ = rateFlashLinQ + sum(rate{5});
    rateTIN = rateTIN + sum(rate{6});
    rateITLinQ = rateITLinQ + sum(rate{7});
    rateITLinQP = rateITLinQP + sum(rate{8});
    rateITLinQP_pc = rateITLinQP_pc + sum(rate{9});
%     scheduleFP2 = scheduleFP2 + numSchedule{2};
%     scheduleFP3 = scheduleFP3 + numSchedule{3};
%     scheduleFull = scheduleFull + numSchedule{5};
%     scheduleTIN = scheduleTIN + numSchedule{6};
%     scheduleITLinQ = scheduleITLinQ + numSchedule{7};
%     scheduleITLinQP = scheduleITLinQP + numSchedule{8};
%     scheduleFlashLinQ = scheduleFlashLinQ + numSchedule{9};
end

rateFP = rateFP/nSeeds;
rateFP2 = rateFP2/nSeeds;
rateFP3 = rateFP3/nSeeds;
rateFull = rateFull/nSeeds;
rateFlashLinQ = rateFlashLinQ/nSeeds;
rateTIN = rateTIN/nSeeds;
rateITLinQ = rateITLinQ/nSeeds;
rateITLinQP = rateITLinQP/nSeeds;
rateITLinQP_pc = rateITLinQP_pc/nSeeds;

save('100', 'rateFP', 'rateFP2', 'rateFP3', 'rateFull', 'rateTIN', 'rateITLinQ', 'rateITLinQP', 'rateFlashLinQ', 'rateITLinQP_pc');


% [rateFP rateFP2 rateFP3 rateFull rateFlashLinQ rateTIN rateITLinQ rateITLinQP rateITLinQP_pc ]/nSeeds

% [scheduleFP2 scheduleFP3 scheduleFull scheduleTIN scheduleITLinQ scheduleITLinQP scheduleFlashLinQ]/nSeeds