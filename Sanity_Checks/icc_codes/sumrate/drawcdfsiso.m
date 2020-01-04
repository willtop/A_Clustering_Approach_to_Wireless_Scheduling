clear;

s = what; %look in current directory
%s=what('dir') %change dir for your directory name 
matfiles=s.mat;
nSeeds = numel(matfiles);

[RateFP RateFP2 RateFP3 RateFull RateTIN RateITLinQ RateITLinQP RateFlashLinQ RateITLinQP_pc] = deal(nan(5,1));

for a=1:nSeeds
    load(char(matfiles(a)));
    
    RateFP(a) = rateFP;
    RateFP2(a) = rateFP2;
    RateFP3(a) = rateFP3;
    RateFull(a) = rateFull;
    RateFlashLinQ(a) = rateFlashLinQ;
    RateTIN(a) = rateTIN;
    RateITLinQ(a) = rateITLinQ;
    RateITLinQP(a) = rateITLinQP;
    RateITLinQP_pc(a) = rateITLinQP_pc;
end


figure; hold on
plot(100:200:900, sort(RateFP),'b')
plot(100:200:900, sort(RateFP2),'--b')
plot(100:200:900, sort(RateFP3),'-.b')
plot(100:200:900, sort(RateFull),'r')
plot(100:200:900, sort(RateTIN),'k')
plot(100:200:900, sort(RateFlashLinQ),'--r')
plot(100:200:900, sort(RateITLinQ),'g')
plot(100:200:900, sort(RateITLinQP),'--g')
plot(100:200:900, sort(RateITLinQP_pc),'c')