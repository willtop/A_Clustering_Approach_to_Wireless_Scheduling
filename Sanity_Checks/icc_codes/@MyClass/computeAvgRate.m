function [ numSchedule, avgRate ] = computeAvgRate( obj )

L = obj.numBS;
sumRate = zeros(L,1);
weight = ones(L,1)*1e3;
alpha = .95;
numSchedule = 0;

for t = 1:obj.numSlot
    [ schedule, power ] = runAlgorithm(obj, weight);
    
    fprintf('slot %d: %s\n', t, obj.algorithm);
    schedule
    currentRate = computeCurrentRate(obj, schedule, power)
    sumRate = sumRate + currentRate;
    weight = 1 ./ (alpha./weight + (1-alpha)*currentRate)
    numSchedule = numSchedule + sum(schedule~=0);
end
    
disp('avgRate info')
avgRate = sumRate/obj.numSlot
size(avgRate)

numSchedule = numSchedule/L/obj.numSlot; % percentage of scheduling

end

