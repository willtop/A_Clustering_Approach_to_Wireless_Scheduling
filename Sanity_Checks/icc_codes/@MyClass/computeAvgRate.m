function [ numSchedule, avgRate, whether_encountered ] = computeAvgRate( obj )

L = obj.numBS;
sumRate = zeros(L,1);
weight = ones(L,1)*1e3;
alpha = .95;
numSchedule = 0;

whether_encountered = false;

for t = 1:obj.numSlot
    [ schedule, power ] = runAlgorithm(obj, weight);
    
    %fprintf('slot %d: %s\n', t, obj.algorithm);
    
    currentRate = computeCurrentRate(obj, schedule, power);
    sumRate = sumRate + currentRate;
    weight = 1 ./ (alpha./weight + (1-alpha)*currentRate);
    numSchedule = numSchedule + sum(schedule~=0);
end
    
avgRate = sumRate/obj.numSlot;
if any(avgRate==0)
    disp('Found it: with the average rates: ')
    obj.algorithm
    sumRate
    avgRate
    whether_encountered = true;
end
numSchedule = numSchedule/L/obj.numSlot; % percentage of scheduling

end

