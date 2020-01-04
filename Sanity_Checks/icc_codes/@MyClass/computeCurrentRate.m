function [ currentRate ] = computeCurrentRate(obj, schedule, power)

currentSINR = computeSINR(obj, schedule, power);
currentRate = obj.bandwidth*log2(1+currentSINR);

end