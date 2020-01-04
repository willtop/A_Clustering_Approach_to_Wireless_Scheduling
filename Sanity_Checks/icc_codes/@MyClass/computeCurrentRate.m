function [ currentRate ] = computeCurrentRate(obj, schedule, power)

currentSINR = computeSINR(obj, schedule, power);
currentRate = obj.bandwidth*1e6*log2(1+currentSINR);

end