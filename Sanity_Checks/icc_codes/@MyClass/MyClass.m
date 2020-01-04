classdef MyClass
    
    properties
        bandwidth
        chnGain
        maxPower
        numBS
        numUser
        numSlot
        noise
        algorithm
    end
    
    methods
        function obj = MyClass( bandwidth, numBS, numUser, noise, numSlot,...
                chnGain, maxPower, algorithm)
            obj.bandwidth = bandwidth;
            obj.numBS = numBS;
            obj.numUser = numUser;
            obj.noise = noise;
            obj.numSlot = numSlot;
            obj.chnGain = chnGain;
            obj.maxPower = maxPower;
            obj.algorithm = algorithm;
        end
    end
    
end

