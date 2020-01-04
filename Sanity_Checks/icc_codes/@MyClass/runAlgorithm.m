function [ schedule, power ] = runAlgorithm( obj, weight )

numIter = 100;

switch obj.algorithm
    case 'FP'
        option = 0;
        [ schedule,power ] = runProposed( obj, weight, option );
    case 'FP2'
        option = 2;
        [ schedule,power ] = runProposed( obj, weight, option );
    case 'FP3'
        option = 3;
        [ schedule,power ] = runProposed( obj, weight, option );
    case 'FP4'
        option = 4;
        [ schedule,power ] = runProposed( obj, weight, option );        
    case 'full'
        schedule = 1:obj.numBS;
        power = obj.maxPower;
    case 'TIN'
        [ schedule, power ] = runTIN( obj, weight );
    case 'ITLinQ'
        [ schedule, power ] = runITLinQ( obj, weight );
    case 'ITLinQP'
        [ schedule, power ] = runITLinQP( obj, weight );    
    case 'FlashLinQ'
        [ schedule, power ] = runFlashLinQ( obj, weight );
    case 'ITLinQP_pc'
        [ schedule, power ] = runITLinQP_pc( obj, weight );
    otherwise
        error('unknown algorithm')
end

end