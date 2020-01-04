function [ schedule, power ] = runITLinQP_pc( obj, weight )

[ schedule, power ] = runITLinQP( obj, weight );
power = runNewton( obj, weight, 50, power, schedule );

end