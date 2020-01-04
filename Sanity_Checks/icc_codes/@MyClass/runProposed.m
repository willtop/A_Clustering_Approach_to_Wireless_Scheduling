function [ schedule, power ] = runProposed( obj, weight, option )

L = obj.numBS;
K = obj.numUser;
maxPower = obj.maxPower;
schedule = 1:L;
numIter = 100;

global converge

V = sqrt(maxPower);%.*rand(1,L);

if option==0
    [ rate ] = computeCurrentRate(obj, schedule, V.^2);
    converge(1) = sum(rate);
end

% algorithm iterations
for iter = 1:numIter

    power = V.^2;
    SINR = computeSINR(obj, schedule, power);
    y = updateY( obj, schedule, V, weight, SINR );
    V = updateV( obj, V, weight, y, SINR );
    
    if option==0
        [ rate ] = computeCurrentRate(obj, schedule, V.^2);
        converge(iter+1) = sum(rate);
    end    
end

% if option==0
%     figure; plot(converge)  
% end

switch option
    case 0
        return
    case 2
        for i = 1:K
            if V(i)>=sqrt(maxPower(i))/2
                power(i) = maxPower(i);
                schedule(i) = i;
            else
                schedule(i) = 0;
            end
        end
    case 3
        for i = 1:K
            if V(i)>=sqrt(maxPower(i))*3/4
                power(i) = maxPower(i);
                schedule(i) = i;
            else if V(i)>=sqrt(maxPower(i))*1/4
                    power(i) = maxPower(i)/2;
                    schedule(i) = i;
                else
                    schedule(i) = 0;
                end
            end
        end
    case 4
        for i = 1:K
            if V(i)^2>=maxPower(i)/2
                power(i) = maxPower(i);
                schedule(i) = i;
            else
                schedule(i) = 0;
            end
        end    
    otherwise
        error('unknown option')
end

end

%%
function [ y ] = updateY( obj, schedule, V, weight, SINR )

L = obj.numBS;
H = sqrt(obj.chnGain);

y = zeros(L,1);
for j = 1:L
    i = schedule(j);
    if i~=0
        y(j) = sqrt(weight(i)*(1+SINR(j))) / (1+1/SINR(j))...
            /norm(H(i,j)*V(i));
    end
    if isnan(y(j,1))
        y(j) = 0;
    end
end

end

%%
function [ V ] = updateV( obj, V, weight, y, SINR )

K = obj.numUser;
H = sqrt(obj.chnGain);
maxPower = obj.maxPower;

for i = 1:K
    B = 0;
    for n = 1:K
        B = B + y(n)^2*norm(H(n,i))^2;
    end
    
    A = y(i)*sqrt(weight(i)*(1+SINR(i)))*norm(H(i,i));
    
    V(i) = min(A/B, sqrt(maxPower(i)));
end
            
end