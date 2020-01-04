function [ power ] = runNewton( obj, weight, numIter, power, schedule )

% consider uplink
L = obj.numBS;
K = obj.numUser;
G = obj.chnGain;
maxPower = obj.maxPower;

for iter = 1:numIter
%     fprintf('Newton___%d\n', iter)
    
%     [ V ] = convertP2V( obj, power );
    sinr = computeSINR(obj, schedule, power);
    [ rate ] = computeCurrentRate(obj, schedule, power);
    converge(iter) = sum(sum(rate));
    
    delt_p = zeros(1,K);
    for j = 1:L
        i = schedule(j);
        if i==0 || power(i)==0
            continue
        end
        A = weight(j)/power(i)/(1+1/sinr(j));
        for n = 1:L
            m = schedule(n);
            if n==j || m==0 || power(m)==0
                continue
            end
            A = A - weight(n)*G(n,i)*sinr(n)^2/power(m)/G(m,n)/(1+sinr(n));
        end
        B = weight(j)/power(i)^2/(1+1/sinr(j))^2;
        delt_p(i) = A/B;
        if isnan(delt_p(i))
            delt_p(i) = 0;
        end
    end
    
    % backtracking line search
    step = 1;
    swr = sum(sum(rate.*weight));
    while 1
        power_new = min(maxPower,max(0,power+step*delt_p));
        [ rate_new] = computeCurrentRate(obj, schedule, power_new);
        swr_new = sum(sum(rate_new.*weight));
        if swr_new >= swr || abs(swr_new-swr)<1e-10
            power = power_new;
            break
        else
            step = step/2;
        end
    end
end

% [ V ] = convertP2V( obj, power );
% 
% figure;
% plot(converge);

end

function [ V ] = convertP2V( obj, power )

K = obj.numUser;
M = obj.numTxAnte;

V = zeros(M,M,K);

for i=1:K
    V(:,:,i) = sqrt(power(i));
end

end