function [ SINR ] = computeSINR( obj, schedule, power )

L = obj.numBS;
K = obj.numUser;
G = obj.chnGain;
noise = obj.noise;
SINR = zeros(K,1);

for j = 1:L
    i = schedule(j);
    if i==0
        continue % the case where no user is scheduled
    end

    A = G(i,i)*power(i);
    if A == 0
        continue
    end

    B = noise;
    for n = 1:L
        m = schedule(n);
        if m~=0 && m~=i
            B = B + G(i,m)*power(m);
        end
    end

    SINR(i) = A/B;
end

end