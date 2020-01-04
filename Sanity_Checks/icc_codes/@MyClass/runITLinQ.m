function [ schedule, power ] = runITLinQ( obj, weight )

L = obj.numBS;
G = obj.chnGain;
maxPower = obj.maxPower;
noise = obj.noise;
schedule = 1:L;
power = maxPower;
[~,sort_list] = sort(weight,'descend');

M = 10^(25/10); % 25dB
eta = 0.7;

for i = sort_list'
    if i==sort_list(1)
        scheduled_users = sort_list(1);
        continue
    end
    
    SNR = G(i,i)*power(i)/noise;
    if max(G(scheduled_users,i)*power(i)/noise)<=M*SNR^eta && ...
            max(G(i,scheduled_users).*power(scheduled_users)/noise)<=M*SNR^eta
        scheduled_users = [scheduled_users i];
    else
        schedule(i) = 0;
    end
end
            
end