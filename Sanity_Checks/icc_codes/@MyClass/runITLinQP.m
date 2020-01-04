function [ schedule, power ] = runITLinQP( obj, weight )

L = obj.numBS;
G = obj.chnGain;
maxPower = obj.maxPower;
noise = obj.noise;
schedule = 1:L;
power = maxPower;
[~,sort_list] = sort(weight,'descend');

eta = 0.9;
gamma = 0.1;

for i = sort_list'
    if i==sort_list(1)
        scheduled_users = sort_list(1);
        continue
    end
    
    SNR = G(i,i)*power(i)/noise;
    
    flag = 1;
    for j = scheduled_users
        other_users = scheduled_users(scheduled_users~=j);
        if isempty(other_users)
            continue
        end
        %
        INR = G(j,i)*power(i)/noise;
        B = min(G(j,other_users).*power(other_users)/noise);
        if SNR^eta < INR/B^gamma
            flag = 0;
            break
        end
        %
        INR = G(i,j)*power(j)/noise;
        B = min(G(other_users,j)*power(j)/noise);
        if SNR^eta < INR/B^gamma
            flag = 0;
            break
        end
    end
    
    if flag==1
        scheduled_users = [scheduled_users i];
    else
        schedule(i) = 0;
    end
end

% GG = squeeze(G);
% cvx_begin gp
%     variables power(L,1) t(L,1)
%     maximize ( weight(scheduled_users)'*log(t(scheduled_users)) )
%     subject to
%         power >= zeros(L,1)
%         power <= maxPower
%         for i = scheduled_users
%             GG(i,i)*power(i)>=t(i)*(GG(i,1:L~=i)*power(1:L~=i)+noise)%-GG(i,i)*power(i))
% %             power(i) >= 0
% %             power(i) <= maxPower(i)
%         end
% cvx_end
            
end