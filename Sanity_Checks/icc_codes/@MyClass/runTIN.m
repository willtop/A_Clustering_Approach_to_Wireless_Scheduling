function [ schedule, power ] = runTIN( obj, weight )

L = obj.numBS;
G = obj.chnGain;
maxPower = obj.maxPower;
noise = obj.noise;
schedule = 1:L;
power = maxPower;
[~,sort_list] = sort(weight,'descend');

for i = sort_list'
    if i==sort_list(1) % first user always scheduled
        scheduled_user = sort_list(1);
        continue
    end
    
    flag = 1; % set zero if not obeying TIN
    temp_scheduled_user = [scheduled_user i];
    for j = temp_scheduled_user
        other_users = temp_scheduled_user(temp_scheduled_user~=j);
        if G(j,j)*power(j)/noise < max(G(j,other_users).*power(other_users)/noise) * max(G(other_users,j)*power(j))/noise
            flag = 0;
            break
        end
    end
    
    if flag~=0
        scheduled_user = temp_scheduled_user;
    else
        schedule(i) = 0;
    end
end
            
end