function [ schedule, power ] = runFlashLinQ( obj, weight )

L = obj.numBS;
G = obj.chnGain;
maxPower = obj.maxPower;
schedule = 1:L;
power = maxPower;
[~,sort_list] = sort(weight,'descend');

thres = 10^(9/10);

for i = sort_list'
    if i==sort_list(1)
        scheduled_users = sort_list(1);
        continue
    end
    
    flag = 1;
    for j = scheduled_users
        if G(j,j)*power(j)/(G(j,i)*power(i))<=thres
            flag = 0;
            break
        end
    end
    
    if G(i,i)/sum(G(i,scheduled_users).*power(scheduled_users))<=thres
        flag = 0;
    end
    
    if flag==1
        scheduled_users = [scheduled_users i];
    else
        schedule(i) = 0;
    end
end
            
end