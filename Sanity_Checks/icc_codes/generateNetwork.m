function [ G ] = generateNetwork( L )

% ITU-1411

range = 500; % here and throughout, unit is meter
txPosition = (rand(L,1) + 1i*rand(L,1))*range;
maxDist = 50;
minDist = 30;

rxPosition = nan(L,1);
for i = 1:L
    while(1)
        rand_dir = randn() + randn()*1i;
        dist = (minDist+(maxDist-minDist)*rand);
        rxPosition(i) = dist*rand_dir/norm(rand_dir) + txPosition(i);
        norm(rxPosition(i)-txPosition(i)) % print dist of each link
        if real(rxPosition(i))>=0 && real(rxPosition(i))<=range ...
            && imag(rxPosition(i))>=0 && imag(rxPosition(i))<=range
            break
        end
    end
end

% plot the topology
%figure; 
%hold on;
%for i = 1:L
%    plot([real(txPosition(i)),real(rxPosition(i))],[imag(txPosition(i)),imag(rxPosition(i))],'k');
%end
%plot(real(txPosition), imag(txPosition),'k.');
%plot(real(rxPosition), imag(rxPosition),'k.');
% axis([-1 1 -1 1]*1.5); legend('Macro BS','Femto BS','MS');
%xlabel('km'); ylabel('km');

c = 3e8; % speed of light
freq = 2.4e9; % in Hz
wavelength = c/freq; % in meter
Rbp = 4*1.5^2/wavelength;
Lbp = abs(20*log10(wavelength^2/(8*pi*1.5^2)));

PL = nan(L,L); % pathloss in dB
for i = 1:L
    for j = 1:L
        dist = abs(txPosition(j)-rxPosition(i));
        if dist<=Rbp
            PL(i,j) = Lbp + 6 + 20*log10(dist/Rbp);
        else
            PL(i,j) = Lbp + 6 + 40*log10(dist/Rbp);
        end
    end
end

PL = PL + randn(L,L)*8 - 2.5 + 7; % shadowing; ante gain; noie figure
% add in fast fading
%PL = PL * (randn(L,L).^2 + randn(L,L).^2)/2
G = 10.^(-PL/10);

end