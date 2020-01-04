clc; clear;
K = 300;
s = what; %look in current directory
%s=what('dir') %change dir for your directory name 
matfiles=s.mat;
nSeeds = numel(matfiles);

[ rateFP rateFP2 rateFP3 rateFull rateFlash rateTIN rateIT rateITP rateITPP ] = deal(zeros(K,1));

for a=1:nSeeds
    load(char(matfiles(a)));
    
    rateFP = rateFP + sort(rate{1});
    rateFP2 = rateFP2 + sort(rate{2});
    rateFP3 = rateFP3 + sort(rate{3});
    rateFull = rateFull + sort(rate{4});
    rateFlash = rateFlash + sort(rate{5});
    rateTIN = rateTIN + sort(rate{6});
    rateIT = rateIT + sort(rate{7});
    rateITP = rateITP + sort(rate{8});
    rateITPP = rateITPP + sort(rate{9});
end

rateFP = rateFP/nSeeds;
rateFP2 = rateFP2/nSeeds;
rateFP3 = rateFP3/nSeeds;
rateFull = rateFull/nSeeds;
rateFlash = rateFlash/nSeeds;
rateTIN = rateTIN/nSeeds;
rateIT = rateIT/nSeeds;
rateITP = rateITP/nSeeds;
rateITPP = rateITPP/nSeeds;

figure; hold on
[x,y]=ecdf(rateFP); plot(y,x,'b')
[x,y]=ecdf(rateFP2); plot(y,x,'--b')
[x,y]=ecdf(rateFP3); plot(y,x,'-.b')
[x,y]=ecdf(rateFull); plot(y,x,'r')
[x,y]=ecdf(rateFlash); plot(y,x,'--r')
[x,y]=ecdf(rateTIN); plot(y,x,'k')
[x,y]=ecdf(rateIT); plot(y,x,'g')
[x,y]=ecdf(rateITP); plot(y,x,'--g')
[x,y]=ecdf(rateITPP); plot(y,x,'c')

[x,y]=ecdf(avgRateProposed(:,1)); plot(y,x,'b')
[x,y]=ecdf(avgRateWMMSE(:,1)); plot(y,x,'r')
[x,y]=ecdf(avgRateFixed(:,1)); plot(y,x,'g')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
open('rcdf.fig')
h = get(gca,'Children');
xdata = get(h,'Xdata');
ydata = get(h,'Ydata');

figure; hold on
grid on
plot(101,1,'r->')
plot(102,1,'g-o')
plot(103,1,'b-*')


% plot(100,1,'k-s')
% plot(100,1,'b-x')
% plot(100,1,'r-<')
% plot(100,1,'k-o')
% plot(100,1,'b-+')
% plot(100,1,'r->')

legend('Proposed', 'WMMSE', 'Fixed interference')

% legend('Fixed (2X1)','Proposed (2X1)','MMSE (2X1)',...
%     'Fixed (2X2)','Proposed (2X2)','MMSE (2X2)',...
%     'Fixed (4X2)','Proposed (4X2)','MMSE (4X2)');

%!!

plot(cell2mat(xdata(1)),cell2mat(ydata(1)),'-g')
plot(cell2mat(xdata(2)),cell2mat(ydata(2)),'-r')
plot(cell2mat(xdata(3)),cell2mat(ydata(3)),'-b')
% plot(cell2mat(xdata(4)),cell2mat(ydata(4)),'-r')
% plot(cell2mat(xdata(5)),cell2mat(ydata(5)),'-b')
% plot(cell2mat(xdata(6)),cell2mat(ydata(6)),'-k')
% plot(cell2mat(xdata(7)),cell2mat(ydata(4)),'-r')
% plot(cell2mat(xdata(8)),cell2mat(ydata(5)),'-b')
% plot(cell2mat(xdata(9)),cell2mat(ydata(6)),'-k')

len = length(cell2mat(xdata(1)));
x = cell2mat(xdata(1)); y = cell2mat(ydata(1)); plot(x(1:5:len),y(1:5:len),'g>')
x = cell2mat(xdata(2)); y = cell2mat(ydata(2)); plot(x(1:5:len),y(1:5:len),'ro')
x = cell2mat(xdata(3)); y = cell2mat(ydata(3)); plot(x(1:5:len),y(1:5:len),'b*')
% x = cell2mat(xdata(4)); y = cell2mat(ydata(4)); plot(x(1:5:len),y(1:5:len),'r<')
% x = cell2mat(xdata(5)); y = cell2mat(ydata(5)); plot(x(1:5:len),y(1:5:len),'bx')
% x = cell2mat(xdata(6)); y = cell2mat(ydata(6)); plot(x(1:5:len),y(1:5:len),'ks')
% x = cell2mat(xdata(7)); y = cell2mat(ydata(7)); plot(x(1:5:len),y(1:5:len),'r^')
% x = cell2mat(xdata(8)); y = cell2mat(ydata(8)); plot(x(1:5:len),y(1:5:len),'b*')
% x = cell2mat(xdata(9)); y = cell2mat(ydata(9)); plot(x(1:5:len),y(1:5:len),'kd')

xlabel('Data rate (Mbps)'); ylabel('Cumulative distribution')