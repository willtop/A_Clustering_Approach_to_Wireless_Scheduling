%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
open('rcdf.fig')
h = get(gca,'Children');
xdata = get(h,'Xdata');
ydata = get(h,'Ydata');

figure; hold on
grid on
plot(0,1.1,'c-o')
plot(0,1.2,'g-o')
plot(0,1.3,'b-o')
plot(0,1.4,'r-o')
plot(0,1.5,'k-o')
plot(0,1.6,'y-o')
plot(0,1.7,'m-o')
plot(0,1.8,'c->')
plot(0,1.9,'c-s')

legend('1','2','3','4','5','6','7','8','9')


% plot(100,1,'k-s')
% plot(100,1,'b-x')
% plot(100,1,'r-<')
% plot(100,1,'k-o')
% plot(100,1,'b-+')
% plot(100,1,'r->')



% legend('Fixed (2X1)','Proposed (2X1)','MMSE (2X1)',...
%     'Fixed (2X2)','Proposed (2X2)','MMSE (2X2)',...
%     'Fixed (4X2)','Proposed (4X2)','MMSE (4X2)');

%!!

plot(cell2mat(xdata(1)),cell2mat(ydata(1)),'-c')
plot(cell2mat(xdata(2)),cell2mat(ydata(2)),'--g')
plot(cell2mat(xdata(3)),cell2mat(ydata(3)),'-g')
plot(cell2mat(xdata(4)),cell2mat(ydata(4)),'-k')
plot(cell2mat(xdata(5)),cell2mat(ydata(5)),'--r')
plot(cell2mat(xdata(6)),cell2mat(ydata(6)),'-r')
plot(cell2mat(xdata(7)),cell2mat(ydata(7)),'-.b')
plot(cell2mat(xdata(8)),cell2mat(ydata(8)),'--b')
plot(cell2mat(xdata(9)),cell2mat(ydata(9)),'b')

len = length(cell2mat(xdata(1)));
d = 30;
x = cell2mat(xdata(1)); y = cell2mat(ydata(1)); plot(x(1:d:len),y(1:d:len),'g>')
x = cell2mat(xdata(2)); y = cell2mat(ydata(2)); plot(x(1:d:len),y(1:d:len),'ro')
x = cell2mat(xdata(3)); y = cell2mat(ydata(3)); plot(x(1:d:len),y(1:d:len),'b*')
x = cell2mat(xdata(4)); y = cell2mat(ydata(4)); plot(x(1:d:len),y(1:d:len),'r<')
x = cell2mat(xdata(5)); y = cell2mat(ydata(5)); plot(x(1:d:len),y(1:d:len),'bx')
x = cell2mat(xdata(6)); y = cell2mat(ydata(6)); plot(x(1:d:len),y(1:d:len),'ks')
x = cell2mat(xdata(7)); y = cell2mat(ydata(7)); plot(x(1:d:len),y(1:d:len),'r^')
x = cell2mat(xdata(8)); y = cell2mat(ydata(8)); plot(x(1:d:len),y(1:d:len),'b*')
x = cell2mat(xdata(9)); y = cell2mat(ydata(9)); plot(x(1:d:len),y(1:d:len),'kd')

xlabel('Data rate (Mbps)'); ylabel('Cumulative distribution')