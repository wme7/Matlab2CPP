function myplot(result,nx)

L = 2.0;

x = linspace(0,L,nx);

h = plot(x,result(:,2),'.b'); 
%axis tight;
axis([0,2,-0.5,0.5]);
grid on;

print('heat1d','-dpng')
