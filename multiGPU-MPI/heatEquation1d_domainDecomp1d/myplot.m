function myplot(result,nx)

L = 10.0;

x = linspace(0,L,nx);

h = plot(x,result(:,2),'.b'); 
axis tight;

%print('heat1d','-dpng')
