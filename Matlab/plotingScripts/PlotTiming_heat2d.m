%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          Plot Simple solutions
%
%               Coded by Manuel Diaz, NTU, 2015.01.07.
%                   Copyright (c) 2014, Manuel Diaz.
%                           All rights reserved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1.1 Load Results
cells = [128^2,256^2,512^2,1024^2,2048^2];
tCPU = [0.03,0.53,8.15,129.89,2094.94];
tGPU = [0.01,0.05,0.42,3.94,55.65];

% 1.2 Set Saving Path
folder='/home/manuel/Dropbox/Apps/Texpad/MyPresentations/GPU-CPU/figures/';

% 2.1 Set plotting defaults
set(0,'defaultTextInterpreter','latex')
set(0,'DefaultTextFontName','Times',...
'DefaultTextFontSize',20,...
'DefaultAxesFontName','Times',...
'DefaultAxesFontSize',20,...
'DefaultLineLineWidth',4.0,...
'DefaultAxesBox','on',...
'defaultAxesLineWidth',2.0,...
'DefaultFigureColor','w',...
'DefaultLineMarkerSize',7.75)

% 2.2 list of options for lines
color=['k','r','m','b','c','g','y','w'];
lines={'-',':','--','-.','-.','none'};
mark=['s','+','o','x','v','none'];

% 2.3 set local marker size
ms = 5;

% Plot figures
semilogy(cells,tCPU,[mark(3),lines{1},color(2)],cells,tGPU,[mark(3),lines{1},color(4)]);

% 3.1 Print axis lines
%x = line([-2 2],[0,0],'color','k');
%y = line([0 0],[-2,2],'color','k');
%axis([0,N,0,rhomax*1.1]); 
grid on; 

% 3.2 Set title, labels and legend
xlabel('Number of cells','FontSize',24); ylabel('$log(time)$','FontSize',24);
hleg = legend('CPU solver','GPU solver','location','southeast');
set(hleg,'FontAngle','italic','FontSize',20)
%legend(hleg,'boxoff')

% 3.3 Export plot to *.eps figure
print('-depsc',[folder,'PlotHeat2D_timming.eps']);