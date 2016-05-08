%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          Plot Simple solutions
%
%               Coded by Manuel Diaz, NTU, 2015.01.07.
%                   Copyright (c) 2014, Manuel Diaz.
%                           All rights reserved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1.1 Load Results
cells = [10^4,10^5,10^6,10^7,10^8];
tCPU = [0.96,2.18,20.94,212.96,2122.09];
tGPU = [0.06,0.11,0.70,5.89,58.46];

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
print('-depsc',[folder,'PlotHeat1D_timming.eps']);