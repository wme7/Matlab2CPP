%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          Plot Simple solutions
%
%               Coded by Manuel Diaz, NTU, 2015.01.07.
%                   Copyright (c) 2014, Manuel Diaz.
%                           All rights reserved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1.1 Load Results
cells = [(32^2)*64,(64^2)*128,(128^2)*256,(256^2)*512,(512^2)*1024];
tCPU = [0.04,1.41,177.72,5871.83,200000];
tGPU = [0.0005,0.01,0.41,11.35,378.56];

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
print('-depsc',[folder,'PlotHeat3D_timming.eps']);