%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          Plot Simple solutions
%
%               Coded by Manuel Diaz, NTU, 2015.01.07.
%                   Copyright (c) 2014, Manuel Diaz.
%                           All rights reserved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1.1 Load Results
load('Results.mat'); N=length(rho);

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

% 3. Plot Density
figure(1); rhomax = max(rho); plot(rho,[lines{1},color(2)]); 


% 3.1 Print axis lines
%x = line([-2 2],[0,0],'color','k');
%y = line([0 0],[-2,2],'color','k');
axis([0,N,0,rhomax*1.1]); grid on;

% 3.2 Set title, labels and legend
xlabel('Cells','FontSize',24); ylabel('$\rho$','FontSize',24);
hleg = legend('SHLL solution','location','southwest');
set(hleg,'FontAngle','italic','FontSize',20)
legend(hleg,'boxoff')

% 3.3 Export plot to *.eps figure
print('-depsc',[folder,'PlotEuler_rho.eps']);




% 4. Plot Velocity UX
figure(2); uxmax=max(ux); plot(ux,[lines{1},color(2)]); 


% 4.1 Print axis lines
%x = line([-2 2],[0,0],'color','k');
%y = line([0 0],[-2,2],'color','k');
axis([0,N,0,uxmax*1.1]); grid on;

% 4.2 Set title, labels and legend
xlabel('Cells','FontSize',24); ylabel('$u_x$','FontSize',24);
hleg = legend('SHLL solution','location','northwest');
set(hleg,'FontAngle','italic','FontSize',20)
legend(hleg,'boxoff')

% 4.3 Export plot to *.eps figure
print('-depsc',[folder,'PlotEuler_ux.eps']);



% 5. Plot Total Energy
figure(3); Tmax=max(T); plot(T,[lines{1},color(2)]); 


% 5.1 Print axis lines
%x = line([-2 2],[0,0],'color','k');
%y = line([0 0],[-2,2],'color','k');
axis([0,N,0.5,1.5]); grid on;

% 5.2 Set title, labels and legend
xlabel('Cells','FontSize',24); ylabel('$T$','FontSize',24);
hleg = legend('SHLL solution','location','northwest');
set(hleg,'FontAngle','italic','FontSize',20)
legend(hleg,'boxoff')

% 5.3 Export plot to *.eps figure
print('-depsc',[folder,'PlotEuler_T.eps']);