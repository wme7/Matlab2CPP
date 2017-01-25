% Run Linea propagation
clear; clc; close all;

% Numerical Domain
   L = 2.0; % m
   W = 2.0; % m
  nx = 1001; 
  ny = 1201; 

% Set manually dt
  dt = 0.0010; 
  
% Set run parameters
alpha= 1.0; % advection velocity
  CFL= 0.60; % Stability parameter
iters= round(0.2/dt);

% Verify the time step is correct
dt0 = CFL/alpha*max(L/(nx-1),W/(ny-1));
fprintf('dt0 : %g\n',dt0);
tFinal = iters*dt0; fprintf('tEnd: %g\n',tFinal);
if dt>dt0; error('time step is too large!'); end

% GPU Block Sizes
block_X = 16;
block_Y = 16;

%% Derived Parameters

% Mesh 2D
x=linspace(-L/2,L/2,nx);
y=linspace(-W/2,W/2,ny);
[X,Y]= meshgrid(x,y);
dx = L/(nx-1); fprintf(' dx : %g\n',dx);
dy = W/(ny-1); fprintf(' dy : %g\n',dy);

%% Write sh.run
fID = fopen('run.sh','wt'); fprintf(fID,'make\n'); 
args = sprintf('%1.1f %1.6f %1.6f %1.6f %d %d %d %d %d',alpha,dt,dx,dy,nx,ny,iters,block_X,block_Y);
fprintf(fID,'./advection2d.run %s\n',args); fclose(fID);

% Execute sh.run
! sh run.sh

%% Load data
fID = fopen('result.bin');
output = fread(fID,nx*ny,'float')';

% 3D Plot data
myplot(output,nx,ny,L,W);

% 2D Plot of pressure profile
figure; surf(reshape(output,nx,ny),'EdgeColor','none'); axis tight; 

% Clean up
! rm -rf *.bin
