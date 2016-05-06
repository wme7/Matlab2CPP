%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            Solving 3-D heat equation with jacobi method
%
%                u_t = D*(u_xx + u_yy + u_zz) + s(u), 
%      for (x,y,z) \in [0,L]x[0,W]x[0,H] and S = s(u): source term
%
%             coded by Manuel Diaz, manuel.ade'at'gmail.com
%        National Health Research Institutes, NHRI, 2016.02.11
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; %close all; clc;
 
%% Parameters
D = 1.0; % alpha
tFinal = 0.1;	% End time
L = 1; nx = 32; dx = L/nx; 
W = 1; ny = 32; dy = W/ny;
H = 1; nz = 64; dz = H/nz;
Dx = D/dx^2; Dy = D/dy^2; Dz = D/dz^2;

% Build Numerical Mesh
[x,y,z] = meshgrid(0:dx:L,0:dy:W,0:dz:H);

% Add source term
sourcefun='dont'; % add source term
switch sourcefun
    case 'add';     S = @(w) 0.1*w.^2;
    case 'dont';    S = @(w) zeros(size(w));
end

% Build IC
u0 = sin(pi*x).*sin(pi*y).*sin(pi*z);

% Build Exact solution
uE = exp(-3*D*tFinal*pi^2)*sin(pi*x).*sin(pi*y).*sin(pi*z);

% Set Initial time step
dt0 = 1/(2*D*(1/dx^2+1/dy^2+1/dz^2)); % stability condition

% Set plot region
region = [0,L,0,W,0,H]; 

%% Solver Loop 
% load initial conditions 
t=dt0; it=0; u=u0; dt=dt0;
 
while t < tFinal
    % RK stages
    uo=u;

    % forward euler solver
    u = Laplace3d(uo,nx+1,ny+1,nz+1,Dx,Dy,Dz,S,dt);
    
    % set BCs
    u(1,:,:) = 0; u(nx+1,:,:) = 0;
    u(:,1,:) = 0; u(:,ny+1,:) = 0;
    u(:,:,1) = 0; u(:,:,nz+1) = 0;

    % compute time step
    if t+dt>tFinal, dt=tFinal-t; end; 
    
    % Update iteration counter and time
    it=it+1; t=t+dt;
    
    % plot solution
    if mod(it,100); slice(x,y,z,u,L/2,W/2,H/2); axis(region); drawnow; end
end
 
%% % Post Process 
% Final Plot
figure(2);
subplot(121); h=slice(x,y,z,u,L/2,W/2,H/2); axis(region);
title('heat3d, Cell Averages','interpreter','latex','FontSize',18);
h(1).EdgeColor = 'none';
h(2).EdgeColor = 'none';
h(3).EdgeColor = 'none';
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{y}$','interpreter','latex','FontSize',14);
zlabel('$\it{z}$','interpreter','latex','FontSize',14);
colorbar;
subplot(122); q=slice(x,y,z,uE,L/2,W/2,H/2); axis(region);
title('heat3d, Exact solution','interpreter','latex','FontSize',18);
q(1).EdgeColor = 'none';
q(2).EdgeColor = 'none';
q(3).EdgeColor = 'none';
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{y}$','interpreter','latex','FontSize',14);
zlabel('$\it{z}$','interpreter','latex','FontSize',14);
colorbar;

% Error norms
err = abs(uE(:)-u(:));
L1 = dx*dy*dz*norm(err,1); fprintf('L_1 norm: %1.2e \n',L1);
L2 = dx*dy*dz*norm(err,2); fprintf('L_2 norm: %1.2e \n',L2);
Linf = norm(err,inf); fprintf('L_inf norm: %1.2e \n',Linf);