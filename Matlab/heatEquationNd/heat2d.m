%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            Solving 2-D heat equation with jacobi method
%
%                   u_t = D*(u_xx + u_yy) + s(u), 
%         for (x,y) \in [0,L]x[0,W] and S = s(u): source term
%
%             coded by Manuel Diaz, manuel.ade'at'gmail.com
%        National Health Research Institutes, NHRI, 2016.02.11
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; %close all; clc;
 
%% Parameters
D = 1.0; % alpha
tFinal = 0.1;	% End time
L = 2; nx = 32; dx = L/nx; 
W = 2; ny = 32; dy = W/ny;
Dx = D/dx^2; Dy = D/dy^2; 

% Build Numerical Mesh
[x,y] = meshgrid(0:dx:L,0:dy:W);

% Add source term
sourcefun='dont'; % add source term
switch sourcefun
    case 'add';     S = @(w) 0.1*w.^2;
    case 'dont';    S = @(w) zeros(size(w));
end

% Build IC
u0 = sin(pi*x).*sin(pi*y);

% Build Exact solution
uE = exp(-2*D*tFinal*pi^2)*sin(pi*x).*sin(pi*y);

% Set Initial time step
dt0 = 1/(2*D*(1/dx^2+1/dy^2)); % stability condition

% Set plot region
region = [0,L,0,W,-0.2,0.2]; 

%% Solver Loop 
% load initial conditions 
t=dt0; it=0; u=u0; dt=dt0;
 
while t < tFinal
    % RK stages
    uo=u;

    % forward euler solver
    u = Laplace2d(uo,nx+1,ny+1,Dx,Dy,S,dt);
    
    % set BCs
    u(1,:) = 0; u(nx+1,:) = 0;
    u(:,1) = 0; u(:,ny+1) = 0;

    % compute time step
    if t+dt>tFinal, dt=tFinal-t; end; 
    
    % Update iteration counter and time
    it=it+1; t=t+dt;
    
    % plot solution
    if mod(it,100); surf(x,y,u); axis([0,L,0,W,-1,1]); drawnow; end
end
 
%% % Post Process 
% Final Plot
subplot(121); h=surf(x,y,u); axis(region);
title('heat3d, Cell Averages','interpreter','latex','FontSize',18);
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{y}$','interpreter','latex','FontSize',14);
subplot(122); q=surf(x,y,uE); axis(region);
title('heat3d, Exact solution','interpreter','latex','FontSize',18);
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{y}$','interpreter','latex','FontSize',14);

% Error norms
err = abs(uE(:)-u(:));
L1 = dx*dy*norm(err,1); fprintf('L_1 norm: %1.2e \n',L1);
L2 = dx*dy*norm(err,2); fprintf('L_1 norm: %1.2e \n',L2);
Linf = norm(err,inf); fprintf('L_inf norm: %1.2e \n',Linf);