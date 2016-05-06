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
Dx = D/dx^2; 

% Build Numerical Mesh
x = 0:dx:L;

% Add source term
sourcefun='dont'; % add source term
switch sourcefun
    case 'add';     S = @(w) 0.1*w.^2;
    case 'dont';    S = @(w) zeros(size(w));
end

% Build IC
u0 = sin(pi*x);

% Build Exact solution
uE = exp(-D*tFinal*pi^2)*sin(pi*x);

% Set Initial time step
dt0 = 1/(2*D*(1/dx^2)); % stability condition

% Set plot region
region = [0,L,-0.5,0.5]; 

%% Solver Loop 
% load initial conditions 
t=dt0; it=0; u=u0; dt=dt0;
 
while t < tFinal
    
    % RK stages
    uo=u;

    % forward euler solver
    u = Laplace1d(uo,nx+1,Dx,S,dt);
    
    % set BCs
    u(1) = 0; u(nx+1) = 0;

    % compute time step
    if t+dt>tFinal, dt=tFinal-t; end; 
    
    % Update iteration counter and time
    it=it+1; t=t+dt;
    
    % plot solution
    if mod(it,100); plot(x,u,'.b'); axis([0,L,-1,1]); drawnow; end
end
 
%% % Post Process 
% Final Plot
h=plot(x,u,'.b',x,uE,'-r'); axis(region);
title('heat1d, Cell Averages','interpreter','latex','FontSize',18);
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{u(x)}$','interpreter','latex','FontSize',14);
legend('Jacobi Method','Exact Solution');

% Error norms
err = abs(uE(:)-u(:));
L1 = dx*norm(err,1); fprintf('L_1 norm: %1.2e \n',L1);
L2 = dx*norm(err,2); fprintf('L_1 norm: %1.2e \n',L2);
Linf = norm(err,inf); fprintf('L_inf norm: %1.2e \n',Linf);