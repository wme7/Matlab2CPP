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
function [L1,Linf] = heat1dTest(nx,tFinal) 
%% Parameters
D = 1.0; % alpha
%tFinal = 0.1;	% End time
L = 2; 
%nx = 32; 
dx = L/(nx-1); 
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

%% Solver Loop 
% load initial conditions 
t=dt0; it=0; u=u0; dt=dt0;
 
while t < tFinal
    
    % RK stages
    uo=u;

    % forward euler solver
    u = Laplace1d(uo,nx,Dx,S,dt);
    
    % set BCs
    u(1) = 0; u(nx) = 0;

    % compute time step
    if t+dt>tFinal, dt=tFinal-t; end; 
    
    % Update iteration counter and time
    it=it+1; t=t+dt;
    
end

% Error norms
err = abs(uE(:)-u(:));
L1 = dx*sum(abs(err)); fprintf('L_1 norm: %1.2e \n',L1);
L2 = (dx*sum(err.^2))^0.5; fprintf('L_2 norm: %1.2e \n',L2);
Linf = norm(err,inf); fprintf('L_inf norm: %1.2e \n',Linf);