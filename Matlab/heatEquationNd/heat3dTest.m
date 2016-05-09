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
function [L1,Linf] = heat3dTest(nx,ny,nz,tFinal)
%% Parameters
D = 1.0; % alpha
%tFinal = 0.1;	% End time
L = 1; 
%nx = 32; 
dx = L/(nx-1); 
W = 1; 
%ny = 32; 
dy = W/(ny-1);
H = 1; 
%nz = 64; 
dz = H/(nz-1);
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

%% Solver Loop 
% load initial conditions 
t=0; it=0; u=u0; dt=dt0;
 
while t < tFinal
    % RK stages
    uo=u;

    % forward euler solver
    u = Laplace3d(uo,nx,ny,nz,Dx,Dy,Dz,S,dt);
    
    % set BCs
    u(1,:,:) = 0; u(nx,:,:) = 0;
    u(:,1,:) = 0; u(:,ny,:) = 0;
    u(:,:,1) = 0; u(:,:,nz) = 0;

    % compute time step
    if t+dt>tFinal, dt=tFinal-t; end; 
    
    % Update iteration counter and time
    it=it+1; t=t+dt;
    
end

% Error norms
err = abs(uE(:)-u(:));
L1 = dx*dy*dz*sum(abs(err)); fprintf('L_1 norm: %1.2e \n',L1);
L2 = (dx*dy*dz*sum(err.^2))^0.5; fprintf('L_2 norm: %1.2e \n',L2);
Linf = norm(err,inf); fprintf('L_inf norm: %1.2e \n',Linf);