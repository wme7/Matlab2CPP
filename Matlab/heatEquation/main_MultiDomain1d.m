
% Heat Equation 
% Matlab Prototype solver
% by Manuel A. Diaz, NTU, 2013.02.14

clear; %all

NX =1024; % number of cells in the x-direction
L =1.0; % domain length
C =1.0; % c, material conductivity. Uniform assumption.
TEND =0.02; % tEnd, output time
DX =(L/NX); % dx, cell size
DT =(1/(2*C*(1/DX/DX))); % dt, fix time step size
KX =(C*DT/(DX*DX)); % numerical conductivity
NO_STEPS =(TEND/DT); % No. of time steps
XGRID =4; % No. of subdomains in the x-direction
OMP_THREADS =XGRID; % No. of OMP threads
SNX =(NX/XGRID); % subregion size

% Build Domain
u = zeros(NX+2,1);

% Build initial condition
u(1)=0; u(NX+2)=1;

% Build OMP threads
tid = 0:OMP_THREADS-1;
u0 = zeros(SNX+2,1);
u1 = zeros(SNX+2,1);
u2 = zeros(SNX+2,1);
u3 = zeros(SNX+2,1);

% Set IC, u = u#
u0 = Set_IC_MultiDomain1d(0,u0,SNX);
u1 = Set_IC_MultiDomain1d(1,u1,SNX);
u2 = Set_IC_MultiDomain1d(2,u2,SNX);
u3 = Set_IC_MultiDomain1d(3,u3,SNX);

tic
for step=0:2:NO_STEPS
    if mod(step,100)==0, fprintf('Step %d of %d\n',step,NO_STEPS); end

    % Communicate boundaries
	[u,u0]=Call_Comms1d(0,u,u0,SNX,NX,OMP_THREADS);
    [u,u1]=Call_Comms1d(1,u,u1,SNX,NX,OMP_THREADS);
    [u,u2]=Call_Comms1d(2,u,u2,SNX,NX,OMP_THREADS);
    [u,u3]=Call_Comms1d(3,u,u3,SNX,NX,OMP_THREADS);
    
    % Synchthreads!
    %plot([u0;u1;u2;u3],'o'); drawnow;
    
    % Compute Laplace stencil
    u0n=Call_Laplace1d(u0,KX,SNX+2);
    u1n=Call_Laplace1d(u1,KX,SNX+2);
    u2n=Call_Laplace1d(u2,KX,SNX+2);
    u3n=Call_Laplace1d(u3,KX,SNX+2);    
    
    % Communicate boundaries
	[u,u0n]=Call_Comms1d(0,u,u0n,SNX,NX,OMP_THREADS);
    [u,u1n]=Call_Comms1d(1,u,u1n,SNX,NX,OMP_THREADS);
    [u,u2n]=Call_Comms1d(2,u,u2n,SNX,NX,OMP_THREADS);
    [u,u3n]=Call_Comms1d(3,u,u3n,SNX,NX,OMP_THREADS);
    
    % Synchthreads!
    
    % Compute Laplace stencil
    u0=Call_Laplace1d(u0n,KX,SNX+2);
    u1=Call_Laplace1d(u1n,KX,SNX+2);
    u2=Call_Laplace1d(u2n,KX,SNX+2);
    u3=Call_Laplace1d(u3n,KX,SNX+2);
    
end
disp(toc);

% Call update domain
u = Call_Update1d(0,u,u0n,SNX,NX);
u = Call_Update1d(1,u,u1n,SNX,NX);
u = Call_Update1d(2,u,u2n,SNX,NX);
u = Call_Update1d(3,u,u3n,SNX,NX);

% plot
plot(u,'o'); axis([0,NX+2,0,1]);