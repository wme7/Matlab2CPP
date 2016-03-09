
% Heat Equation 2d
% Matlab Prototype for OMP solver
% by Manuel A. Diaz, NTU, 2013.02.14

clear; % all

NX =64; % number of cells in the x-direction
NY =64; % number of cells in the y-direction
L =1.0; % domain length
W =1.0; % domain width
C =1.0; % c, material conductivity. Uniform assumption.
TEND =1.0; % tEnd, output time
DX =(L/NX); % dx, cell size
DY =(W/NY); % dy, cell size
DT =(1/(2*C*(1/DX/DX+1/DY/DY))); % dt, fix time step size
KX =(C*DT/(DX*DX)); % numerical conductivity
KY =(C*DT/(DY*DY)); % numerical conductivity
NO_STEPS =(TEND/DT); % No. of time steps
XGRID =4; % No. of subdomains in the x-direction
YGRID =1; % No. of subdomains in the y-direction
OMP_THREADS =XGRID*YGRID; % No. of OMP threads
SNX =(NX/XGRID); % subregion size
SNY =(NY/YGRID); % subregion size

% Build Domain
u = zeros((NY+2)*(NX+2),1);

% Set IC
for i = 1:NX+2, u(   i  +(NX+2)*(   1-1))=0.0; end % down
for j = 1:NY+2, u(   1  +(NX+2)*(   j-1))=0.0; end % left
for i = 1:NX+2, u(   i  +(NX+2)*(NY+2-1))=1.0; end % up
for j = 1:NY+2, u((NX+2)+(NX+2)*(   j-1))=1.0; end % right

%figure(1); surf(reshape(u,[NX+2,NY+2]))

% Build OMP threads
tid = 0:OMP_THREADS-1;
u0 = zeros((SNY+2)*(SNX+2),1);
u1 = zeros((SNY+2)*(SNX+2),1);
u2 = zeros((SNY+2)*(SNX+2),1);
u3 = zeros((SNY+2)*(SNX+2),1);

% Set IC, u = u#
u0 = Set_IC_MultiDomain2d(0,u0,SNY,SNX); 
u1 = Set_IC_MultiDomain2d(1,u1,SNY,SNX); 
u2 = Set_IC_MultiDomain2d(2,u2,SNY,SNX); 
u3 = Set_IC_MultiDomain2d(3,u3,SNY,SNX); 

%%figure(1); surf(reshape(u0,[SNX+2,SNY+2]))

tic;
for step=0:2:NO_STEPS
    if mod(step,100)==0, fprintf('Step %d of %d\n',step,NO_STEPS); end

    % Communicate boundaries
    [u,u0] = Call_Comms2d(1,0,u,u0,SNY,SNX,NY,NX);
    [u,u1] = Call_Comms2d(1,1,u,u1,SNY,SNX,NY,NX);
    [u,u2] = Call_Comms2d(1,2,u,u2,SNY,SNX,NY,NX); 
    [u,u3] = Call_Comms2d(1,3,u,u3,SNY,SNX,NY,NX);
    
    [u,u0] = Call_Comms2d(2,0,u,u0,SNY,SNX,NY,NX);
    [u,u1] = Call_Comms2d(2,1,u,u1,SNY,SNX,NY,NX);
    [u,u2] = Call_Comms2d(2,2,u,u2,SNY,SNX,NY,NX);
    [u,u3] = Call_Comms2d(2,3,u,u3,SNY,SNX,NY,NX);
       
    % Compute Laplace stencil
    u0n = Call_Laplace2d(u0,KX,KY,SNX+2,SNY+2);
    u1n = Call_Laplace2d(u1,KX,KY,SNX+2,SNY+2);
    u2n = Call_Laplace2d(u2,KX,KY,SNX+2,SNY+2);
    u3n = Call_Laplace2d(u3,KX,KY,SNX+2,SNY+2);

    % Communicate boundaries
    [u,u0n] = Call_Comms2d(1,0,u,u0n,SNY,SNX,NY,NX);
    [u,u1n] = Call_Comms2d(1,1,u,u1n,SNY,SNX,NY,NX);
    [u,u2n] = Call_Comms2d(1,2,u,u2n,SNY,SNX,NY,NX);
    [u,u3n] = Call_Comms2d(1,3,u,u3n,SNY,SNX,NY,NX);
    
    [u,u0n] = Call_Comms2d(2,0,u,u0n,SNY,SNX,NY,NX);
    [u,u1n] = Call_Comms2d(2,1,u,u1n,SNY,SNX,NY,NX);
    [u,u2n] = Call_Comms2d(2,2,u,u2n,SNY,SNX,NY,NX);
    [u,u3n] = Call_Comms2d(2,3,u,u3n,SNY,SNX,NY,NX);
    
    % Compute Laplace stencil (again)
    u0 = Call_Laplace2d(u0n,KX,KY,SNX+2,SNY+2);
    u1 = Call_Laplace2d(u1n,KX,KY,SNX+2,SNY+2);
    u2 = Call_Laplace2d(u2n,KX,KY,SNX+2,SNY+2);
    u3 = Call_Laplace2d(u3n,KX,KY,SNX+2,SNY+2);
    
end
disp(toc);

% Call update domain
u = Call_Update2d(0,u,u0,SNY,SNX,NY,NX);
u = Call_Update2d(1,u,u1,SNY,SNX,NY,NX);
u = Call_Update2d(2,u,u2,SNY,SNX,NY,NX);
u = Call_Update2d(3,u,u3,SNY,SNX,NY,NX);

disp(size(u));

figure(2); surf(reshape(u,[NX+2,NY+2]));
