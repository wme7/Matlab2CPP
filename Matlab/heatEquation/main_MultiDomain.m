
% Heat Equation 
% Matlab Prototype solver
% by Manuel A. Diaz, NTU, 2013.02.14

clear all;

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
SNX =(NX/XGRID)+2; % subregion size + BC cells
SNY =(NY/YGRID)+2; % subregion size + BC cells

% Build Domain
u = zeros(NY*NX,1);

% Build OMP threads
tid = 0:OMP_THREADS-1;
u0 = zeros(SNY*SNX,1);
u1 = zeros(SNY*SNX,1);
u2 = zeros(SNY*SNX,1);
u3 = zeros(SNY*SNX,1);

% Set IC, u = u#
u0 = Set_IC_MultiDomain(0,u0,SNY,SNX,OMP_THREADS); 
u1 = Set_IC_MultiDomain(1,u1,SNY,SNX,OMP_THREADS); 
u2 = Set_IC_MultiDomain(2,u2,SNY,SNX,OMP_THREADS); 
u3 = Set_IC_MultiDomain(3,u3,SNY,SNX,OMP_THREADS); 

%figure(1); surf(reshape(u0,[SNX,SNY]))

% for step=0:2:NO_STEPS
%     if mod(step,100)==0, fprintf('Step %d of %d\n',step,NO_STEPS); end
% 
%     % Compute Laplace stencil
%     un = Call_Laplace(u,KX,KY,NX,NY);
% 
%     % Compute Laplace stencil (again)
%     u = Call_Laplace(un,KX,KY,NX,NY);
%     
%     % update 
%     %u = un;
% end

% Call update domain
u = Call_Update(0,u,u0,SNY,SNX,NY,NX);
u = Call_Update(1,u,u1,SNY,SNX,NY,NX);
u = Call_Update(2,u,u2,SNY,SNX,NY,NX);
u = Call_Update(3,u,u3,SNY,SNX,NY,NX);

disp(size(u));

figure(2); surf(reshape(u,[NY,NX]));