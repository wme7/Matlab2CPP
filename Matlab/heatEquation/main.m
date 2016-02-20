
% Heat Equation 
% Matlab Prototype solver
% by Manuel A. Diaz, NTU, 2013.02.14

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

u0= zeros(NY*NX,1);

for j = 1:NY
    for i = 1:NX
        o = i+NX*(j-1);
        % but ...
        if (i==1),  u0(o) = 0.0; end
        if (j==1),  u0(o) = 0.0; end
        if (i==NX), u0(o) = 1.0; end
        if (j==NY), u0(o) = 1.0; end
    end
end

u=u0;

%figure(1); surf(reshape(u0,[NY,NX]));

for step=0:2:NO_STEPS
    if mod(step,100)==0, fprintf('Step %d of %d\n',step,NO_STEPS); end

    % Compute Laplace stencil
    un = Call_Laplace(u,KX,KY,NX,NY);

    % Compute Laplace stencil (again)
    u = Call_Laplace(un,KX,KY,NX,NY);
    
    % update 
    %u = un;
end

figure(2); surf(reshape(u,[NY,NX]));
