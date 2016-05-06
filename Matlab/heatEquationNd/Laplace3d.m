function un = Laplace3d(u,nx,ny,nz,Dx,Dy,Dz,S,dt)

% set the shape of un
un=zeros(size(u));

for k = 1:nz
    for j = 1:ny
        for i = 1:nx
            
            %o = i + nx*j + xy*k; % node( j,i,k )      n  b
            %n = o + nx;          % node(j+1,i,k)      | /
            %s = o - nx;          % node(j-1,i,k)      |/
            %e = o + 1;           % node(j,i+1,k)  w---o---e
            %w = o - 1;           % node(j,i-1,k)     /|
            %t = o + xy;          % node(j,i,k+1)    / |
            %b = o - xy;          % node(j,i,k-1)   t  s
            
            if(i>1 && i<nx && j>1 && j<ny && k>1 && k<nz)
                un(i,j,k)= u(i,j,k) ...
                    + dt*Dx*(u(i+1,j,k) - 2*u(i,j,k) + u(i-1,j,k)) ...
                    + dt*Dy*(u(i,j+1,k) - 2*u(i,j,k) + u(i,j-1,k)) ...
                    + dt*Dz*(u(i,j,k+1) - 2*u(i,j,k) + u(i,j,k-1));
            else
                un(i,j,k)= u(i,j,k);
            end
        end
    end
end
