function un = Laplace2d(u,nx,ny,Dx,Dy,S,dt)

% set the shape of un
un=zeros(size(u));

for j = 1:ny
    for i = 1:nx
        if(i>1 && i<nx && j>1 && j<ny)
            un(i,j)= u(i,j) ...
                + dt*Dx*(u(i+1,j) - 2*u(i,j) + u(i-1,j)) ...
                + dt*Dy*(u(i,j+1) - 2*u(i,j) + u(i,j-1));
        else
            un(i,j)= u(i,j);
        end
    end
end
