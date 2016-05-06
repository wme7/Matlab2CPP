function un = Laplace1d(u,nx,Dx,S,dt)

% set the shape of un
un=zeros(size(u));

for i = 1:nx
    if(i>1 && i<nx)
        un(i)= u(i) + dt*Dx*(u(i+1) - 2*u(i) + u(i-1));
    else
        un(i)= u(i);
    end
end
