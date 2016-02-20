function un = Call_Laplace(u,KX,KY,NX,NY)

un= zeros(NY*NX,1);

for j = 1:NY
    for i = 1:NX
        
        o = i+NX*(j-1); % node( j,i )     n
        n = o+NX;       % node(j+1,i)     |
        s = o-NX;       % node(j-1,i)  w--o--e
        e = o+1;        % node(j,i+1)     |
        w = o-1;        % node(j,i-1)     s
        
        if (i>1 && i<NX) && (j>1 && j<NY)
            un(o) = u(o) + KX*(u(e)-2*u(o)+u(w)) + KY*(u(n)-2*u(o)+u(s));
        else 
            un(o) = u(o);
        end
    end
end
