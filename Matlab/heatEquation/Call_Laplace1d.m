function un = Call_Laplace1d(u,KX,NX)

un= zeros(NX,1);

for i = 1:NX
    
    o =   i  ; % node( j,i )
    r = (i+1); % node(j-1,i)  l--o--r
    l = (i-1); % node(j,i-1)
    
    if (i>1 && i<NX)
        un(o) = u(o) + KX*(u(r)-2*u(o)+u(l));
    else
        un(o) = u(o);
    end
end
