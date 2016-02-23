function u0 = Set_IC(NY,NX,u0)

for j = 1:NY
    for i = 1:NX
        o=i+NX*(j-1); u0(o)=0.0;
        % but ...
        if (i==1),  u0(o) = 0.0; end
        if (j==1),  u0(o) = 0.0; end
        if (i==NX), u0(o) = 1.0; end
        if (j==NY), u0(o) = 1.0; end
    end
end