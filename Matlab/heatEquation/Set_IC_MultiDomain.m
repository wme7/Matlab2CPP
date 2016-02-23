function u0 = Set_IC_MultiDomain(tid,u0,SNY,SNX,OMP_THREADS)

if (tid==0)
    for j = 2:SNY-1
        for i = 2:SNX-1
            u0(i+SNX*(j-1)) = 0.25;
            % but ...
            if (i==2),     u0(i+SNX*(j-1)) = 0.0; end
            if (j==2),     u0(i+SNX*(j-1)) = 0.0; end
            if (j==SNY-1), u0(i+SNX*(j-1)) = 1.0; end
        end
    end
end
if (tid>0 && tid<OMP_THREADS)
    for j = 2:SNY-1
        for i = 2:SNX-1
            u0(i+SNX*(j-1)) = 0.50;
            % but ...
            if (j==2),     u0(i+SNX*(j-1)) = 0.0; end
            if (j==SNY-1), u0(i+SNX*(j-1)) = 1.0; end
        end
    end
end
if (tid==OMP_THREADS-1)
    for j = 2:SNY-1
        for i = 2:SNX-1
            u0(i+SNX*j) = 0.75;
            % but ...
            if (j==2),     u0(i+SNX*(j-1)) = 0.0; end
            if (i==SNX-1), u0(i+SNX*(j-1)) = 1.0; end
            if (j==SNY-1), u0(i+SNX*(j-1)) = 1.0; end
        end
    end
end